import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import FairRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

class FREEDOM(FairRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(FREEDOM, self).__init__(config, dataset)

        # Load hyperparameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.n_mm_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.weight_decay = config['weight_decay']
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout'] 

        self.n_nodes = self.n_users + self.n_items

        # 1. UI Graph Setup
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        # Ensure norm_adj is on the correct device
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values = self.edge_values.to(self.device)
        self.masked_adj = None

        # 2. Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # 3. Multimodal Features (Moved to device immediately)
        v_feat = dataset.get_preload_weight('vit_iid')
        t_feat = dataset.get_preload_weight('minilm_iid')
        
        if v_feat is not None:
            self.v_feat = torch.from_numpy(v_feat).to(self.device).float()
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim).to(self.device)
        if t_feat is not None:
            self.t_feat = torch.from_numpy(t_feat).to(self.device).float()
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim).to(self.device)

        # 4. Construct Item-Item Graph
        self.mm_adj = self.build_mm_adj().to(self.device)

    def get_norm_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        
        # Optimized dictionary update for sparse matrix
        rows, cols = zip(*data_dict.keys())
        values = list(data_dict.values())
        adj_mat = sp.coo_matrix((values, (rows, cols)), shape=(self.n_nodes, self.n_nodes)).todok()

        row_sum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(row_sum + 1e-7, -0.5)
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        
        return self._convert_sp_mat_to_sp_tensor(norm_adj)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        indices = torch.LongTensor(np.array([coo.row, coo.col]))
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(indices, data, torch.Size(coo.shape))

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).long() # Default CPU
        
        # Use CPU for initial calculation to avoid sparse tensor device mismatch, then move result
        adj = torch.sparse.FloatTensor(edges, torch.ones_like(edges[0]), torch.Size((self.n_users, self.n_items)))
        row_sum = torch.sparse.sum(adj, -1).to_dense() + 1e-7
        col_sum = torch.sparse.sum(adj.t(), -1).to_dense() + 1e-7
        values = (row_sum[edges[0]] * col_sum[edges[1]]).pow(-0.5)
        return edges, values

    def build_mm_adj(self):
        def get_knn(mm_emb):
            norm_emb = F.normalize(mm_emb, p=2, dim=-1)
            sim = torch.mm(norm_emb, norm_emb.t())
            _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
            
            # FIXED: Explicitly set device for arange
            indices0 = torch.arange(self.n_items, device=self.device).unsqueeze(1).expand(-1, self.knn_k).reshape(-1)
            indices1 = knn_ind.reshape(-1)
            indices = torch.stack([indices0, indices1], 0)
            
            # FIXED: Ensure ones are on the same device as indices
            adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0], device=self.device), torch.Size((self.n_items, self.n_items)))
            row_sum = torch.sparse.sum(adj, -1).to_dense() + 1e-7
            v = row_sum[indices[0]].pow(-0.5) * row_sum[indices[1]].pow(-0.5)
            return torch.sparse.FloatTensor(indices, v, torch.Size((self.n_items, self.n_items)))

        adj_v = get_knn(self.v_feat) if hasattr(self, 'v_feat') else None
        adj_t = get_knn(self.t_feat) if hasattr(self, 't_feat') else None
        
        if adj_v is not None and adj_t is not None:
            return self.mm_image_weight * adj_v + (1.0 - self.mm_image_weight) * adj_t
        return adj_v if adj_v is not None else adj_t

    def pre_epoch_processing(self):
        if self.dropout <= 0:
            self.masked_adj = self.norm_adj
            return

        keep_len = int(self.edge_values.size(0) * (1. - self.dropout))
        indices = torch.multinomial(self.edge_values, keep_len)
        
        keep_indices = self.edge_indices[:, indices]
        
        # FIXED: Ensure ones are on the same device
        adj = torch.sparse.FloatTensor(keep_indices, torch.ones_like(keep_indices[0], device=self.device), torch.Size((self.n_users, self.n_items)))
        row_sum = torch.sparse.sum(adj, -1).to_dense() + 1e-7
        col_sum = torch.sparse.sum(adj.t(), -1).to_dense() + 1e-7
        keep_values = (row_sum[keep_indices[0]] * col_sum[keep_indices[1]]).pow(-0.5)

        row = torch.cat([keep_indices[0], keep_indices[1] + self.n_users])
        col = torch.cat([keep_indices[1] + self.n_users, keep_indices[0]])
        all_indices = torch.stack([row, col])
        all_values = torch.cat([keep_values, keep_values])
        
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def forward(self, adj=None):
        # Fallback logic: if adj is None (e.g., during first call or evaluation), 
        # use the full normalized adjacency matrix.
        if adj is None:
            adj = self.norm_adj

        # 1. Item-Item Multimodal Propagation (Freezing)
        # Propagates item ID embeddings through the pre-built multimodal similarity graph
        h = self.item_id_embedding.weight
        for _ in range(self.n_mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        # 2. User-Item Collaborative Propagation (Denoising)
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        u_g, i_g = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        # Combine the collaborative signal with the multimodal structural signal
        return u_g, i_g + h

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Ensure we have a masked_adj; if not (e.g. first batch), use norm_adj
        train_adj = self.masked_adj if self.masked_adj is not None else self.norm_adj
        ua_emb, ia_emb = self.forward(train_adj)

        u_g = ua_emb[user]
        pos_i_g = ia_emb[pos_item]
        neg_i_g = ia_emb[neg_item]

        # Main BPR Loss
        pos_scores = torch.mul(u_g, pos_i_g).sum(dim=1)
        neg_scores = torch.mul(u_g, neg_i_g).sum(dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Multimodal Alignment Regularization
        reg_mm_loss = 0
        if hasattr(self, 't_feat'):
            t_feats = self.text_trs(self.t_feat)
            reg_mm_loss += -torch.mean(F.logsigmoid((u_g * t_feats[pos_item]).sum(1) - (u_g * t_feats[neg_item]).sum(1)))
        if hasattr(self, 'v_feat'):
            v_feats = self.image_trs(self.v_feat)
            reg_mm_loss += -torch.mean(F.logsigmoid((u_g * v_feats[pos_item]).sum(1) - (u_g * v_feats[neg_item]).sum(1)))

        return mf_loss + self.weight_decay * reg_mm_loss

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        u_emb, i_emb = self.forward(self.norm_adj)
        return torch.matmul(u_emb[user], i_emb.t())
    
    def predict(self, interaction):
        """
        Predict the score of specific user-item pairs.
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        u_g_embeddings, i_g_embeddings = self.forward(self.norm_adj)
        u_embeddings = u_g_embeddings[user]
        i_embeddings = i_g_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores