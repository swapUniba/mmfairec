import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import FairRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

class MMGCF(FairRecommender):
    r"""MMGCF implemented with equal weighting and concat fusion strategies.
    This model propagates ID embeddings through GCN layers and fuses them 
    with projected multimodal features (Visual + Text) for the final prediction.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MMGCF, self).__init__(config, dataset)

        # 1. Load Config & Dataset Info
        self.latent_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["weight_decay"]
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.require_pow = config["require_pow"]
        if self.require_pow is None:
            self.require_pow = False

        # 2. Define Trainable Embeddings (ID Branch)
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.latent_dim)

        # 3. Multimodal Data Loading (Content Branch)
        v_feat = dataset.get_preload_weight('vit_iid')
        t_feat = dataset.get_preload_weight('minilm_iid')
        
        # Merge Vision and Text features
        raw_mm_features = torch.cat([torch.from_numpy(t_feat), torch.from_numpy(v_feat)], dim=-1)
        self.register_buffer('item_mm_features', raw_mm_features.float())

        # 4. Projection Layers
        # Project raw multimodal features to latent ID dimension
        self.mm_projection = nn.Linear(self.item_mm_features.shape[1], self.latent_dim)
        # Late Fusion: Merge [GCN_Refined_ID || Projected_MM] -> Final Latent Space
        self.fusion_projection = nn.Linear(self.latent_dim * 2, self.latent_dim)

        # 5. Loss and Graph Setup
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # Parameters initialization
        self.apply(xavier_uniform_initialization)

    def get_norm_adj_mat(self):
        r"""Constructs the Laplacian normalized adjacency matrix: D^-0.5 * A * D^-0.5"""
        # Build symmetric adjacency matrix
        adj_shape = (self.n_users + self.n_items, self.n_users + self.n_items)
        A = sp.dok_matrix(adj_shape, dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        
        # User-Item interactions (top right) and Item-User interactions (bottom left)
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        
        for (row, col), value in data_dict.items():
            A[row, col] = value

        # Normalize matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D

        # Convert to Sparse Tensor
        L = sp.coo_matrix(L)
        indices = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(indices, data, torch.Size(L.shape))

    def get_ego_embeddings(self):
        """Initial embeddings before GCN propagation."""
        return torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)

    def forward(self):
        """GCN propagation followed by Multimodal Late Fusion."""
        # --- GCN Part ---
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        # Mean pooling over layers (LightGCN standard)
        lightgcn_all_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)
        user_id_e, item_id_e = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])

        # --- Multimodal Late Fusion Part ---
        # 1. Project Multimodal Features
        mm_projected = F.leaky_relu(self.mm_projection(self.item_mm_features))

        # 2. Fuse with GCN-refined item ID embeddings
        # We concatenate the structural ID signal with the content signal
        item_combined = torch.cat([item_id_e, mm_projected], dim=-1)
        final_item_e = self.fusion_projection(item_combined)

        return user_id_e, final_item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Get final representations
        user_all, item_all = self.forward()
        
        u_e = user_all[user]
        p_e = item_all[pos_item]
        n_e = item_all[neg_item]

        # 1. Ranking Loss (BPR)
        pos_scores = torch.mul(u_e, p_e).sum(dim=1)
        neg_scores = torch.mul(u_e, n_e).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # 2. L2 Regularization on raw ID ego-embeddings
        u_ego = self.user_embedding(user)
        p_ego = self.item_id_embedding(pos_item)
        n_ego = self.item_id_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego, p_ego, n_ego, require_pow=self.require_pow)

        return mf_loss + self.reg_weight * reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_all, item_all = self.forward()
        
        return torch.mul(user_all[user], item_all[item]).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_all, item_all = self.forward()
        
        u_e = user_all[user]
        return torch.matmul(u_e, item_all.transpose(0, 1))