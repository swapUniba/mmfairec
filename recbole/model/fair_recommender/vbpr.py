import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import FairRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

class VBPR(FairRecommender):
    r"""VBPR: Visual Bayesian Personalized Ranking
    This implementation follows the 'concatenation' version where ID and 
    visual features are fused before the dot product.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(VBPR, self).__init__(config, dataset)

        # Load parameters
        self.embedding_size = config['embedding_size']
        self.weight_decay = config['weight_decay']

        # 1. Define Embeddings
        # User embedding is twice the size to match [ID_lat | Visual_lat]
        self.u_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.n_users, self.embedding_size * 2))
        )
        # Item latent ID embedding
        self.i_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.n_items, self.embedding_size))
        )

        # 2. Multimodal Feature Loading
        # Get pre-loaded features from the dataset
        v_feat = dataset.get_preload_weight('vit_iid')
        t_feat = dataset.get_preload_weight('minilm_iid')

        # Concatenate features if both exist, otherwise use available one
        if v_feat is not None and t_feat is not None:
            raw_features = torch.cat((torch.from_numpy(t_feat), torch.from_numpy(v_feat)), -1)
        elif v_feat is not None:
            raw_features = torch.from_numpy(v_feat)
        else:
            raw_features = torch.from_numpy(t_feat)
        
        # Register as buffer so it moves to GPU but isn't a trainable parameter
        self.register_buffer('item_raw_features', raw_features.float())

        # 3. Visual Projection Layer
        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.embedding_size)

        # 4. Loss Functions
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # Init
        self.apply(xavier_normal_initialization)

    def forward(self, dropout=0.0):
        # Project multimodal features to latent space
        item_visual_embeddings = self.item_linear(self.item_raw_features)
        
        # Final Item Representation: [ID_latent | Visual_projected]
        # Shape: [n_items, embedding_size * 2]
        item_e = torch.cat((self.i_embedding, item_visual_embeddings), dim=-1)

        user_e = F.dropout(self.u_embedding, p=dropout, training=self.training)
        item_e = F.dropout(item_e, p=dropout, training=self.training)
        
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Get the full projected matrices
        user_all, item_all = self.forward()

        # Slice for the specific batch
        u_e = user_all[user]
        p_e = item_all[pos_item]
        n_e = item_all[neg_item]

        # Pairwise Ranking Score
        pos_item_score = torch.mul(u_e, p_e).sum(dim=1)
        neg_item_score = torch.mul(u_e, n_e).sum(dim=1)

        # BPR + L2 Regularization
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(u_e, p_e, n_e)
        
        return mf_loss + self.weight_decay * reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        user_all, item_all = self.forward()
        u_e = user_all[user]
        i_e = item_all[item]
        
        return torch.mul(u_e, i_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_all, item_all = self.forward()
        
        u_e = user_all[user]
        # Multiplies batch of users by all items
        score = torch.matmul(u_e, item_all.transpose(0, 1))
        return score