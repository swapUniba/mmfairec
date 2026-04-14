import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import FairRecommender
from recbole.model.layers import MLPLayers
from recbole.utils import InputType

class MMNFCF(FairRecommender):
    r""" MMNFCF: Neural Fair Collaborative Filtering 
    Enhanced with Multimodal Concatenation and Double Projection.
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(MMNFCF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']
        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout = config['dropout']
        self.sst_attr = config['sst_attr_list'][0]
        self.fair_weight = config['fair_weight']
        load_pretrain = config['load_pretrain']
        self.load_pretrain_path = config['load_pretrain_path']

        # 1. Define Standard Layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding_layer = nn.Embedding(self.n_items, self.embedding_size)

        # 2. Load Multimodal Data (with .float() fix)
        minilm_emb = dataset.get_preload_weight('minilm_iid')
        self.text_emb = nn.Embedding.from_pretrained(torch.from_numpy(minilm_emb).float(), freeze=True)
        vit_emb = dataset.get_preload_weight('vit_iid')
        self.image_emb = nn.Embedding.from_pretrained(torch.from_numpy(vit_emb).float(), freeze=True)

        # 3. Multimodal Projection Layers (The "MMFOCF" logic)
        self.mm_raw_dim = self.text_emb.embedding_dim + self.image_emb.embedding_dim
        self.mm_projection = nn.Linear(self.mm_raw_dim, self.embedding_size)
        self.final_item_projection = nn.Linear(self.embedding_size * 2, self.embedding_size)

        # 4. Neural Architecture
        # The input to MLP is [user_e || fused_item_e], hence 2 * embedding_size
        self.mlp_layers = MLPLayers([2 * self.embedding_size] + self.mlp_hidden_size + [1], self.dropout)
        self.mlp_layers.logger = None 
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # parameters initialization
        if load_pretrain:
            self.reset_params(self.load_pretrain_path, dataset.get_user_feature()[1:])

    def reset_params(self, pretrain_path, user_data):
        # Note: Pretraining logic remains centered on user-side debiasing
        checkpoint = torch.load(pretrain_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'], strict=False)

        sst_value = user_data[self.sst_attr]
        sst_unique_value = torch.unique(sst_value)
        sst1_indices = sst_value == sst_unique_value[0]
        sst2_indices = sst_value == sst_unique_value[1]
        ncf_user_embedding = self.user_embedding.weight.data[1:].clone()

        sst_embedding1 = ncf_user_embedding[sst1_indices].mean(dim=0)
        sst_embedding2 = ncf_user_embedding[sst2_indices].mean(dim=0)
        
        diff = sst_embedding1 - sst_embedding2
        vector_bias = diff / torch.linalg.norm(diff, keepdim=True)
        vector_bias = torch.mul(ncf_user_embedding, vector_bias).sum(dim=1, keepdim=True) * vector_bias        
        user_embedding = ncf_user_embedding - vector_bias

        self.user_embedding.weight.data[1:] = user_embedding
        self.user_embedding.weight.requires_grad = False
        # Re-init item embedding to allow multimodal learning to lead
        self.item_embedding_layer = nn.Embedding(self.n_items, self.embedding_size)

    def get_combined_item_embedding(self, item_indices):
        # Step 1: CF Embedding
        cf_e = self.item_embedding_layer(item_indices)

        # Step 2: Multimodal Projection
        t_e = self.text_emb(item_indices)
        v_e = self.image_emb(item_indices)
        mm_raw = torch.cat([t_e, v_e], dim=-1)
        mm_projected = F.leaky_relu(self.mm_projection(mm_raw))

        # Step 3: Fusion
        combined_concat = torch.cat([cf_e, mm_projected], dim=-1)
        return self.final_item_projection(combined_concat)

    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.get_combined_item_embedding(item)
        
        # Concat for MLP input
        output = self.mlp_layers(torch.cat((user_e, item_e), dim=-1))
        return self.sigmoid(output.squeeze(-1))

    def get_differential_fairness(self, interaction, score):
        # Keeping the original NFCF Differential Fairness logic
        pos_idx = interaction[self.LABEL] == 1
        if not pos_idx.any():
            return torch.tensor(0.0, device=self.device)
            
        score = score[pos_idx]
        sst_unique_values, sst_indices = torch.unique(interaction[self.sst_attr][pos_idx], return_inverse=True)
        iid_unique_values, iid_indices = torch.unique(interaction[self.ITEM_ID][pos_idx], return_inverse=True)
        
        score_matrix = torch.zeros((len(iid_unique_values), len(sst_unique_values)), device=self.device)
        norm_matrix = torch.zeros((len(iid_unique_values), len(sst_unique_values)), device=self.device)
        
        concentration_parameter = 1.0
        dirichlet_alpha = concentration_parameter / len(iid_unique_values)

        score_matrix.index_put_((iid_indices, sst_indices), score, accumulate=True)
        norm_matrix.index_put_((iid_indices, sst_indices), torch.ones(len(sst_indices), device=self.device), accumulate=True)
        score_matrix = (score_matrix + dirichlet_alpha) / (norm_matrix + concentration_parameter)

        epsilon_values = torch.zeros(len(iid_unique_values), device=self.device)
        for i in range(len(sst_unique_values)):
            for j in range(i+1, len(sst_unique_values)):
                epsilon = torch.abs(torch.log(score_matrix[:, i]) - torch.log(score_matrix[:, j]))
                epsilon_values = torch.max(epsilon_values, epsilon)
        
        return epsilon_values.mean()

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.forward(user, item)
        rec_loss = self.loss(output, label)
        
        if self.load_pretrain_path is None:
            return rec_loss
            
        fair_loss = self.get_differential_fairness(interaction, output)
        return rec_loss + self.fair_weight * fair_loss

    def predict(self, interaction):
        return self.forward(interaction[self.USER_ID], interaction[self.ITEM_ID])