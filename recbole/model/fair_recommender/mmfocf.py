# _*_ coding: utf-8 _*_
# @Time   : 2022/3/8
# @Author : Jiakai Tang
# @Email  : whut_tangjiakai@qq.com

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from recbole.model.init import xavier_normal_initialization
from recbole.model.abstract_recommender import FairRecommender
from recbole.utils import InputType

class MMFOCF(FairRecommender):
    r""" FOCF: Fairness-Oriented Collaborative Filtering 
    Enhanced with Multimodal Concatenation and Double Projection.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(MMFOCF, self).__init__(config, dataset)

        # load dataset info
        self.embedding_size = config['embedding_size']
        self.RATING = config['RATING_FIELD']
        self.SST_FIELD = config['sst_attr_list'][0]
        self.fair_weight = config['fair_weight']
        self.max_rating = dataset.inter_feat[self.RATING].max()

        # define layers
        self.user_embedding_layer = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding_layer = nn.Embedding(self.n_items, self.embedding_size)
        
        # load multimodal data
        minilm_emb = dataset.get_preload_weight('minilm_iid')
        self.text_emb = nn.Embedding.from_pretrained(torch.from_numpy(minilm_emb).float(), freeze=True)
        vit_emb = dataset.get_preload_weight('vit_iid')
        self.image_emb = nn.Embedding.from_pretrained(torch.from_numpy(vit_emb).float(), freeze=True)

        # 1. First Projection: Merge Text + Image
        self.mm_raw_dim = self.text_emb.embedding_dim + self.image_emb.embedding_dim
        print(self.mm_raw_dim)
        self.mm_projection = nn.Linear(self.mm_raw_dim, self.embedding_size)

        # 2. Second Projection: Merge (CF Embedding || Projected MM)
        # Input: embedding_size (from CF) + embedding_size (from mm_projection)
        self.final_item_projection = nn.Linear(self.embedding_size * 2, self.embedding_size)

        self.rating_loss_fun = nn.MSELoss()
        self.fair_loss_fun = self.get_loss_fun(config['fair_objective'])

        self.apply(xavier_normal_initialization)

    def get_loss_fun(self, fair_objective):
        fair_objective = fair_objective.strip().lower()
        mapping = {
            'none': None,
            'value': self.value_unfairness,
            'absolute': self.absolute_unfairness,
            'under': self.under_unfairness,
            'over': self.over_unfairness,
            'nonparity': self.nonparity_unfairness
        }
        return mapping.get(fair_objective, None)

    def get_combined_item_embedding(self, item_indices):
        # Step 1: Get CF latent vector
        cf_e = self.item_embedding_layer(item_indices)

        # Step 2: Concat Raw Multimodal and Project to latent space
        t_e = self.text_emb(item_indices)
        v_e = self.image_emb(item_indices)
        mm_raw = torch.cat([t_e, v_e], dim=-1)
        mm_projected = F.leaky_relu(self.mm_projection(mm_raw))

        # Step 3: Concat CF and MM, then Project again to final embedding size
        combined_concat = torch.cat([cf_e, mm_projected], dim=-1)
        final_item_e = self.final_item_projection(combined_concat)
        
        return final_item_e

    def forward(self, user, item):
        user_e = self.user_embedding_layer(user)
        item_e = self.get_combined_item_embedding(item)
        
        pred_scores = torch.mul(user_e, item_e).sum(dim=-1)
        return pred_scores, user_e, item_e

    # --- Unfairness Logic ---
    def get_item_ratings(self, pred_scores, interaction):
        sst_inverse = torch.unique(interaction[self.SST_FIELD], return_inverse=True)[1]
        iid_unique_value, iid_inverse = torch.unique(interaction[self.ITEM_ID], return_inverse=True)
        
        avg_pred_list = torch.zeros((len(iid_unique_value), 2), device=self.device)
        sst_num = torch.zeros((len(iid_unique_value), 2), device=self.device)
        avg_true_list = torch.zeros((len(iid_unique_value), 2), device=self.device)

        index = (iid_inverse, sst_inverse)
        avg_pred_list.index_put_(index, pred_scores, accumulate=True)
        avg_true_list.index_put_(index, interaction[self.RATING], accumulate=True)
        sst_num.index_put_(index, torch.ones(len(pred_scores), device=self.device), accumulate=True)
        sst_num += 1e-5

        return avg_pred_list / sst_num, avg_true_list / sst_num

    def value_unfairness(self, pred_scores, interaction):
        avg_pred, avg_true = self.get_item_ratings(pred_scores, interaction)
        diff = avg_pred - avg_true 
        return F.smooth_l1_loss(torch.abs(diff[:,0] - diff[:,1]), torch.zeros_like(diff[:,0]))

    def absolute_unfairness(self, pred_scores, interaction):
        avg_pred, avg_true = self.get_item_ratings(pred_scores, interaction)
        diff = torch.abs(avg_pred - avg_true)
        return F.smooth_l1_loss(torch.abs(diff[:,0] - diff[:,1]), torch.zeros_like(diff[:,0]))

    def under_unfairness(self, pred_scores, interaction):
        avg_pred, avg_true = self.get_item_ratings(pred_scores, interaction)
        diff = torch.clamp(avg_true - avg_pred, min=0.)
        return F.smooth_l1_loss(torch.abs(diff[:,0] - diff[:,1]), torch.zeros_like(diff[:,0]))

    def over_unfairness(self, pred_scores, interaction):
        avg_pred, avg_true = self.get_item_ratings(pred_scores, interaction)
        diff = torch.clamp(avg_pred - avg_true, min=0.)
        return F.smooth_l1_loss(torch.abs(diff[:,0] - diff[:,1]), torch.zeros_like(diff[:,0]))

    def nonparity_unfairness(self, pred_scores, interaction):
        sst_values = interaction[self.SST_FIELD]
        sst_unique = torch.unique(sst_values)
        avg_1 = pred_scores[sst_values == sst_unique[0]].mean()
        avg_2 = pred_scores[sst_values == sst_unique[1]].mean()
        return F.smooth_l1_loss(avg_1, avg_2)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        pred_scores, _, _ = self.forward(user, item)
        return torch.clamp(pred_scores, min=0., max=self.max_rating) / self.max_rating

    def calculate_loss(self, interaction):
        pred_scores, _, _ = self.forward(interaction[self.USER_ID], interaction[self.ITEM_ID])
        rating_loss = self.rating_loss_fun(pred_scores, interaction[self.RATING])

        fair_loss = 0.
        if self.fair_loss_fun:
            fair_loss = self.fair_loss_fun(pred_scores, interaction)

        return rating_loss + self.fair_weight * fair_loss

    def full_sort_predict(self, interaction):
        user_e = self.user_embedding_layer(interaction[self.USER_ID])
        all_item_indices = torch.arange(self.n_items, device=self.device)
        all_item_e = self.get_combined_item_embedding(all_item_indices)
        
        pred_scores = torch.mm(user_e, all_item_e.t())
        return torch.clamp(pred_scores, min=0., max=self.max_rating) / self.max_rating