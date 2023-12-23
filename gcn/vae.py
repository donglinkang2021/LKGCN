import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RatingModel(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super(RatingModel, self).__init__()
        self.user_embed = nn.Embedding(num_embeddings=n_users, embedding_dim=n_factors)
        self.item_embed = nn.Embedding(num_embeddings=n_items, embedding_dim=n_factors)

        for param in self.parameters():
            nn.init.normal_(param, std=0.02)

    def forward(self, user_id, item_id = None):
        """
        @param user: torch.LongTensor of shape (batch_size, )
        @param item: torch.LongTensor of shape (batch_size, )
        @return scores: torch.FloatTensor of shape (batch_size, n_items)
        """
        P_u = self.user_embed(user_id)
        if item_id is not None:
            Q_i = self.item_embed(item_id)
            return torch.matmul(P_u, Q_i.t())
        else:
            Q_i = self.item_embed.weight
        return torch.matmul(P_u, Q_i.t())