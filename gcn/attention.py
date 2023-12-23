"""
这里来专门研究一下attention机制的model能否对GCN的效果有所提升
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention_DotProduct(queries, keys):
    """
    多元注意力回归

    Parameters
    ----------
    @param queries: torch.Tensor
        queries, shape (m, d)
    @param keys: torch.Tensor
        keys, shape (n, d)
    @returns: torch.Tensor
        attention_weights, shape (m, n)
    """
    d = queries.shape[-1]

    return F.softmax(
        torch.matmul(
            queries, 
            keys.T
        ) / math.sqrt(d), 
        dim=1
    )

class AttMFui(nn.Module):
    def __init__(self, n_users:int, m_items:int, embed_dim:int):
        super(AttMFui, self).__init__()
        self.user_embed = torch.nn.Embedding(
            num_embeddings = n_users, embedding_dim = embed_dim
        )
        self.item_embed = torch.nn.Embedding(
            num_embeddings = m_items, embedding_dim = embed_dim
        )
        self._init_weight()

    def _init_weight(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.01)
        
    def forward(self, user, item):
        """
        @param user: torch.LongTensor of shape (batch_size, )
        @param item: torch.LongTensor of shape (batch_size, )
        @return scores: torch.FloatTensor of shape (batch_size,)
        """
        P_u = self.user_embed(user)
        Q_i = self.item_embed(item)
        # 实现neighborhood aggregation
        # user -> item
        Q_i = torch.matmul(attention_DotProduct(Q_i, P_u), P_u)
        return torch.sum(P_u * Q_i, dim=1) 

# MSE:    
# AttMFui Max Recall@10: 0.0410 at epoch 2
# AttMFui Min RMSE: 1.0771 at epoch 2
# note: we can't use the AttMFui model to bpr task, for the loss will be constant
# because we just aggregate user info to item embedding
# which make no sense in bpr task (we use user embedding to predict item embedding)
    
class AttMFiu(nn.Module):
    def __init__(self, n_users:int, m_items:int, embed_dim:int):
        super(AttMFiu, self).__init__()
        self.user_embed = torch.nn.Embedding(
            num_embeddings = n_users, embedding_dim = embed_dim
        )
        self.item_embed = torch.nn.Embedding(
            num_embeddings = m_items, embedding_dim = embed_dim
        )
        self._init_weight()

    def _init_weight(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.01)
        
    def forward(self, user, item):
        """
        @param user: torch.LongTensor of shape (batch_size, )
        @param item: torch.LongTensor of shape (batch_size, )
        @return scores: torch.FloatTensor of shape (batch_size,)
        """
        P_u = self.user_embed(user)
        Q_i = self.item_embed(item)
        # 实现neighborhood aggregation
        # item -> user
        P_u = torch.matmul(attention_DotProduct(P_u, Q_i), Q_i)
        return torch.sum(P_u * Q_i, dim=1) 

# MSE:
# AttMFiu Max Recall@10: 0.0410 at epoch 1
# AttMFiu Min RMSE: 1.2542 at epoch 11
# BPR:
# AttMFiu Max Recall@10: 0.0410 at epoch 9
# note: in our mse task, as we only add the item info to the user embedding
# the recall@10 get worse and worse
    
class AttMFuu(nn.Module):
    def __init__(self, n_users:int, m_items:int, embed_dim:int):
        super(AttMFuu, self).__init__()
        self.user_embed = torch.nn.Embedding(
            num_embeddings = n_users, embedding_dim = embed_dim
        )
        self.item_embed = torch.nn.Embedding(
            num_embeddings = m_items, embedding_dim = embed_dim
        )
        self._init_weight()

    def _init_weight(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.01)
        
    def forward(self, user, item):
        """
        @param user: torch.LongTensor of shape (batch_size, )
        @param item: torch.LongTensor of shape (batch_size, )
        @return scores: torch.FloatTensor of shape (batch_size,)
        """
        P_u = self.user_embed(user)
        Q_i = self.item_embed(item)
        # 实现一次完整的neighborhood aggregation
        # user -> item
        Q_i_new = torch.matmul(attention_DotProduct(Q_i, P_u), P_u)
        # item -> user
        P_u_new = torch.matmul(attention_DotProduct(P_u, Q_i), Q_i)
        return torch.sum(P_u_new * Q_i_new, dim=1) 

# MSE:
# AttMFuu Max Recall@10: 0.0410 at epoch 1
# AttMFuu Min RMSE: 1.0265 at epoch 66    
# note: we can't also use the AttMFuu model to bpr task, for the loss will be constant
# as long as we add the user info to the item embedding at first, the bpr loss will be constant
    
class AttMFii(nn.Module):
    def __init__(self, n_users:int, m_items:int, embed_dim:int):
        super(AttMFii, self).__init__()
        self.user_embed = torch.nn.Embedding(
            num_embeddings = n_users, embedding_dim = embed_dim
        )
        self.item_embed = torch.nn.Embedding(
            num_embeddings = m_items, embedding_dim = embed_dim
        )
        self._init_weight()

    def _init_weight(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.01)
        
    def forward(self, user, item):
        """
        @param user: torch.LongTensor of shape (batch_size, )
        @param item: torch.LongTensor of shape (batch_size, )
        @return scores: torch.FloatTensor of shape (batch_size,)
        """
        P_u = self.user_embed(user)
        Q_i = self.item_embed(item)
        # 实现neighborhood aggregation
        # item -> user
        P_u = torch.matmul(attention_DotProduct(P_u, Q_i), Q_i)
        # user -> item
        Q_i = torch.matmul(attention_DotProduct(Q_i, P_u), P_u)
        return torch.sum(P_u * Q_i, dim=1) 

# MSE:
# AttMFii Max Recall@10: 0.0344 at epoch 1
# AttMFii Min RMSE: 1.0712 at epoch 93
# BPR:
# AttMFii Max Recall@10: 0.0459 at epoch 11    
# here we add the item info to the user embedding at first, and then add the updated user info to the item embedding
# this is okay