"""
这里来专门研究一下attention机制的model能否对GCN的效果有所提升
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
dataset: ml-lastest-small

baseline:
- mse task 100 epoch
- bpr task 20 epoch

PureMF
- MSE:
    - PureMF Max Recall@10: 0.0590 at epoch 1
    - PureMF Min RMSE: 1.2579 at epoch 7
- BPR:
    - PureMF Max Recall@10: 0.0590 at epoch 15

LightGCN
- MSE:
    - LightGCN Max Recall@10: 0.0066 at epoch 54
    - LightGCN Min RMSE: 1.5265 at epoch 34
- BPR:
    - LightGCN Max Recall@10: 0.0492 at epoch 15
"""

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
    """
    AttMFui
    - MSE:    
        - AttMFui Max Recall@10: 0.0410 at epoch 2
        - AttMFui Min RMSE: 1.0771 at epoch 2

    note: we can't use the AttMFui model to bpr task, for the loss will be constant, because we just aggregate user info to item embedding, which make no sense in bpr task (we use user embedding to predict item embedding)
    """
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

    
class AttMFiu(nn.Module):
    """
    - MSE:
        - AttMFiu Max Recall@10: 0.0344 at epoch 1
        - AttMFiu Min RMSE: 1.2482 at epoch 11
    - BPR:
        - AttMFiu Max Recall@10: 0.0410 at epoch 9
        
    note: in our mse task, as we only add the item info to the user embedding, the recall@10 get worse and worse
    """
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

    
class AttMFuu(nn.Module):
    """
    - MSE:
        - AttMFuu Max Recall@10: 0.0410 at epoch 1
        - AttMFuu Min RMSE: 1.0265 at epoch 66  
        
    note: we can't also use the AttMFuu model to bpr task, for the loss will be constant. As long as we add the user info to the item embedding at first, the bpr loss will be constant.
    """
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

    
class AttMFii(nn.Module):
    """
    - MSE:
        - AttMFii Max Recall@10: 0.0344 at epoch 1
        - AttMFii Min RMSE: 1.0712 at epoch 93
    - BPR:
        - AttMFii Max Recall@10: 0.0459 at epoch 11

    Get the **best Recall@10** above all the attention models, but the RMSE is not the best.    

    note: here we add the item info to the user embedding at first, and then add the updated user info to the item embedding, this is okay.
    we can compare the result with AttMFiu, and find that AttMFii's result is better than AttMFiu's

            AttMFii   AttMFiu
    RMSE:      1.0712 < 1.2482
    Recall@10: 0.0459 > 0.0410

    so we can conclude that the order of the aggregation matters
    """
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

class SelfAttMFii(nn.Module):
    """
    - MSE:
        - SelfAttMFii Max Recall@10: 0.0344 at epoch 3
        - SelfAttMFii Min RMSE: 1.0251 at epoch 38
    - BPR:
        - SelfAttMFii Max Recall@10: 0.0426 at epoch 9

    Get the **best RMSE** above all the models, but the recall@10 is not the best.
    """
    def __init__(self, n_users:int, m_items:int, embed_dim:int):
        super(SelfAttMFii, self).__init__()
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
        P_u = torch.matmul(attention_DotProduct(P_u, P_u), P_u)
        Q_i = torch.matmul(attention_DotProduct(Q_i, Q_i), Q_i)
        return torch.sum(P_u * Q_i, dim=1)  

class Samfii(nn.Module):
    """
    mix SelfAttMFii and AttMFii

    - MSE:
        - Samfii Max Recall@10: 0.0279 at epoch 1
        - Samfii Min RMSE: 1.0860 at epoch 75
    - BPR:
        - Samfii Max Recall@10: 0.0410 at epoch 9

    not so good
    """
    def __init__(self, n_users:int, m_items:int, embed_dim:int):
        super(Samfii, self).__init__()
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
        P_u = torch.matmul(attention_DotProduct(P_u, P_u), P_u)
        Q_i = torch.matmul(attention_DotProduct(Q_i, Q_i), Q_i)
        # 实现neighborhood aggregation
        # item -> user
        P_u = torch.matmul(attention_DotProduct(P_u, Q_i), Q_i)
        # user -> item
        Q_i = torch.matmul(attention_DotProduct(Q_i, P_u), P_u)
        return torch.sum(P_u * Q_i, dim=1)     
    

from gcn.posenc import PositionalEncoding

class Pesamf(nn.Module):
    """
    mix PeMF and SelfAttMFii  

    - MSE:
        - Pesamf Max Recall@10: 0.0049 at epoch 1
        - Pesamf Min RMSE: 1.0985 at epoch 5
    - BPR:
        - Pesamf Max Recall@10: 0.0426 at epoch 7

    not so good
    """
    def __init__(self, n_users:int, m_items:int, embed_dim:int):
        super(Pesamf, self).__init__()
        self.embed_dim = embed_dim
        self.user_embed = torch.nn.Embedding(
            num_embeddings = n_users, embedding_dim = embed_dim
        )
        self.item_embed = torch.nn.Embedding(
            num_embeddings = m_items, embedding_dim = embed_dim
        )
        self.pos_encoding = PositionalEncoding(embed_dim)
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
        P_u = self.pos_encoding(self.user_embed(user) * math.sqrt(self.embed_dim))
        Q_i = self.pos_encoding(self.item_embed(item) * math.sqrt(self.embed_dim))
        P_u = torch.matmul(attention_DotProduct(P_u, P_u), P_u)
        Q_i = torch.matmul(attention_DotProduct(Q_i, Q_i), Q_i)
        # 实现neighborhood aggregation
        # item -> user
        # P_u = torch.matmul(attention_DotProduct(P_u, Q_i), Q_i)
        # # user -> item
        # Q_i = torch.matmul(attention_DotProduct(Q_i, P_u), P_u)
        return torch.sum(P_u * Q_i, dim=1) 


class Peamfii(nn.Module):
    """
    mix PeMF and AttMFii  

    - MSE:
        - Peamfii Max Recall@10: 0.0082 at epoch 54
        - Peamfii Min RMSE: 1.1313 at epoch 3
    - BPR:
        - Peamfii Max Recall@10: 0.0426 at epoch 3
    """
    def __init__(self, n_users:int, m_items:int, embed_dim:int):
        super(Peamfii, self).__init__()
        self.embed_dim = embed_dim
        self.user_embed = torch.nn.Embedding(
            num_embeddings = n_users, embedding_dim = embed_dim
        )
        self.item_embed = torch.nn.Embedding(
            num_embeddings = m_items, embedding_dim = embed_dim
        )
        self.pos_encoding = PositionalEncoding(embed_dim)
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
        P_u = self.pos_encoding(self.user_embed(user) * math.sqrt(self.embed_dim))
        Q_i = self.pos_encoding(self.item_embed(item) * math.sqrt(self.embed_dim))
        # 实现neighborhood aggregation
        # item -> user
        P_u = torch.matmul(attention_DotProduct(P_u, Q_i), Q_i)
        # user -> item
        Q_i = torch.matmul(attention_DotProduct(Q_i, P_u), P_u)
        return torch.sum(P_u * Q_i, dim=1) 

class AttGCN(nn.Module):
    """
    deep AttGCN

    - MSE:
        - AttGCN Max Recall@10: 0.0115 at epoch 1
        - AttGCN Min RMSE: 1.1280 at epoch 5
    - BPR:
        - AttGCN Max Recall@10: 0.0311 at epoch 1
    """
    def __init__(self, n_users:int, m_items:int, embed_dim:int):
        super(AttGCN, self).__init__()
        self.n_layers = 2
        self.embed_dim = embed_dim
        self.user_embed = torch.nn.Embedding(
            num_embeddings = n_users, embedding_dim = embed_dim
        )
        self.item_embed = torch.nn.Embedding(
            num_embeddings = m_items, embedding_dim = embed_dim
        )
        self.pos_encoding = PositionalEncoding(embed_dim)
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
        P_u = self.pos_encoding(self.user_embed(user) * math.sqrt(self.embed_dim))
        Q_i = self.pos_encoding(self.item_embed(item) * math.sqrt(self.embed_dim))

        # like LightGCN but attention here
        user_embeds = [P_u]
        item_embeds = [Q_i]
        for _ in range(self.n_layers):
            # 实现neighborhood aggregation
            # item -> user
            P_u = torch.matmul(attention_DotProduct(P_u, Q_i), Q_i)
            # user -> item
            Q_i = torch.matmul(attention_DotProduct(Q_i, P_u), P_u)
            user_embeds.append(P_u)
            item_embeds.append(Q_i)

        P_u = torch.stack(user_embeds, dim=1)
        Q_i = torch.stack(item_embeds, dim=1)
        P_u = torch.mean(P_u, dim=1)
        Q_i = torch.mean(Q_i, dim=1)
        return torch.sum(P_u * Q_i, dim=1) 
    
class BiasAtMF(nn.Module):
    """
    with embed_dim = 64
    BPR:
    BiasMF Max Recall@10: 0.0869 at epoch 14
    MSE:
    BiasMF Max Recall@10: 0.0344 at epoch 69
    BiasMF Min RMSE: 1.1977 at epoch 2
    best model now
    """
    def __init__(self, n_users, n_items, n_factors):
        super(BiasAtMF, self).__init__()
        self.user_embed = nn.Embedding(num_embeddings=n_users, embedding_dim=n_factors)
        self.item_embed = nn.Embedding(num_embeddings=n_items, embedding_dim=n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        for param in self.parameters():
            nn.init.normal_(param, std=0.02)

    def forward(self, user_id, item_id):
        """
        @param user: torch.LongTensor of shape (batch_size, )
        @param item: torch.LongTensor of shape (batch_size, )
        @return scores: torch.FloatTensor of shape (batch_size,)
        """
        P_u = self.user_embed(user_id)
        Q_i = self.item_embed(item_id)
        b_u = self.user_bias(user_id).flatten()
        b_i = self.item_bias(item_id).flatten()
        # 实现neighborhood aggregation
        # item -> user
        P_u = torch.matmul(attention_DotProduct(P_u, Q_i), Q_i)
        # user -> item
        Q_i = torch.matmul(attention_DotProduct(Q_i, P_u), P_u)
        scores = (P_u * Q_i).sum(axis=1) + b_u + b_i
        return scores