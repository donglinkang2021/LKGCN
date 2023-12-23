"""
这里主要研究一下添加位置编码的效果

因为我们的数据本来就处理成了seq-aware的形式，所以我们如果希望模型学习到序列信息有两种办法
- 添加time维度的信息 这是全局的positional encoding
- 添加一个bacth内item维度的信息 这是局部的positional encoding 
我们在这里专注于第二点
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):  #@save
    def __init__(self, embed_dim, max_len=2000):
        """
        Positional encoding.
        
        Parameters
        ----------
        @param embed_dim: int
            hidden size
        @param max_len: int
            maximum length *max batch_size here*
        """
        super().__init__()
        # Create a long enough P
        self.P = torch.zeros((max_len, embed_dim))
        X = torch.arange(
            max_len, 
            dtype=torch.float32
        ).reshape(-1, 1) / torch.pow(
            10000, torch.arange(
                0, embed_dim, 2, dtype=torch.float32
            ) / embed_dim
        )
        self.P[:, 0::2] = torch.sin(X)
        self.P[:, 1::2] = torch.cos(X)

    def forward(self, X):
        """
        Parameters
        ----------
        @param X : torch.Tensor
            Shape (batch_size, embed_dim)
        @return torch.Tensor
            Shape (batch_size, embed_dim)
        """
        X = X + self.P[:X.shape[0], :].to(X.device)
        return X
    
class PeMF(nn.Module):
    """
    with positional encoding

    with train data shuffle
    - MSE:
        - PeMF Max Recall@10: 0.0033 at epoch 1
        - PeMF Min RMSE: 1.1582 at epoch 2
    - BPR:
        - PeMF Max Recall@10: 0.0492 at epoch 18
    already better than LightGCN

    without train data shuffle *not good*
    - MSE:
        - PeMF Max Recall@10: 0.0049 at epoch 2
        - PeMF Min RMSE: 1.3784 at epoch 1
    - BPR:
        - PeMF Max Recall@10: 0.0377 at epoch 12
    """
    def __init__(self, n_users:int, m_items:int, embed_dim:int):
        super(PeMF, self).__init__()
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
        return torch.sum(P_u * Q_i, dim=1)     