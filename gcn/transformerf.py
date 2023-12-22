import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DotProductAttention(nn.Module):  
    def __init__(self, dropout):
        """Scaled dot product attention."""
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        self.attention_weights = torch.matmul(
            queries, 
            keys.T
        ) / math.sqrt(d)
        return torch.matmul(
            self.dropout(self.attention_weights), 
            values
        )

class AddNorm(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(embed_dim, bias=bias)
        self.W_k = nn.LazyLinear(embed_dim, bias=bias)
        self.W_v = nn.LazyLinear(embed_dim, bias=bias)
        self.W_o = nn.LazyLinear(embed_dim, bias=bias)

    def forward(self, queries, keys, values):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        output = self.attention(
            queries, 
            keys, 
            values
        )
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
    
    def transpose_qkv(self, X):
        """
        Transposition for parallel computation of multiple attention heads.
        
        Parameters
        ----------
        @param X: torch.Tensor
            Shape 
            (
                batch_size, 
                num_hiddens
            ).
        @return X: torch.Tensor
            Shape 
            (
                batch_size * num_heads, 
                num_hiddens / num_heads
            )
        """
        X = X.reshape(X.shape[0], self.num_heads, -1)
        return X.reshape(-1, X.shape[2])

    def transpose_output(self, X):
        """
        Reverse the operation of transpose_qkv.
        
        Parameters
        ----------
        @param X: torch.Tensor
            Shape 
            (
                batch_size * num_heads, 
                num_hiddens / num_heads
            ).
        @return X: torch.Tensor
            Shape 
            (
                batch_size, 
                num_hiddens
            )
        """
        X = X.reshape(-1, self.num_heads, X.shape[1])
        return X.reshape(X.shape[0], -1)

class SelfBlock(nn.Module):  
    def __init__(self, embed_dim, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.addnorm1 = AddNorm(embed_dim, dropout)
        self.attention = MultiHeadAttention(
            embed_dim, num_heads,
            dropout, use_bias
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim, bias=use_bias),
        )
        self.addnorm2 = AddNorm(embed_dim, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.attention(X, X, X))
        return self.addnorm2(Y, self.ffn(Y))
    
class CrossBlock(nn.Module):  
    def __init__(self, embed_dim, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.addnorm1 = AddNorm(embed_dim, dropout)
        self.attention = MultiHeadAttention(
            embed_dim, num_heads,
            dropout, use_bias
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim, bias=use_bias),
        )
        self.addnorm2 = AddNorm(embed_dim, dropout)

    def forward(self, user, item):
        Y = self.addnorm1(item, self.attention(item, user, user))
        return self.addnorm2(Y, self.ffn(Y))

class TransforMerF(nn.Module):
    def __init__(self, n_users:int, m_items:int, embed_dim:int, num_heads:int=8, dropout:float=0.1):
        super(TransforMerF, self).__init__()
        self.user_embed = torch.nn.Embedding(
            num_embeddings = n_users, embedding_dim = embed_dim
        )
        self.item_embed = torch.nn.Embedding(
            num_embeddings = m_items, embedding_dim = embed_dim
        )
        self.uu_block = SelfBlock(embed_dim, num_heads, dropout)
        self.ii_block = SelfBlock(embed_dim, num_heads, dropout)
        self.ui_block = CrossBlock(embed_dim, num_heads, dropout)
        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_embed.weight, std=0.1)
        nn.init.normal_(self.item_embed.weight, std=0.1)
        
    def forward(self, user, item):
        """
        @param user: torch.LongTensor of shape (batch_size, )
        @param item: torch.LongTensor of shape (batch_size, )
        @return scores: torch.FloatTensor of shape (batch_size,)
        """
        P_u = self.user_embed(user)
        Q_i = self.item_embed(item)
        P_u = self.uu_block(P_u)
        Q_i = self.ii_block(Q_i)
        Q_i = self.ui_block(P_u, Q_i)
        return torch.sum(P_u * Q_i, dim=1)