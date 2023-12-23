import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_ratings(model, user_id, item_ids):
    """
    获取用户的评分 0~1
    @param model: torch.nn.Module
    @param user: torch.LongTensor of shape (1, )
    @param item: torch.LongTensor of shape (m, ) m <= m_items
    @return rating: torch.FloatTensor of shape (1, m)
    """
    user_emb = model.user_embed(user_id)
    item_emb = model.item_embed(item_ids)
    rating = F.sigmoid(torch.matmul(user_emb, item_emb.t()))
    return rating

class PureMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, n_factors)
        self.item_embed = nn.Embedding(n_items, n_factors)
        self._init_weight()

    def _init_weight(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.02)

    def forward(self, user, item):
        """
        @param user: torch.LongTensor of shape (batch_size, )
        @param item: torch.LongTensor of shape (batch_size, )
        @return scores: torch.FloatTensor of shape (batch_size,)
        """
        P_u = self.user_embed(user)
        Q_i = self.item_embed(item)
        scores = (P_u * Q_i).sum(dim=1)
        return scores
    
class SimplifiedGCN(nn.Module):
    def __init__(self, input_dim, output_dim, self_loop=True, norm=True):
        """
        graph convolution

        Parameters
        ----------
        @param input_dim: input dimension
        @param output_dim: output dimension
        @param self_loop: whether to add self-loop
        @param norm: whether to normalize
        """
        super(SimplifiedGCN, self).__init__()
        self.weights = nn.Parameter(torch.rand(input_dim, output_dim).float())
        self.self_loop = self_loop
        self.norm = norm
        
    def forward(self, A, X):
        """
        @param A: adjacency matrix, (num_nodes, num_nodes)
        @param X: feature inputs, (num_nodes, input_dim)
        @return: output (num_nodes, output_dim)
        """
        if self.self_loop:   # add self-loop 
            A = A + torch.eye(A.size(0))    # \tilde{A} = A + I (num_nodes, num_nodes)
        if self.norm:        # normalize
            D = torch.sum(A, axis=0)        # D                 (num_nodes,)
            D = torch.diag(D)               # \tilde{D}         (num_nodes, num_nodes)
            D = torch.sqrt(torch.inverse(D))# \tilde{D}^{-1/2}  (num_nodes, num_nodes)
            A = D @ A @ D                   # \hat{A}           (num_nodes, num_nodes)
        return A @ X @ self.weights         # \hat{A}XW         (num_nodes, output_dim)
    
class LightGCN(nn.Module):
    def __init__(self, 
                 n_users:int,
                 m_items:int,
                 latent_dim:int,
                 sparse_rating_matrix: torch.sparse.FloatTensor):
        super(LightGCN, self).__init__()
        self.num_users  = n_users
        self.num_items  = m_items
        self.latent_dim = latent_dim
        self.n_layers = 3
        self.Graph = sparse_rating_matrix 
        # Graph = D^{-1/2} @ A @D^{-1/2}
        """
        A = 
            |I,   R|
            |R^T, I|
        """ 
        
        self.user_embed = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.item_embed = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )
        self.Sigmoid = nn.Sigmoid()

        self._init_weight()

    def _init_weight(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.01)
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.user_embed.weight
        items_emb = self.item_embed.weight
        all_emb = torch.cat([users_emb, items_emb])  # (num_users + num_items, latent_dim)
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        #  recover 
        users, items = torch.split(
            light_out, [self.num_users, self.num_items]
        )
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.Sigmoid(torch.matmul(users_emb, items_emb.t()))
        return rating
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

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

class AttentionGCN(nn.Module):
    def __init__(self, n_users:int, m_items:int, embed_dim:int):
        super(AttentionGCN, self).__init__()
        self.fc = nn.Linear(embed_dim, 1)

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
        A_ui = attention_DotProduct(P_u, Q_i)
        return self.fc(A_ui @ Q_i)
    
class DemoGCN(nn.Module):
    def __init__(self, n_users:int, m_items:int, embed_dim:int, n_layers:int, A_norm:torch.FloatTensor):
        """
        graph convolution

        Parameters
        ----------
        @param input_dim: input dimension
        @param output_dim: output dimension
        @param self_loop: whether to add self-loop
        @param norm: whether to normalize
        """
        super(DemoGCN, self).__init__()
        self.weights = nn.Parameter(torch.rand(embed_dim, 1).float())
        self.A_norm = A_norm # (n_users, m_items)
        self.n_layers = n_layers

        self.user_embed = torch.nn.Embedding(
            num_embeddings = n_users, embedding_dim = embed_dim
        )
        self.item_embed = torch.nn.Embedding(
            num_embeddings = m_items, embedding_dim = embed_dim
        )
        self._init_weight()

    def _init_weight(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.1)

    def compute_embedding(self):
        """
        what LightGCN do
        """
        users_embs = []
        items_embs = []
        users_emb = self.user_embed.weight
        items_emb = self.item_embed.weight
        users_embs.append(users_emb)
        items_embs.append(items_emb)
        for layer in range(self.n_layers - 1):
            users_emb = users_emb + self.A_norm @ items_emb
            items_emb = self.A_norm.T @ users_emb + items_emb
            users_embs.append(users_emb)
            items_embs.append(items_emb)

        users_emb = torch.stack(users_embs, dim=1)
        users_emb = torch.mean(users_emb, dim=1)
        items_emb = torch.stack(items_embs, dim=1)
        items_emb = torch.mean(items_emb, dim=1)
        return users_emb, items_emb

        
    def forward(self, user, item):
        """
        @param X: feature inputs, (m_items, embed_dim)
        @return: output (m_items, 1)
        """
        users_emb, items_emb = self.compute_embedding()
        users_emb = users_emb[user]
        items_emb = items_emb[item]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
        
class AttentionMF(nn.Module):
    def __init__(self, n_users:int, m_items:int, embed_dim:int):
        super(AttentionMF, self).__init__()
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
        Q_i = torch.matmul(attention_DotProduct(P_u, Q_i), Q_i)
        return torch.sum(P_u * Q_i, dim=1) 