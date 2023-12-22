import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class PureMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, n_factors)
        self.item_embed = nn.Embedding(n_items, n_factors)
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
        scores = (P_u * Q_i).sum(dim=1)
        return scores
    
class SigmoidMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, n_factors)
        self.item_embed = nn.Embedding(n_items, n_factors)
        self._init_weight()
        self.sigmoid = nn.Sigmoid()

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
        scores = (P_u * Q_i).sum(dim=1)
        return self.sigmoid(scores)
    
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
        
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )
        self.Sigmoid = nn.Sigmoid()

        self._init_weight()

    def _init_weight(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.1)
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
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
    def __init__(self, n_users:int, m_items:int, embed_dim:int, self_loop=False, norm=False):
        super(AttentionGCN, self).__init__()
        self.self_loop = self_loop
        self.norm = norm
        self.weights = nn.Parameter(torch.rand(embed_dim, 1).float())
        self.user_embed = torch.nn.Embedding(
            num_embeddings = n_users, embedding_dim = embed_dim
        )
        self.item_embed = torch.nn.Embedding(
            num_embeddings = m_items, embedding_dim = embed_dim
        )
        
    def forward(self, user, item):
        """
        @param user: torch.LongTensor of shape (batch_size, )
        @param item: torch.LongTensor of shape (batch_size, )
        @return scores: torch.FloatTensor of shape (batch_size,)
        """
        P_u = self.user_embed(user)
        Q_i = self.item_embed(item)
        A = attention_DotProduct(P_u, Q_i)

        if self.self_loop:   # add self-loop 
            A = A + torch.eye(A.size(0))    # \tilde{A} = A + I (num_nodes, num_nodes)
        if self.norm:        # normalize
            D = torch.sum(A, axis=0)        # D                 (num_nodes,)
            D = torch.diag(D)               # \tilde{D}         (num_nodes, num_nodes)
            D = torch.sqrt(torch.inverse(D))# \tilde{D}^{-1/2}  (num_nodes, num_nodes)
            A = D @ A @ D                   # \hat{A}           (num_nodes, num_nodes)

        return A @ Q_i @ self.weights       # \hat{A}XW         (num_nodes, output_dim)