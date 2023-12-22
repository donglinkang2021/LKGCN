import numpy as np

def recall_precision_at_k(candidates:list, recommends:list, topk: int):
    """
    Recall@K and Precision@K metrics for a single user.

    Parameters
    ----------
    @param candidates: list
        List of items index that user has interacted with. (ground truth)
    @param recommends: list
        List of recommended items index for a user. (predicted items)
        *need to be sorted by relevance in descending order*
    @param topk: int
        The number of top items to be recommended.

    Return
    ------
    @return recall_at_k: float
        Recall@K
    @return precision_at_k: float
        Precision@K

    Example
    -------
    >>> candidates = [2, 3, 4, 1]
    >>> recommends = [6, 2, 3, 4, 5]
    >>> topk = 3
    >>> recall_precision_at_k(candidates, recommends, topk)
    (0.5, 0.6666666666666666)
    """
    relevant_items = set(candidates)
    top_k_items = set(recommends[:topk])
    inter_items = relevant_items.intersection(top_k_items)
    recall_at_k = len(inter_items) / len(relevant_items)
    precision_at_k = len(inter_items) / topk
    return recall_at_k, precision_at_k

def hit_and_auc(candidates:list, recommends:list, topk: int):
    """
    Hit@K and AUC metrics for a single user.

    Parameters
    ----------
    @param candidates: list
        List of items index that user has interacted with. (ground truth)
    @param recommends: list
        List of recommended items index for a user. (predicted items)
        *need to be sorted by relevance in descending order*
    @param topk: int
        The number of top items to be recommended.

    Return
    ------
    @return hit_at_k: int
        Hit@K
    @return auc: float
        AUC

    Example
    -------
    >>> candidates = [2, 3, 4, 1]
    >>> recommends = [6, 2, 3, 4, 5]
    >>> topk = 3
    >>> hit_and_auc(candidates, recommends, topk)
    (2, 0.75)
    """
    relevant_items = set(candidates)
    hit_at_k = [(idx, val) for idx, val in enumerate(recommends[:topk])
              if val in relevant_items]
    hits_all = [(idx, val) for idx, val in enumerate(recommends)
                if val in relevant_items]
    max_idx = len(recommends) - 1
    auc = 1.0 * (max_idx - hits_all[0][0]) / max_idx if len(hits_all) > 0 else 0
    return len(hit_at_k), auc