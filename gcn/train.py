import torch
import math
from gcn.metrics import recall_precision_at_k
from gcn.models import get_ratings

def evaluate_ranking(model, test_candidates: dict, train_candidates: dict, num_users: int, num_items: int, top_k: int = 4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recall_list = []
    all_items = set([i for i in range(num_items)])
    for u in range(num_users):
        neg_items = list(all_items - set(train_candidates[u]))
        item_ids = [i for i in neg_items]
        item_ids = torch.tensor(item_ids, dtype=torch.long).to(device)
        user_id = torch.tensor(u, dtype=torch.long).to(device)
        scores = get_ratings(model, user_id, item_ids)
        _, indices = torch.topk(scores, top_k)
        recommends = torch.take(item_ids, indices).cpu().numpy().tolist()
        candidates = test_candidates[u]
        recall, _ = recall_precision_at_k(candidates, recommends, top_k)
        recall_list.append(recall)
    return sum(recall_list) / len(recall_list) # mean


def evaluate_rating(model, test_users: list, test_items: list, test_ratings: list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    user = torch.tensor(test_users, dtype=torch.long).to(device)
    item = torch.tensor(test_items, dtype=torch.long).to(device)
    rating = torch.tensor(test_ratings, dtype=torch.float).to(device)
    mse = torch.nn.MSELoss()
    user = user.to(device)
    item = item.to(device)
    rating = rating.to(device)
    pred = model(user, item)
    loss = mse(pred, rating).item()
    return math.sqrt(loss) # rmse