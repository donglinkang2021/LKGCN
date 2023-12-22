import torch
import math
from gcn.metrics import recall_precision_at_k

class ML100KPair(torch.utils.data.Dataset):
    def __init__(self, users, items):
        assert len(users) == len(items)
        self.users = users
        self.items = items

    def __getitem__(self, index):
        return (self.users[index], self.items[index])

    def __len__(self):
        return len(self.users)


def evaluate_ranking(model, test_candidates: dict, train_candidates: dict, num_users: int, num_items: int, batch_size: int = 256, top_k: int = 4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ranked_list = {}
    recall_list = []
    all_items = set([i for i in range(num_items)])
    for u in range(num_users):
        neg_items = list(all_items - set(train_candidates[int(u)]))
        user_ids, item_ids, scores = [], [], []
        [item_ids.append(i) for i in neg_items]
        [user_ids.append(u) for _ in neg_items]
        test_loader = torch.utils.data.DataLoader(
            ML100KPair(user_ids, item_ids), 
            batch_size = batch_size, 
            shuffle=False
        )
        for user, item in test_loader:
            user = user.to(device)
            item = item.to(device)
            score = model(user, item)
            scores.extend(score.cpu().detach().numpy().tolist())

        item_scores = list(zip(item_ids, scores))
        ranked_list[u] = sorted(item_scores, key=lambda t: t[1], reverse=True)
        recommends = [r[0] for r in ranked_list[u]]
        candidates = test_candidates[u]
        recall, _ = recall_precision_at_k(candidates, recommends, top_k)
        recall_list.append(recall)
    return sum(recall_list) / len(recall_list) # mean


class ML100KPoint(torch.utils.data.Dataset):
    def __init__(self, users, items, ratings):
        assert len(users) == len(items)
        self.users = users
        self.items = items
        self.ratings = ratings

    def __getitem__(self, index):
        return (self.users[index], self.items[index], self.ratings[index])

    def __len__(self):
        return len(self.users)


def evaluate_rating(model, test_users: list, test_items: list, test_ratings: list, batch_size: int = 256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = torch.utils.data.DataLoader(
        ML100KPoint(test_users, test_items, test_ratings), 
        batch_size = batch_size, 
        shuffle=False
    )
    mse = torch.nn.MSELoss()
    loss = 0
    for user, item, rating in test_loader:
        user = user.to(device)
        item = item.to(device)
        rating = rating.to(device)
        pred = model(user, item)
        loss += mse(pred, rating).item()
    return math.sqrt(loss / len(test_loader)) # rmse