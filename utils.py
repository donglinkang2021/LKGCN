"""
将常用的获取训练数据的函数放在这里

我们主要有两种任务
1. BPR
    - 用来训练ranking的任务 *排名任务*
    - 用来训练implicit feedback的任务 *隐式反馈任务*
    - edge existence
2. MSE
    - 用来训练rating的任务 *评分任务*
    - 用来训练explicit feedback的任务 *显式反馈任务*
    - edge rating
"""
import torch
import random
from gcn.data import read_data_ml100k, split_data_ml100k, load_data_ml100k


def get_bpr_data():    
    ratings_df, num_users, num_items = read_data_ml100k("./data/ml-latest-small")
    train_data, test_data = split_data_ml100k(ratings_df, num_users, 'seq-aware', test_ratio=0.1)
    users_train, items_train, ratings_train, train_candidates = load_data_ml100k(
        train_data, num_users, num_items, feedback="implicit"
    ) # train_candidates is a dict of {user: [items]}
    users_test, items_test, ratings_test, test_candidates = load_data_ml100k(
        test_data, num_users, num_items, feedback="implicit"
    ) # test_candidates is a dict of {user: [items]}
    return num_users, num_items, users_train, items_train, train_candidates, test_candidates

class PRDataset(torch.utils.data.Dataset):
    def __init__(self, users, items, candidates, num_items):
        self.users = users
        self.items = items
        self.cand = candidates
        self.all = set([i for i in range(num_items)])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        neg_items = list(self.all - set(self.cand[int(self.users[idx])]))
        indices = random.randint(0, len(neg_items) - 1)
        return self.users[idx], self.items[idx], neg_items[indices]

def get_bpr_train_loader(users_train, items_train, train_candidates, num_items, shuffle=True):
    batch_size = 1024
    trainset = PRDataset(users_train, items_train, train_candidates, num_items)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    return train_loader


def get_mse_data():
    ratings_df, num_users, num_items = read_data_ml100k("./data/ml-latest-small")
    train_data, test_data = split_data_ml100k(ratings_df, num_users, 'seq-aware', test_ratio=0.1)
    _, _, _, train_candidates = load_data_ml100k(
        train_data, num_users, num_items, feedback="implicit"
    ) # train_candidates is a dict of {user: [items]}
    users_train, items_train, ratings_train, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback="explicit"
    ) # train_candidates is a dict of {user: [items]}
    _, _, _, test_candidates = load_data_ml100k(
        test_data, num_users, num_items, feedback="implicit"
    ) # train_candidates is a dict of {user: [items]}
    users_test, items_test, ratings_test, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback="explicit"
    ) # test_candidates is a dict of {user: [items]}
    return num_users, num_items, users_train, items_train, ratings_train, users_test, items_test, ratings_test, train_candidates, test_candidates

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
    
def get_mse_train_loader(users_train, items_train, ratings_train, shuffle=True):
    batch_size = 1024
    trainset = ML100KPoint(users_train, items_train, ratings_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    return train_loader