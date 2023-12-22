import os
import numpy as np
import pandas as pd

def read_data_ml100k(data_dir = "data\ml-latest-small"):
    ratings_df = pd.read_csv(os.path.join(data_dir, "ratings_processed.csv"))
    num_users = ratings_df.userId.unique().shape[0]
    num_items = ratings_df.movieId.unique().shape[0]
    return ratings_df, num_users, num_items

def split_data_ml100k(ratings_df, num_users,split_mode='random', test_ratio=0.1):
    """
    Split the dataset in random mode or seq-aware mode.
    
    Parameters
    ----------
    @param ratings_df: pd.DataFrame
        data frame of the ratings
    @param num_users: int
        number of users
    @param split_mode: str ('random' or 'seq-aware')
        split mode of the dataset
    @param test_ratio: float
        ratio of the test set

    Returns
    -------
    @return train_df: pd.DataFrame
        data frame of the training set
    @return test_df: pd.DataFrame
        data frame of the test set
    """
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in ratings_df.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
    else:
        mask = np.random.uniform(0, 1, (len(ratings_df))) < 1 - test_ratio
        train_df = ratings_df[mask]
        test_df = ratings_df[~mask]
    return train_df, test_df

def load_data_ml100k(ratings_df, num_users, num_items, feedback='explicit'):
    """Load the ml100k dataset.
    
    Parameters
    ----------
    @param ratings_df: pd.DataFrame
        data frame of the dataset
    @param num_users: int
        number of users
    @param num_items: int
        number of items
    @param feedback: str ('explicit' or 'implicit')
        type of the feedback data

    Returns
    -------
    @return users: list
        list of user indices
    @return items: list
        list of item indices
    @return scores: list
        list of scores/ratings
    @return inter: dict or np.ndarray
        interaction information of the dataset
        - dict: {user_index: [item_index_1, item_index_2, ...]} if feedback == 'implicit'
        - np.ndarray: shape (num_items, num_users) if feedback == 'explicit'
    """
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in ratings_df.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter