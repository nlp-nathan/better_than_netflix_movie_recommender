# -*- coding: utf-8 -*-
import math
import numpy as np
import os
import pandas as pd
import requests

from tqdm import tqdm


def download_movie(fp):
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.data'
    r = requests.get(url, stream=True)
    block_size = 1024
    total_size = int(r.headers.get('content-length', 0))
    num_iterables = math.ceil(total_size / block_size)

    # Download if not already downloaded.
    if not os.path.exists(fp):
        dir, name = os.path.split(fp)
        os.makedirs(dir)
        with open(fp, "wb") as file:
            for data in tqdm(
                r.iter_content(block_size), total=num_iterables, unit="KB", unit_scale=True
            ):
                file.write(data)

def stratified_split(df, by, train_size=0.70):
    # Split each user's reviews by % for training.
    splits = []
    for _, group in df.groupby(by):
        group = group.sample(frac=1, random_state=123)
        group_splits = np.split(group, [round(train_size * len(group))])

        # Label the train and test sets.
        for i in range(2):
            group_splits[i]["split_index"] = i
            splits.append(group_splits[i])

    # Concatenate splits for all the groups together.
    splits_all = pd.concat(splits)

    # Take train and test split using split_index.
    train = splits_all[splits_all["split_index"] == 0].drop("split_index", axis=1)
    test = splits_all[splits_all["split_index"] == 1].drop("split_index", axis=1)

    return train, test

def numpy_stratified_split(X, ratio=0.75, seed=42):
    np.random.seed(seed)  # set the random seed
    test_cut = int((1 - ratio) * 100)  # percentage of ratings to go in the test set

    # initialize train and test set matrices
    Xtr = X.copy()
    Xtst = X.copy()

    # find the number of rated movies per user
    rated = np.sum(Xtr != 0, axis=1)

    # for each user, cut down a test_size% for the test set
    tst = np.around((rated * test_cut) / 100).astype(int)

    for u in range(X.shape[0]):
        # For each user obtain the index of rated movies
        idx = np.asarray(np.where(Xtr[u] != 0))[0].tolist()

        # extract a random subset of size n from the set of rated movies without repetition
        idx_tst = np.random.choice(idx, tst[u], replace=False)
        idx_train = list(set(idx).difference(set(idx_tst)))

        # change the selected rated movies to unrated in the train set
        Xtr[u, idx_tst] = 0
        # set the movies that appear already in the train set as 0
        Xtst[u, idx_train] = 0

    del idx, idx_train, idx_tst

    return Xtr, Xtst
