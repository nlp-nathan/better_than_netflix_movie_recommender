import numpy as np
import pandas as pd

def relevant_df(recommended, test, user_col, item_col, rank_col):
    # Movies that a user has not reviewed will not be included
    df_relevant = pd.merge(recommended, test, on=[user_col, item_col])[[user_col, item_col, rank_col]]

    # Out of the # of movies a user has reviewed in the test set, how many are actually recommended
    df_relevant_count = df_relevant.groupby(user_col, as_index=False)[user_col].agg({'relevant': 'count'})
    test_count = test.groupby(user_col, as_index=False)[user_col].agg({'actual': 'count'})
    relevant_ratio = pd.merge(df_relevant_count, test_count, on=user_col)
    return relevant_ratio

def precision_at_k(recommended, test, user_col, item_col, rank_col):
    relevant_ratio = relevant_df(recommended, test, user_col, item_col, rank_col)
    precision_at_k = ((relevant_ratio['relevant'] / 10) / len(test[user_col].unique())).sum()

    return precision_at_k

def recall_at_k(recommended, test, user_col, item_col, rank_col):
    relevant_ratio = relevant_df(recommended, test, user_col, item_col, rank_col)
    recall_at_k = ((relevant_ratio['relevant'] / relevant_ratio['actual']) / len(test[user_col].unique())).sum()

    return recall_at_k

def mean_average_precision(recommended, test, user_col, item_col, rank_col):
    # Movies that a user has not reviewed will not be included
    df_relevant = pd.merge(recommended, test, on=[user_col, item_col])[[user_col, item_col, rank_col]]

    relevant_ratio = relevant_df(recommended, test, user_col, item_col, rank_col)
    df_relevant['precision@k'] = (df_relevant.groupby(user_col).cumcount() + 1) / df_relevant[rank_col]

    # Calculate average precision for each user.
    relevant_ordered = df_relevant.groupby(user_col).agg({'precision@k': 'sum'}).reset_index()
    merged = pd.merge(relevant_ordered, relevant_ratio, on=user_col)
    merged['avg_precision'] = merged['precision@k'] / merged['actual']

    # Calculate mean average precision
    score = (merged['avg_precision'].sum() / len(test[user_col].unique()))

    return score

def ndcg(recommended, test, user_col, item_col, rank_col):
    # Movies that a user has not reviewed will not be included
    df_relevant = pd.merge(recommended, test, on=[user_col, item_col])[[user_col, item_col, rank_col]]

    relevant_ratio = relevant_df(recommended, test, user_col, item_col, rank_col)

    df_relevant['dcg'] = 1 / np.log1p(df_relevant[rank_col])
    dcg = df_relevant.groupby(user_col, as_index=False, sort=False).agg({'dcg': 'sum'})
    ndcg = pd.merge(dcg, relevant_ratio, on=user_col)
    ndcg['idcg'] = ndcg['actual'].apply(lambda x: sum(1 / np.log1p(range(1, min(x, 10) + 1))))
    score = (ndcg['dcg'] / ndcg['idcg']).sum() / len(test[user_col].unique())

    return score