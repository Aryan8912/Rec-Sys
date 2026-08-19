import numpy as np
from scipy.sparse import csr_matrix
from .base import BaseRecommender


class ItemItemCF(BaseRecommender):
    def __init__(self, top_k=50):
        self.top_k = top_k

    def fit(self, train_df, n_users, n_items):
        self.n_items = n_items
        R = csr_matrix((np.ones(len(train_df)), (train_df.u, train_df.i)),
                        shape=(n_users, n_items))
        self.user_history = R

        norms = np.sqrt(R.power(2).sum(axis=0)).A1
        norms[norms == 0] = 1e-9
        R_norm = R.multiply(1.0 / norms)
        dense_sim = (R_norm.T @ R_norm).toarray() 
        np.fill_diagonal(dense_sim, 0)

        k = min(self.top_k, n_items - 1)
        rows, cols, vals = [], [], []
        for i in range(n_items):
            row = dense_sim[i]
            top_idx = np.argpartition(-row, k)[:k]
            top_idx = top_idx[row[top_idx] > 0]  
            rows.extend([i] * len(top_idx))
            cols.extend(top_idx.tolist())
            vals.extend(row[top_idx].tolist())

        self.item_sim = csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))
        del dense_sim  
        return self

    def score(self, u, candidate_items):
        user_items = self.user_history[u].indices
        if len(user_items) == 0:
            return np.zeros(len(candidate_items))
        sub = self.item_sim[candidate_items][:, user_items]
        return np.asarray(sub.sum(axis=1)).ravel()

    def recommend_for_items(self, liked_items, k, exclude=None):
        exclude = exclude or set()
        all_items = np.arange(self.n_items)
        if not liked_items:
            scores = np.zeros(self.n_items)
        else:
            sub = self.item_sim[all_items][:, list(liked_items)]
            scores = np.asarray(sub.sum(axis=1)).ravel()
        exclude_all = set(exclude) | set(liked_items)
        if exclude_all:
            scores = scores.copy()
            scores[list(exclude_all)] = -np.inf
        top_k = np.argpartition(-scores, k)[:k]
        return top_k[np.argsort(-scores[top_k])]