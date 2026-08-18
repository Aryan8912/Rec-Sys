import numpy as np


class BaseRecommender:
    def fit(self, train_df, n_users, n_items):
        """Train on train_df (columns: u, i, label, timestamp). Sets self.n_items."""
        raise NotImplementedError

    def score(self, u, candidate_items):
        """Return np.array of scores for candidate_items, same order, for user u."""
        raise NotImplementedError

    def recommend(self, u, k, exclude=None):
        """Top-k item indices for user u, excluding a set of already-seen items."""
        exclude = exclude or set()
        all_items = np.arange(self.n_items)
        scores = self.score(u, all_items)
        if exclude:
            scores = scores.copy()
            scores[list(exclude)] = -np.inf
        top_k = np.argpartition(-scores, k)[:k]
        return top_k[np.argsort(-scores[top_k])]