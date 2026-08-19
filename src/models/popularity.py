import numpy as np
from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    def fit(self, train_df, n_users, n_items):
        self.n_items = n_items
        counts = np.zeros(n_items)
        vc = train_df.i.value_counts()
        counts[vc.index.values] = vc.values
        self.pop_scores = counts / (counts.max() + 1e-9)
        return self

    def score(self, u, candidate_items):
        return self.pop_scores[candidate_items]

    def recommend_for_items(self, liked_items, k, exclude=None):
        exclude = exclude or set()
        scores = self.pop_scores.copy()
        if exclude:
            scores[list(exclude)] = -np.inf
        top_k = np.argpartition(-scores, k)[:k]
        return top_k[np.argsort(-scores[top_k])]