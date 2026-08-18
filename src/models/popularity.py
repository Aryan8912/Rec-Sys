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