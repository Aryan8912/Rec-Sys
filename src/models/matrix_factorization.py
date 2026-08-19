import numpy as np
from .base import BaseRecommender


class MatrixFactorization(BaseRecommender):
    def __init__(self, n_factors=32, lr=0.05, reg=0.01, epochs=15, seed=42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)

    def fit(self, train_df, n_users, n_items):
        self.n_items = n_items
        self.P = self.rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = self.rng.normal(0, 0.1, (n_items, self.n_factors))

        users = train_df.u.values
        items = train_df.i.values
        user_pos = train_df.groupby('u')['i'].apply(set).to_dict()

        n = len(train_df)
        for epoch in range(self.epochs):
            order = self.rng.permutation(n)
            total_loss = 0.0
            for idx in order:
                u, i = users[idx], items[idx]
                j = self.rng.integers(0, n_items)
                while j in user_pos.get(u, ()):
                    j = self.rng.integers(0, n_items)

                pu, qi, qj = self.P[u], self.Q[i], self.Q[j]
                x_uij = pu @ qi - pu @ qj
                sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x_uij, -30, 30)))
                grad = 1.0 - sigmoid

                self.P[u] += self.lr * (grad * (qi - qj) - self.reg * pu)
                self.Q[i] += self.lr * (grad * pu - self.reg * qi)
                self.Q[j] += self.lr * (-grad * pu - self.reg * qj)
                total_loss += -np.log(sigmoid + 1e-9)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'  MF epoch {epoch+1}/{self.epochs}, avg BPR loss: {total_loss/n:.4f}')
        return self

    def score(self, u, candidate_items):
        return self.Q[candidate_items] @ self.P[u]

    def recommend_for_items(self, liked_items, k, exclude=None):
        exclude = exclude or set()
        if not liked_items:
            pseudo_user = np.zeros(self.n_factors)
        else:
            pseudo_user = self.Q[list(liked_items)].mean(axis=0)
        scores = self.Q @ pseudo_user
        exclude_all = set(exclude) | set(liked_items)
        if exclude_all:
            scores = scores.copy()
            scores[list(exclude_all)] = -np.inf
        top_k = np.argpartition(-scores, k)[:k]
        return top_k[np.argsort(-scores[top_k])]