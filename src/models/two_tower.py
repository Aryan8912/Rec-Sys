import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseRecommender


class TwoTowerNet(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.user_tower = nn.Linear(embedding_dim, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.item_tower = nn.Linear(embedding_dim, embedding_dim)

    def get_user_vec(self, user_ids):
        return self.user_tower(self.user_embedding(user_ids))

    def get_item_vec(self, item_ids):
        return self.item_tower(self.item_embedding(item_ids))

    def forward(self, user_ids, item_ids):
        u = self.get_user_vec(user_ids)
        i = self.get_item_vec(item_ids)
        return (u * i).sum(dim=-1)


class TwoTowerRecommender(BaseRecommender):
    def __init__(self, embedding_dim=32, lr=0.005, epochs=15, batch_size=1024, seed=42):
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, train_df, n_users, n_items):
        torch.manual_seed(self.seed)
        self.n_items = n_items
        self.net = TwoTowerNet(n_users, n_items, self.embedding_dim)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        users = torch.tensor(train_df.u.values, dtype=torch.long)
        items = torch.tensor(train_df.i.values, dtype=torch.long)
        user_pos = train_df.groupby('u')['i'].apply(set).to_dict()
        n = len(train_df)
        rng = np.random.default_rng(self.seed)

        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                u_batch = users[idx]
                i_batch = items[idx]
                # sample one negative item per positive
                j_batch = torch.tensor(
                    [self._sample_neg(rng, u.item(), n_items, user_pos) for u in u_batch],
                    dtype=torch.long)

                pos_score = self.net(u_batch, i_batch)
                neg_score = self.net(u_batch, j_batch)
                loss = -F.logsigmoid(pos_score - neg_score).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
            print(f'  TwoTower epoch {epoch+1}/{self.epochs}, avg BPR loss: {total_loss/n_batches:.4f}')

        # precompute all item embeddings once for fast scoring/retrieval
        self.net.eval()
        with torch.no_grad():
            all_item_ids = torch.arange(n_items, dtype=torch.long)
            self.item_embeddings = self.net.get_item_vec(all_item_ids).numpy()
        return self

    @staticmethod
    def _sample_neg(rng, u, n_items, user_pos):
        j = rng.integers(0, n_items)
        while j in user_pos.get(u, ()):
            j = rng.integers(0, n_items)
        return j

    def get_user_embedding(self, u):
        self.net.eval()
        with torch.no_grad():
            uid = torch.tensor([u], dtype=torch.long)
            return self.net.get_user_vec(uid).numpy()[0]

    def score(self, u, candidate_items):
        user_vec = self.get_user_embedding(u)
        return self.item_embeddings[candidate_items] @ user_vec