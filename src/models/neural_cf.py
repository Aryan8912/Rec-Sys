import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseRecommender


class NeuralCFNet(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=32, hidden_dims=(64, 32)):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2  
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        x = torch.cat([u, i], dim=-1)
        return self.mlp(x).squeeze(-1)


class NeuralCFRecommender(BaseRecommender):
    def __init__(self, embedding_dim=32, hidden_dims=(64, 32), lr=0.005,
                 epochs=15, batch_size=1024, seed=42):
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, train_df, n_users, n_items):
        torch.manual_seed(self.seed)
        self.n_items = n_items
        self.net = NeuralCFNet(n_users, n_items, self.embedding_dim, self.hidden_dims)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        users = train_df.u.values
        items = train_df.i.values
        user_pos = train_df.groupby('u')['i'].apply(set).to_dict()
        n = len(train_df)
        rng = np.random.default_rng(self.seed)

        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                u_np, i_np = users[idx], items[idx]
                j_list = []
                for u in u_np:
                    j = rng.integers(0, n_items)
                    up = user_pos.get(u, ())
                    while j in up:
                        j = rng.integers(0, n_items)
                    j_list.append(j)

                u_batch = torch.tensor(u_np, dtype=torch.long)
                i_batch = torch.tensor(i_np, dtype=torch.long)
                j_batch = torch.tensor(j_list, dtype=torch.long)

                pos_score = self.net(u_batch, i_batch)
                neg_score = self.net(u_batch, j_batch)
                loss = -F.logsigmoid(pos_score - neg_score).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
            print(f'  NeuralCF epoch {epoch+1}/{self.epochs}, avg BPR loss: {total_loss/n_batches:.4f}')

        self.net.eval()
        return self

    def score(self, u, candidate_items):
        self.net.eval()
        with torch.no_grad():
            u_batch = torch.full((len(candidate_items),), u, dtype=torch.long)
            i_batch = torch.tensor(candidate_items, dtype=torch.long)
            scores = self.net(u_batch, i_batch)
        return scores.numpy()

    def rerank(self, u, candidate_items, k):
        scores = self.score(u, candidate_items)
        order = np.argsort(-scores)[:k]
        return np.array(candidate_items)[order]


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    import pandas as pd
    from data_pipeline import InteractionData
    from evaluation.metrics import evaluate_model

    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    users = pd.read_csv('data/users.csv')
    data = InteractionData(ratings, movies, users, implicit_threshold=4.0)
    train, test = data.time_split(test_frac=0.2)
    train_seen = train.groupby('u')['i'].apply(set).to_dict()

    ncf = NeuralCFRecommender(embedding_dim=32, epochs=15).fit(
        train, data.n_users, data.n_items)

    recs = ncf.recommend(0, 10, exclude=train_seen.get(0, set()))
    movies_indexed = movies[movies.item_id.isin(data.iid2idx)].copy()
    movies_indexed['i'] = movies_indexed.item_id.map(data.iid2idx)
    lookup = movies_indexed.set_index('i')['title'].to_dict()
    print('Top-10 for user 0:', [lookup.get(i, '?') for i in recs])

    print('\nEvaluating NeuralCF as a standalone scorer (whole-catalog, for comparison):')
    r = evaluate_model(ncf, test, data.n_items, k=10, train_seen=train_seen)
    print(f"  P@10={r['precision@k']:.4f}  R@10={r['recall@k']:.4f}  "
          f"NDCG@10={r['ndcg@k']:.4f}  Coverage={r['coverage']:.4f}")

    print('\nDemo: Stage 1 (FAISS) -> Stage 2 (NeuralCF rerank) pipeline:')
    from models.two_tower import TwoTowerRecommender
    from retrieval.faiss_index import FaissRetriever

    tt = TwoTowerRecommender(embedding_dim=32, lr=0.005, epochs=15).fit(
        train, data.n_users, data.n_items)
    retriever = FaissRetriever(embedding_dim=32).build(tt.item_embeddings)
    user_vec = tt.get_user_embedding(0)
    candidates, _ = retriever.retrieve(user_vec, top_k=200)
    seen = train_seen.get(0, set())
    candidates = [c for c in candidates if c not in seen]
    final_recs = ncf.rerank(0, candidates, k=10)
    print('Final top-10 (Two-Tower retrieval -> NeuralCF rerank):',
          [lookup.get(i, '?') for i in final_recs])