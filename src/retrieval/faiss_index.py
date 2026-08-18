import numpy as np
import faiss


class FaissRetriever:
    def __init__(self, embedding_dim):
        self.dim = embedding_dim
        self.index = faiss.IndexFlatIP(self.dim)
        self.n_items = 0

    def build(self, item_embeddings: np.ndarray):
        """item_embeddings: shape (n_items, embedding_dim), from a fitted
        TwoTowerRecommender's .item_embeddings attribute."""
        embeddings = np.ascontiguousarray(item_embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.n_items = embeddings.shape[0]
        return self

    def retrieve(self, user_vector: np.ndarray, top_k: int = 200):
        """
        user_vector: shape (embedding_dim,), from TwoTowerRecommender.get_user_embedding(u)
        Returns: (candidate_item_indices, scores), both length top_k
            (or fewer if top_k > n_items)
        """
        top_k = min(top_k, self.n_items)
        query = np.ascontiguousarray(user_vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, top_k)
        return indices[0], scores[0]


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    import pandas as pd
    from data_pipeline import InteractionData
    from models.two_tower import TwoTowerRecommender

    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    users = pd.read_csv('data/users.csv')
    data = InteractionData(ratings, movies, users, implicit_threshold=4.0)
    train, test = data.time_split(test_frac=0.2)

    tt = TwoTowerRecommender(embedding_dim=32, lr=0.005, epochs=15).fit(
        train, data.n_users, data.n_items)

    retriever = FaissRetriever(embedding_dim=32).build(tt.item_embeddings)

    user_vec = tt.get_user_embedding(0)
    candidate_idx, scores = retriever.retrieve(user_vec, top_k=200)
    print(f'Retrieved {len(candidate_idx)} candidates for user 0')
    print('Top-5 candidate indices:', candidate_idx[:5])
    print('Top-5 scores:', scores[:5])

    movies_indexed = movies[movies.item_id.isin(data.iid2idx)].copy()
    movies_indexed['i'] = movies_indexed.item_id.map(data.iid2idx)
    lookup = movies_indexed.set_index('i')['title'].to_dict()
    print('Top-5 titles:', [lookup.get(i, '?') for i in candidate_idx[:5]])

    brute_recs = tt.recommend(0, 5)
    print('Brute-force top-5 (for comparison):', brute_recs)