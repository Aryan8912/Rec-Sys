import numpy as np
import pandas as pd


class InteractionData:
    """
    Wraps a raw interactions dataframe + optional item/user side-features.
    Handles: contiguous re-indexing (needed for embedding tables),
    time-based leave-last-out split, and negative sampling for training.
    """

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame = None,
                 users: pd.DataFrame = None, implicit_threshold: float = None):
        """
        ratings: columns [user_id, item_id, rating, timestamp]
        implicit_threshold: if set, ratings >= threshold become positive
            implicit feedback (1), rest dropped. If None, keep explicit ratings.
        """
        self.raw = ratings.copy()
        self.movies = movies
        self.users = users
        self.implicit_threshold = implicit_threshold

        # Re-index users/items to contiguous 0..N-1 ids (required for nn.Embedding)
        self.user_ids = np.sort(self.raw.user_id.unique())
        self.item_ids = np.sort(self.raw.item_id.unique())
        self.uid2idx = {u: i for i, u in enumerate(self.user_ids)}
        self.iid2idx = {it: i for i, it in enumerate(self.item_ids)}

        df = self.raw.copy()
        df['u'] = df.user_id.map(self.uid2idx)
        df['i'] = df.item_id.map(self.iid2idx)

        if implicit_threshold is not None:
            df = df[df.rating >= implicit_threshold].copy()
            df['label'] = 1.0
        else:
            df['label'] = df.rating.astype(float)

        self.df = df.sort_values('timestamp').reset_index(drop=True)
        self.n_users = len(self.user_ids)
        self.n_items = len(self.item_ids)

    def time_split(self, test_frac: float = 0.2):
        """
        Leave-last-out style split per user: each user's most recent
        interactions go to test. Realistic eval (no future leakage into
        train), unlike a random row split.
        """
        train_rows, test_rows = [], []
        for _, grp in self.df.groupby('u'):
            grp = grp.sort_values('timestamp')
            n_test = max(1, int(len(grp) * test_frac))
            train_rows.append(grp.iloc[:-n_test])
            test_rows.append(grp.iloc[-n_test:])
        train = pd.concat(train_rows).reset_index(drop=True)
        test = pd.concat(test_rows).reset_index(drop=True)
        return train, test

    def item_genre_matrix(self):
        """Multi-hot genre matrix aligned to item index `i`, for content features."""
        if self.movies is None:
            return None
        m = self.movies.copy()
        m = m[m.item_id.isin(self.iid2idx)]
        m['i'] = m.item_id.map(self.iid2idx)
        all_genres = sorted(set(g for gs in m.genres.str.split('|') for g in gs))
        g2idx = {g: i for i, g in enumerate(all_genres)}
        mat = np.zeros((self.n_items, len(all_genres)), dtype=np.float32)
        for _, row in m.iterrows():
            for g in row.genres.split('|'):
                mat[row['i'], g2idx[g]] = 1.0
        return mat, all_genres


if __name__ == '__main__':
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    users = pd.read_csv('data/users.csv')

    data = InteractionData(ratings, movies, users, implicit_threshold=4.0)
    print(f'Users: {data.n_users}, Items: {data.n_items}, '
          f'Positive interactions: {len(data.df)}')

    train, test = data.time_split(test_frac=0.2)
    print(f'Train: {len(train)}, Test: {len(test)}')
    print(f'Train users: {train.u.nunique()}, Test users: {test.u.nunique()}')

    genre_mat, genre_names = data.item_genre_matrix()
    print(f'Genre matrix: {genre_mat.shape}, genres: {genre_names[:5]}...')