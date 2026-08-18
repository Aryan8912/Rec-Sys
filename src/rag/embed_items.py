import os
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts')


def build_item_text(movies_df: pd.DataFrame) -> pd.Series:
    """Combine title + genres into one text string per item for embedding."""
    genres_readable = movies_df['genres'].str.replace('|', ', ', regex=False)
    return movies_df['title'] + ' — ' + genres_readable


def embed_items(movies_df: pd.DataFrame, iid2idx: dict, model_name='all-MiniLM-L6-v2'):
    """
    Returns (embeddings, item_texts) where embeddings is shape
    (n_items, embedding_dim), aligned to the re-indexed item id `i`
    (same indexing as everywhere else in this project).
    """
    movies_indexed = movies_df[movies_df.item_id.isin(iid2idx)].copy()
    movies_indexed['i'] = movies_indexed.item_id.map(iid2idx)
    movies_indexed = movies_indexed.sort_values('i').reset_index(drop=True)

    item_texts = build_item_text(movies_indexed)

    print(f'Loading sentence-transformer model: {model_name}...')
    model = SentenceTransformer(model_name)

    print(f'Embedding {len(item_texts)} items...')
    embeddings = model.encode(item_texts.tolist(), show_progress_bar=True,
                               convert_to_numpy=True)
    return embeddings, item_texts.tolist()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_pipeline import InteractionData

    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    users = pd.read_csv('data/users.csv')
    data = InteractionData(ratings, movies, users, implicit_threshold=4.0)

    embeddings, item_texts = embed_items(movies, data.iid2idx)
    print(f'Embeddings shape: {embeddings.shape}')
    print(f'Example item text: "{item_texts[0]}"')

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACTS_DIR, 'item_text_embeddings.pkl'), 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'item_texts': item_texts}, f)
    print(f'Saved to {ARTIFACTS_DIR}/item_text_embeddings.pkl')

    from numpy.linalg import norm
    norms = norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / norms
    sims = normed @ normed[0]
    top5 = np.argsort(-sims)[:5]
    print(f'\nMost similar to "{item_texts[0]}":')
    for idx in top5:
        print(f'  {item_texts[idx]}  (sim={sims[idx]:.3f})')