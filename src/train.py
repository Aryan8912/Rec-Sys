import os
import pickle
import sys

import faiss
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import InteractionData
from models.popularity import PopularityRecommender
from models.item_cf import ItemItemCF
from models.matrix_factorization import MatrixFactorization
from models.two_tower import TwoTowerRecommender
from models.neural_cf import NeuralCFRecommender
from retrieval.faiss_index import FaissRetriever
from evaluation.metrics import evaluate_model

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
EMBEDDING_DIM = 32


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print('Loading data...')
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    users = pd.read_csv('data/users.csv')

    data = InteractionData(ratings, movies, users, implicit_threshold=4.0)
    train, test = data.time_split(test_frac=0.2)
    train_seen = train.groupby('u')['i'].apply(set).to_dict()

    print(f'Users: {data.n_users}, Items: {data.n_items}')
    print(f'Train: {len(train)}, Test: {len(test)}\n')

    print('Training Popularity...')
    pop_model = PopularityRecommender().fit(train, data.n_users, data.n_items)

    print('Training Item-Item CF...')
    icf_model = ItemItemCF().fit(train, data.n_users, data.n_items)

    print('Training Matrix Factorization...')
    mf_model = MatrixFactorization(n_factors=32, epochs=10).fit(
        train, data.n_users, data.n_items)

    print('Training Two-Tower...')
    tt_model = TwoTowerRecommender(embedding_dim=EMBEDDING_DIM, lr=0.005, epochs=15).fit(
        train, data.n_users, data.n_items)

    print('Training Neural CF...')
    ncf_model = NeuralCFRecommender(embedding_dim=EMBEDDING_DIM, epochs=15).fit(
        train, data.n_users, data.n_items)

    print('\nBuilding FAISS index over Two-Tower item embeddings...')
    retriever = FaissRetriever(embedding_dim=EMBEDDING_DIM).build(tt_model.item_embeddings)

    print('\nEvaluating all models...')
    report = {}
    for name, model in [('Popularity', pop_model), ('Item-Item CF', icf_model),
                         ('Matrix Factorization', mf_model), ('Two-Tower', tt_model),
                         ('Neural CF', ncf_model)]:
        report[name] = evaluate_model(model, test, data.n_items, k=10, train_seen=train_seen)
        r = report[name]
        print(f"  {name:<25} P@10={r['precision@k']:.4f}  R@10={r['recall@k']:.4f}  "
              f"NDCG@10={r['ndcg@k']:.4f}  Coverage={r['coverage']:.4f}")

    print('\nSaving artifacts...')

    for fname, obj in [
        ('popularity_model.pkl', pop_model),
        ('item_cf_model.pkl', icf_model),
        ('mf_model.pkl', mf_model),
        ('two_tower_model.pkl', tt_model),
        ('neural_cf_model.pkl', ncf_model),
    ]:
        with open(os.path.join(ARTIFACTS_DIR, fname), 'wb') as f:
            pickle.dump(obj, f)

    # FAISS has its own binary format — don't pickle the index itself
    faiss.write_index(retriever.index, os.path.join(ARTIFACTS_DIR, 'faiss_index.bin'))

    movies_indexed = movies[movies.item_id.isin(data.iid2idx)].copy()
    movies_indexed['i'] = movies_indexed.item_id.map(data.iid2idx)
    movies_lookup = movies_indexed.set_index('i')[['title', 'genres']].to_dict('index')
    with open(os.path.join(ARTIFACTS_DIR, 'movies_lookup.pkl'), 'wb') as f:
        pickle.dump(movies_lookup, f)

    id_maps = {
        'uid2idx': data.uid2idx, 'iid2idx': data.iid2idx,
        'user_ids': data.user_ids, 'item_ids': data.item_ids,
        'n_users': data.n_users, 'n_items': data.n_items,
        'embedding_dim': EMBEDDING_DIM,
    }
    with open(os.path.join(ARTIFACTS_DIR, 'id_maps.pkl'), 'wb') as f:
        pickle.dump(id_maps, f)

    with open(os.path.join(ARTIFACTS_DIR, 'train_seen.pkl'), 'wb') as f:
        pickle.dump(train_seen, f)

    with open(os.path.join(ARTIFACTS_DIR, 'eval_report.pkl'), 'wb') as f:
        pickle.dump(report, f)

    print(f'Done. Artifacts saved to {os.path.abspath(ARTIFACTS_DIR)}/')


if __name__ == '__main__':
    main()