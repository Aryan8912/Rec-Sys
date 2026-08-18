import numpy as np


def precision_at_k(recommended, relevant, k):
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(top_k)
              if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def coverage(all_recommended_lists, n_items):
    recommended_items = set()
    for rec_list in all_recommended_lists:
        recommended_items.update(rec_list)
    return len(recommended_items) / n_items


def evaluate_model(model, test_df, n_items, k=10, train_seen=None):
    precisions, recalls, ndcgs = [], [], []
    all_rec_lists = []

    for u, grp in test_df.groupby('u'):
        relevant = set(grp.i.values)
        exclude = train_seen.get(u, set()) if train_seen else None
        recommended = model.recommend(u, k, exclude=exclude)

        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        all_rec_lists.append(recommended)

    return {
        'precision@k': np.mean(precisions),
        'recall@k': np.mean(recalls),
        'ndcg@k': np.mean(ndcgs),
        'coverage': coverage(all_rec_lists, n_items),
        'n_users_evaluated': len(precisions),
    }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    import pandas as pd
    from data_pipeline import InteractionData
    from models.popularity import PopularityRecommender
    from models.item_cf import ItemItemCF
    from models.matrix_factorization import MatrixFactorization

    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    users = pd.read_csv('data/users.csv')
    data = InteractionData(ratings, movies, users, implicit_threshold=4.0)
    train, test = data.time_split(test_frac=0.2)

    train_seen = train.groupby('u')['i'].apply(set).to_dict()

    models = {
        'Popularity': PopularityRecommender().fit(train, data.n_users, data.n_items),
        'Item-Item CF': ItemItemCF().fit(train, data.n_users, data.n_items),
        'Matrix Factorization': MatrixFactorization(n_factors=32, epochs=10).fit(
            train, data.n_users, data.n_items),
    }

    print(f"\n{'Model':<25} {'Precision@10':<15} {'Recall@10':<15} {'NDCG@10':<15} {'Coverage':<10}")
    print('-' * 80)
    for name, model in models.items():
        results = evaluate_model(model, test, data.n_items, k=10, train_seen=train_seen)
        print(f"{name:<25} {results['precision@k']:<15.4f} {results['recall@k']:<15.4f} "
              f"{results['ndcg@k']:<15.4f} {results['coverage']:<10.4f}")