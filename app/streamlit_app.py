import os
import pickle
import sys

import faiss
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from retrieval.faiss_index import FaissRetriever
from rag.explain import RecommendationExplainer

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')


@st.cache_resource
def load_artifacts():
    def _load(fname):
        with open(os.path.join(ARTIFACTS_DIR, fname), 'rb') as f:
            return pickle.load(f)

    models = {
        'Popularity': _load('popularity_model.pkl'),
        'Item-Item CF': _load('item_cf_model.pkl'),
        'Matrix Factorization': _load('mf_model.pkl'),
        'Two-Tower': _load('two_tower_model.pkl'),
        'Neural CF': _load('neural_cf_model.pkl'),
    }

    id_maps = _load('id_maps.pkl')

    # FAISS index has its own binary format, not pickle
    faiss_raw_index = faiss.read_index(os.path.join(ARTIFACTS_DIR, 'faiss_index.bin'))
    retriever = FaissRetriever(embedding_dim=id_maps['embedding_dim'])
    retriever.index = faiss_raw_index
    retriever.n_items = faiss_raw_index.ntotal

    return {
        'models': models,
        'retriever': retriever,
        'movies_lookup': _load('movies_lookup.pkl'),
        'id_maps': id_maps,
        'train_seen': _load('train_seen.pkl'),
        'eval_report': _load('eval_report.pkl'),
    }


@st.cache_resource
def load_explainer():
    return RecommendationExplainer()


def get_recommendations(strategy, model_choice, user_idx, k, artifacts):
    """Returns (item_indices, note) where note explains what happened
    (e.g. full-funnel candidate count) for display."""
    models = artifacts['models']
    train_seen = artifacts['train_seen']
    exclude = train_seen.get(user_idx, set())

    if strategy == 'Single model':
        model = models[model_choice]
        recs = model.recommend(user_idx, k, exclude=exclude)
        return recs, None

    # Full funnel: Two-Tower retrieval -> FAISS -> NeuralCF rerank
    tt = models['Two-Tower']
    ncf = models['Neural CF']
    retriever = artifacts['retriever']

    user_vec = tt.get_user_embedding(user_idx)
    candidates, _ = retriever.retrieve(user_vec, top_k=200)
    candidates = [int(c) for c in candidates if c not in exclude]
    if not candidates:
        return np.array([]), 'No candidates left after excluding seen items.'

    final = ncf.rerank(user_idx, candidates, k=min(k, len(candidates)))
    note = f'Retrieved {len(candidates)} candidates via Two-Tower+FAISS, re-ranked by Neural CF'
    return final, note


def inject_youtube_theme():
    st.markdown("""
    <style>
    .stApp {
        background-color: #0f0f0f;
        color: #f1f1f1;
    }
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        color: #ffffff !important;
    }
    .rec-card {
        background-color: #212121;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #FF0000;
    }
    .rec-rank {
        color: #aaaaaa;
        font-size: 0.85em;
        font-weight: 600;
    }
    .rec-title {
        color: #ffffff;
        font-size: 1.15em;
        font-weight: 600;
        margin: 2px 0;
    }
    .rec-genres {
        color: #aaaaaa;
        font-size: 0.9em;
    }
    .rec-explain {
        color: #f1f1f1;
        background-color: #181818;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.92em;
        border: 1px solid #303030;
    }
    div.stButton > button {
        background-color: #FF0000;
        color: white;
        border-radius: 20px;
        border: none;
        font-weight: 600;
        padding: 8px 24px;
    }
    div.stButton > button:hover {
        background-color: #CC0000;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title='Universal Recommender', layout='centered',
                        page_icon='🎬')
    inject_youtube_theme()
    st.title('▶️ Universal Recommender')
    st.caption('Domain-agnostic recommendation engine — demoed on MovieLens-1M · '
               'UI inspired by YouTube\'s recommendation feed')

    artifacts = load_artifacts()
    movies_lookup = artifacts['movies_lookup']
    id_maps = artifacts['id_maps']
    train_seen = artifacts['train_seen']
    n_users = id_maps['n_users']

    col1, col2 = st.columns(2)
    with col1:
        user_idx = st.number_input(
            f'User ID (0 to {n_users - 1})', min_value=0, max_value=n_users - 1,
            value=0, step=1,
        )
    with col2:
        strategy = st.selectbox(
            'Recommendation strategy',
            ['Single model', 'Full funnel (Two-Tower \u2192 FAISS \u2192 NeuralCF)'],
        )

    model_choice = None
    if strategy == 'Single model':
        model_choice = st.selectbox(
            'Model', ['Popularity', 'Item-Item CF', 'Matrix Factorization',
                      'Two-Tower', 'Neural CF'], index=1)

    k = st.slider('Number of recommendations', min_value=5, max_value=20, value=10)
    explain_top = st.checkbox('Generate AI explanation for the #1 recommendation', value=False)

    if st.button('Get Recommendations', type='primary'):
        recs, note = get_recommendations(strategy, model_choice, user_idx, k, artifacts)
        if note:
            st.info(note)

        label = model_choice if strategy == 'Single model' else 'Full Funnel'
        st.subheader(f'Top {len(recs)} recommendations for User {user_idx} ({label})')

        for rank, item_idx in enumerate(recs, start=1):
            info = movies_lookup.get(int(item_idx), {'title': 'Unknown', 'genres': ''})
            genres_readable = info['genres'].replace('|', ' • ')

            explanation_html = ''
            if rank == 1 and explain_top:
                explainer = load_explainer()
                history = [movies_lookup.get(i, {}).get('title', '')
                           for i in list(train_seen.get(user_idx, set()))[-3:]]
                with st.spinner('Generating explanation...'):
                    explanation = explainer.generate_explanation(
                        history, info['title'], info['genres'])
                explanation_html = f'<div class="rec-explain">💬 {explanation}</div>'

            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-rank">#{rank}</div>
                <div class="rec-title">{info['title']}</div>
                <div class="rec-genres">{genres_readable}</div>
                {explanation_html}
            </div>
            """, unsafe_allow_html=True)

        with st.expander('What has this user already liked? (excluded from recs above)'):
            seen = list(train_seen.get(user_idx, set()))[:15]
            for item_idx in seen:
                info = movies_lookup.get(item_idx, {'title': 'Unknown', 'genres': ''})
                st.write(f"- {info['title']} ({info['genres']})")

    with st.expander('Model evaluation metrics (Precision@10 / Recall@10 / NDCG@10 / Coverage)'):
        report = artifacts['eval_report']
        for name, r in report.items():
            st.write(f"**{name}** — P@10={r['precision@k']:.4f}  "
                     f"R@10={r['recall@k']:.4f}  NDCG@10={r['ndcg@k']:.4f}  "
                     f"Coverage={r['coverage']:.4f}")


if __name__ == '__main__':
    main()