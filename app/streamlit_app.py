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


def inject_youtube_theme():
    st.markdown("""
    <style>
    .stApp { background-color: #0f0f0f; color: #f1f1f1; }
    h1, h2, h3 { font-family: 'Roboto', sans-serif; color: #ffffff !important; }
    .rec-card {
        background-color: #212121; border-radius: 12px; padding: 16px 20px;
        margin-bottom: 12px; border-left: 4px solid #FF0000;
    }
    .rec-rank { color: #aaaaaa; font-size: 0.85em; font-weight: 600; }
    .rec-title { color: #ffffff; font-size: 1.15em; font-weight: 600; margin: 2px 0; }
    .rec-genres { color: #aaaaaa; font-size: 0.9em; }
    .rec-score { color: #6ba8ff; font-size: 0.85em; }
    .rec-explain {
        color: #f1f1f1; background-color: #181818; border-radius: 8px;
        padding: 10px 14px; margin-top: 8px; font-size: 0.92em; border: 1px solid #303030;
    }
    .fold-in-note {
        color: #ffb84d; background-color: #2a1f10; border-radius: 8px;
        padding: 8px 14px; margin-bottom: 12px; font-size: 0.85em; border: 1px solid #4a3520;
    }
    div.stButton > button {
        background-color: #FF0000; color: white; border-radius: 20px;
        border: none; font-weight: 600; padding: 8px 24px;
    }
    div.stButton > button:hover { background-color: #CC0000; color: white; }
    </style>
    """, unsafe_allow_html=True)


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
def load_text_embeddings():
    """Returns None if the file doesn't exist yet — Tab 2 shows a clear
    message rather than crashing, since this artifact is built separately
    (python src/rag/embed_items.py) from the main training run."""
    path = os.path.join(ARTIFACTS_DIR, 'item_text_embeddings.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_text_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_explainer():
    return RecommendationExplainer()


def get_recommendations_for_query(strategy, model_choice, liked_items, k, artifacts):
    models = artifacts['models']

    if strategy == 'Single model':
        model = models[model_choice]
        recs = model.recommend_for_items(liked_items, k, exclude=set(liked_items))
        return recs, None

    tt = models['Two-Tower']
    ncf = models['Neural CF']
    retriever = artifacts['retriever']

    pseudo_user_vec = tt.get_pseudo_user_embedding(liked_items)
    candidates, _ = retriever.retrieve(pseudo_user_vec, top_k=200)
    candidates = [int(c) for c in candidates if c not in set(liked_items)]
    if not candidates:
        return np.array([]), 'No candidates left after excluding selected items.'

    final = ncf.rerank_for_items(liked_items, candidates, k=min(k, len(candidates)))
    note = f'Retrieved {len(candidates)} candidates via Two-Tower+FAISS, re-ranked by Neural CF'
    return final, note


def render_personalized_tab(artifacts):
    movies_lookup = artifacts['movies_lookup']

    st.write('Search and pick a few movies you actually enjoy — recommendations are '
             'computed live from your picks through the trained models below, not from '
             'a pre-existing training-set user.')

    all_titles = sorted(set(v['title'] for v in movies_lookup.values()))
    title_to_idx = {v['title']: k for k, v in movies_lookup.items()}

    picked_titles = st.multiselect(
        'Movies you like (type to search)', options=all_titles, key='picked_titles',
        help='Pick at least 1-2 for a meaningful recommendation.',
    )
    liked_items = [title_to_idx[t] for t in picked_titles]

    strategy = st.selectbox(
        'Recommendation strategy',
        ['Single model', 'Full funnel (Two-Tower \u2192 FAISS \u2192 NeuralCF)'],
        key='strategy_select',
    )

    model_choice = None
    if strategy == 'Single model':
        model_choice = st.selectbox(
            'Model', ['Popularity', 'Item-Item CF', 'Matrix Factorization',
                      'Two-Tower', 'Neural CF'], index=1, key='model_select')

    k = st.slider('Number of recommendations', min_value=5, max_value=20, value=10, key='k_slider')
    explain_top = st.checkbox('Generate AI explanation for the #1 recommendation',
                               value=False, key='explain_check')

    if st.button('Get Recommendations', type='primary', key='funnel_btn'):
        if not liked_items and (model_choice != 'Popularity'):
            st.warning('Pick at least one movie, or choose Popularity, which needs no input.')
            return

        approx_notes = {
            'Matrix Factorization': 'Matrix Factorization has no vector for a brand-new '
                'query \u2014 this approximates one by averaging your picks\u2019 learned '
                'vectors (a standard "fold-in" technique).',
            'Two-Tower': 'Two-Tower has no trained embedding for a brand-new query \u2014 '
                'this approximates one by averaging your picks\u2019 item embeddings.',
            'Neural CF': 'Neural CF\u2019s scoring isn\u2019t a simple average, so this is '
                'the roughest approximation of the four personalized models \u2014 treat '
                'its results with the most skepticism.',
        }
        active_model = model_choice if strategy == 'Single model' else 'Two-Tower'
        if active_model in approx_notes:
            st.markdown(f'<div class="fold-in-note">\u26A0\uFE0F {approx_notes[active_model]}</div>',
                        unsafe_allow_html=True)

        recs, note = get_recommendations_for_query(strategy, model_choice, liked_items, k, artifacts)
        if note:
            st.info(note)

        label = model_choice if strategy == 'Single model' else 'Full Funnel'
        st.subheader(f'Top {len(recs)} recommendations ({label})')

        for rank, item_idx in enumerate(recs, start=1):
            info = movies_lookup.get(int(item_idx), {'title': 'Unknown', 'genres': ''})
            genres_readable = info['genres'].replace('|', ' \u2022 ')

            explanation_html = ''
            if rank == 1 and explain_top:
                explainer = load_explainer()
                history = [movies_lookup.get(i, {}).get('title', '') for i in liked_items[-3:]]
                with st.spinner('Generating explanation...'):
                    explanation = explainer.generate_explanation(
                        history, info['title'], info['genres'])
                explanation_html = f'<div class="rec-explain">\U0001F4AC {explanation}</div>'

            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-rank">#{rank}</div>
                <div class="rec-title">{info['title']}</div>
                <div class="rec-genres">{genres_readable}</div>
                {explanation_html}
            </div>
            """, unsafe_allow_html=True)

    with st.expander('Model evaluation metrics (Precision@10 / Recall@10 / NDCG@10 / Coverage)'):
        st.caption('Computed on the held-out MovieLens test set during training \u2014 not on '
                   'live queries above, which have no ground truth to score against.')
        report = artifacts['eval_report']
        for name, r in report.items():
            st.write(f"**{name}** \u2014 P@10={r['precision@k']:.4f}  "
                     f"R@10={r['recall@k']:.4f}  NDCG@10={r['ndcg@k']:.4f}  "
                     f"Coverage={r['coverage']:.4f}")


def render_semantic_search_tab():
    st.write('Search the catalog by natural-language description \u2014 this tests '
             '**semantic content search**, not the recommender models. A movie can '
             'rank highly here purely because its title/genre text matches your '
             'query, regardless of whether anyone actually rated it well.')

    text_data = load_text_embeddings()
    if text_data is None:
        st.error('`artifacts/item_text_embeddings.pkl` not found. Run '
                 '`python src/rag/embed_items.py` first to generate it.')
        return

    embeddings = text_data['embeddings']
    item_texts = text_data['item_texts']

    query = st.text_input(
        'Describe what you want to watch',
        value='mind-bending sci-fi with time travel', key='search_query',
    )
    search_k = st.slider('Results to show', min_value=3, max_value=15, value=5, key='search_k')

    if st.button('Search', type='primary', key='search_btn'):
        with st.spinner('Encoding query and searching...'):
            encoder = load_text_encoder()
            query_vec = encoder.encode(query, convert_to_numpy=True)

            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normed = embeddings / np.clip(norms, 1e-9, None)
            query_norm = query_vec / max(np.linalg.norm(query_vec), 1e-9)
            scores = normed @ query_norm

            top_idx = np.argsort(-scores)[:search_k]

        st.subheader(f'Top {len(top_idx)} semantic matches')
        for rank, idx in enumerate(top_idx, start=1):
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-rank">#{rank}</div>
                <div class="rec-title">{item_texts[idx]}</div>
                <div class="rec-score">similarity: {scores[idx]:.4f}</div>
            </div>
            """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title='Universal Recommender', layout='centered', page_icon='🎬')
    inject_youtube_theme()
    st.title('▶️ Universal Recommender')
    st.caption('Domain-agnostic recommendation engine — demoed on MovieLens-1M')

    artifacts = load_artifacts()

    tab1, tab2 = st.tabs(['\U0001F3AF Personalized Recommender', '\U0001F50D Semantic Search'])
    with tab1:
        render_personalized_tab(artifacts)
    with tab2:
        render_semantic_search_tab()


if __name__ == '__main__':
    main()