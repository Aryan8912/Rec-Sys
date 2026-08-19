# Universal Recommender — Documentation

## 1. Problem Statement

Design and build a functional, deployable recommendation system from
scratch, demonstrating a clear recommendation strategy, sound engineering
practices, and honest evaluation — including where the system fails, not
just where it succeeds.

## 2. Use Case & Motivation

Movie recommendations, built on an architecture designed to be
domain-agnostic: every core file (`data_pipeline.py`, all files in
`models/`) works with any dataset that reduces to `(user_id, item_id,
event, timestamp)`, not just movies. MovieLens-1M was chosen as the
dataset — free, well-documented, and rich enough (ratings + genres +
demographics) to exercise the full pipeline.

## 3. Approach

Built incrementally, verifying each component before building on top of it:

1. Data pipeline → loading, re-indexing, time-based split
2. Three classical baselines (Popularity, Item-Item CF, Matrix Factorization)
3. Evaluation framework (Precision@K, Recall@K, NDCG@K, Coverage)
4. A deployable checkpoint (baselines + Streamlit UI) before adding
   complexity — deliberately prioritized to de-risk the mandatory
   deployment requirement
5. Two-stage deep funnel: Two-Tower retrieval + FAISS + Neural CF ranking
6. RAG explanation layer: semantic item embeddings + LLM-generated explanations
7. Deployment to Streamlit Community Cloud

## 4. System Architecture

```
                         ┌─────────────────────┐
All items (3,706)  ────▶ │  Stage 1: Retrieval  │────▶  ~200 candidates
                         │  (Two-Tower + FAISS) │
                         └─────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Stage 2: Ranking    │────▶  Top-K shown
                         │  (Neural CF)         │
                         └─────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  RAG Explanation     │────▶  Natural-language
                         │  (embeddings + HF    │      "why recommended"
                         │   Inference API)     │
                         └─────────────────────┘
```

**Offline (batch training)** — `src/train.py` loads data, trains all 5
models (3 baselines + Two-Tower + Neural CF), evaluates each, builds the
FAISS index, and saves everything to `artifacts/` so the app never
retrains at startup — this matters specifically because Streamlit
Community Cloud's free tier has limited RAM and build time.

**Online (serving)** — `app/streamlit_app.py` loads pre-trained artifacts
and offers two modes: single-model recommendations, or the full funnel
(Two-Tower embedding → FAISS candidate retrieval → Neural CF re-ranking),
with an optional LLM-generated explanation for the top result.

### File structure

```
reco/
├── data/                       ratings.csv, movies.csv, users.csv (MovieLens-1M)
├── src/
│   ├── data_pipeline.py        loader, time-split, feature matrices
│   ├── models/
│   │   ├── base.py             shared fit/score/recommend interface
│   │   ├── popularity.py       baseline: interaction-count ranking
│   │   ├── item_cf.py          baseline: sparsified top-K item-item cosine similarity
│   │   ├── matrix_factorization.py   baseline: BPR-SGD latent factors
│   │   ├── two_tower.py        deep: separate user/item towers, BPR loss
│   │   └── neural_cf.py        deep: concatenation + MLP, BPR loss
│   ├── retrieval/
│   │   └── faiss_index.py      wraps Two-Tower embeddings in a FAISS index
│   ├── rag/
│   │   ├── embed_items.py      sentence-transformers semantic item embeddings
│   │   └── explain.py          HF Inference API explanation generator + fallback
│   ├── evaluation/
│   │   └── metrics.py          Precision@K, Recall@K, NDCG@K, Coverage
│   └── train.py                 orchestrates training + artifact saving
├── app/
│   └── streamlit_app.py        the deployed UI
├── artifacts/                  trained models, FAISS index, eval report (committed to repo)
├── docs/
│   ├── DOCUMENTATION.md        this file
│   └── COMPARISON.md           product comparison vs. YouTube's architecture
├── requirements.txt
└── README.md
```

## 5. Recommendation Methodology

Five models, in increasing sophistication, all trained on the same
implicit-feedback signal (ratings ≥4 out of 5 treated as a positive interaction):

| Model | Method |
|---|---|
| Popularity | Interaction-count ranking, no personalization — cold-start fallback |
| Item-Item CF | Cosine similarity between items from co-occurrence, sparsified to each item's top-50 neighbors |
| Matrix Factorization | Latent user/item vectors learned via BPR (Bayesian Personalized Ranking) pairwise SGD |
| Two-Tower | Separate user/item embedding towers, BPR loss, trained for FAISS-compatible retrieval |
| Neural CF | Concatenated user+item embeddings through an MLP, BPR loss — used for Stage 2 re-ranking |

**Retrieval**: FAISS `IndexFlatIP` (exact inner-product search) over
Two-Tower item embeddings, retrieving ~200 candidates per user in
milliseconds.

**RAG explanations**: item titles+genres embedded with
`sentence-transformers/all-MiniLM-L6-v2` (384-dim) for semantic similarity;
natural-language explanations generated via a live call to Hugging Face's
free Inference API (`Qwen/Qwen2.5-7B-Instruct`), with a deterministic
template fallback if the API call fails for any reason.

## 6. Dataset Selection

**MovieLens-1M**: 1,000,209 ratings, 6,040 users, 3,706 rated movies (from
3,883 total in the catalog), with genre metadata and user demographics
(age, gender, occupation). Chosen for being free, well-documented, and rich
enough to exercise every layer of the architecture — the assignment
explicitly allows any dataset choice.

## 7. Technologies Used

Python 3.12, pandas, numpy, scipy, PyTorch (CPU-only), faiss-cpu,
sentence-transformers, huggingface_hub, Streamlit. All free and
open-source; no paid services anywhere in the pipeline.

## 8. Assumptions Made

- Ratings ≥4 (out of 5) treated as implicit positive feedback; ratings
  below that are dropped rather than treated as explicit negatives
- A user's most recent 20% of ratings (by timestamp) held out as the test
  set, per-user, to simulate realistically predicting future behavior
  rather than interpolating within known history
- No session/temporal-recency weighting — all of a user's training
  history is treated equally regardless of how long ago it happened
- Cold-start users (no history) fall back to the Popularity model

## 9. Key Design Decisions

- **Vertical-slice-first development**: got baselines + a deployable UI
  working before building deep models, specifically to de-risk the
  assignment's mandatory deployment requirement rather than risk running
  out of time with nothing deployable
- **Common `BaseRecommender` interface**: every model, baseline or deep,
  implements the same `fit`/`score`/`recommend` contract, so evaluation
  and serving code never need to know which model they're calling
- **Item-CF sparsification**: switched from a dense item-item similarity
  matrix to each item's top-50 neighbors only, cutting the saved model
  from 115MB to 7.3MB (both for GitHub's 100MB file limit and Streamlit
  Cloud's RAM budget) — and it also *improved* Recall@10 and Coverage,
  since dropping near-zero similarities sharpened rather than diluted the
  signal
- **No L2-normalization in Two-Tower embeddings**: an initial version
  normalized both towers' outputs to unit vectors (pure cosine
  similarity, as many published two-tower designs do). Measured on this
  dataset, that choice cut Precision@10 by ~45% and Coverage by ~70% —
  likely because normalization discards magnitude information a smaller
  dataset needs. Removed after the drop was measured, not assumed.
- **Pre-trained artifacts committed to the repo, no retraining on
  startup**: Streamlit Cloud's free tier has limited build time and RAM;
  `train.py` runs offline and the app only ever loads saved `.pkl`/`.bin`
  files.

## 10. Evaluation Methodology

**Split**: time-based leave-last-out, per user (last 20% of each user's
ratings by timestamp → test set). Avoids future-data leakage that a
random row split would allow.

**Metrics**: Precision@10, Recall@10, NDCG@10 (rewards relevant items
ranked higher, not just present), and Coverage (fraction of the catalog
that ever gets recommended — a low-coverage model may score well while
only ever recommending the same popular handful).

**Results** (all 5 models, same test set, confirmed on the deployed
system):

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|---|---|---|---|---|
| Popularity | 0.0668 | 0.0427 | 0.0762 | 0.0232 |
| **Item-Item CF** | **0.0838** | **0.0770** | **0.1024** | 0.3030 |
| Matrix Factorization | 0.0772 | 0.0575 | 0.0897 | 0.2372 |
| Two-Tower | 0.0670 | 0.0420 | 0.0756 | 0.1082 |
| Neural CF | 0.0702 | 0.0440 | 0.0781 | 0.1082 |

## 11. Test Cases

### Successful scenarios
- **Semantic sanity check**: nearest neighbors of "Toy Story (1995)" by
  RAG's content embeddings are Toy Story 2, A Goofy Movie, Gumby: The
  Movie — all genuinely similar animated/family titles, confirming the
  embedding space captures real thematic similarity
- **Retrieval correctness**: FAISS's top-5 retrieved candidates matched
  brute-force exact scoring exactly, for every user tested
- **Full funnel produces coherent output**: for a user with a
  sci-fi/action history, the full funnel (Two-Tower→FAISS→NeuralCF)
  surfaced Star Wars Episode V, with a generated explanation correctly
  citing "intense dramas and epic space battles"
- **Explanation fallback**: when no `HF_TOKEN` is set (or the API call
  fails), the system falls back to a deterministic template rather than
  crashing or returning nothing
- **Live deployment verified**: app deployed to Streamlit Community
  Cloud, confirmed working end-to-end at the public URL

### Failure scenarios
- **Deep models underperform classical baselines**: Item-Item CF beats
  both Two-Tower and Neural CF on every ranking metric. This dataset
  (462K training interactions) is likely too small for deep models to
  out-learn a well-tuned classical method — a known, published effect,
  not a bug
- **Cold-start users get generic recommendations**: a user with zero
  rating history receives only Popularity-based fallback recommendations,
  with no personalization at all
- **Small candidate pools for power users**: users with very large
  rating histories can end up with a shrunken FAISS candidate set once
  already-seen items are excluded, occasionally falling below the
  requested top-K count
- **Deployment errors encountered and fixed**: `requirements.txt`
  initially omitted `streamlit`, `faiss-cpu`, and `huggingface_hub`
  (present locally via manual `pip install` but never added to the
  pinned file), causing a `ModuleNotFoundError` on first deploy —
  corrected before final submission

## 12. Known Limitations

- Deep models (Two-Tower, Neural CF) do not outperform Item-CF on this
  dataset — see Section 11
- No session-based or recency-weighted signal — a user's rating from
  years ago counts equally to a recent one
- No online/streaming retraining — the system is entirely batch-trained
- Single domain (movies) actually trained and deployed. The core
  architecture (`data_pipeline.py`, `models/*.py`) is written to be
  domain-agnostic and was verified against Books (Book-Crossing) and
  E-commerce (Online Retail) data during development, but those domains
  were not carried through to the final deployed system — this
  documentation describes what is actually live, not what was tested in
  isolation
- RAG explanations depend on a third-party free API tier (Hugging Face),
  which can rate-limit or occasionally reject specific models
  (encountered during development — an initial model choice was rejected
  by HF's router and swapped for one confirmed to work)

## 13. Future Improvements

1. Actually deploy the multi-domain version (Books, E-commerce) that was
   validated during development but not carried into the final submission
2. Session-based/sequential modeling (e.g. GRU4Rec-style) instead of
   treating all history equally
3. Incorporate user demographics and richer item content (real plot
   summaries via TMDb, not just genre tags) into the towers
4. Off-policy/counterfactual offline evaluation, to approximate what
   online A/B testing would show without live traffic
5. Swap FAISS's exact `IndexFlatIP` for an approximate index (IVF/HNSW)
   to demonstrate retrieval scaling past exact search
6. Scheduled/streaming retraining instead of one-off batch training
