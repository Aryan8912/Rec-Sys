# Universal Recommender

A recommendation engine built on a domain-agnostic architecture, demoed
and deployed on MovieLens-1M (1M movie ratings, 6,040 users, 3,706 movies).

**Live app**: https://rec-sys-gfrqwtuewmgqeuapwosliv.streamlit.app/
**Repo**: https://github.com/Aryan8912/Rec-Sys

## What it does

Enter a User ID, pick a recommendation strategy, get a ranked list of
movie recommendations — optionally with an AI-generated explanation for
the top pick.

Two ways to get recommendations:
- **Single model**: choose directly from Popularity, Item-Item CF, Matrix
  Factorization, Two-Tower, or Neural CF
- **Full funnel**: Two-Tower generates a user embedding → FAISS retrieves
  ~200 candidates → Neural CF re-ranks them to the final list (the same
  retrieval-then-ranking pattern used in production systems like YouTube)

## Setup (local)

```bash
python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Train all models and save artifacts:
```bash
python src/train.py
```

Run the app:
```bash
streamlit run app/streamlit_app.py
```

## Project structure

```
├── data/                  MovieLens-1M CSVs (ratings, movies, users)
├── src/
│   ├── data_pipeline.py   loads and splits data
│   ├── models/            Popularity, Item-CF, MF, Two-Tower, Neural CF
│   ├── retrieval/         FAISS candidate retrieval
│   ├── rag/                semantic embeddings + LLM explanation generator
│   ├── evaluation/        Precision@K, Recall@K, NDCG@K, Coverage
│   └── train.py           trains everything, saves to artifacts/
├── app/
│   └── streamlit_app.py   the deployed UI
├── artifacts/             trained models (committed — app loads these,
│                            does not retrain on startup)
├── docs/
│   ├── DOCUMENTATION.md   full write-up: architecture, methodology,
│   │                        evaluation, test cases, limitations
│   └── COMPARISON.md      comparison against YouTube's public architecture
└── requirements.txt
```

## Results

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|---|---|---|---|---|
| Popularity | 0.0668 | 0.0427 | 0.0762 | 0.0232 |
| **Item-Item CF** | **0.0838** | **0.0770** | **0.1024** | 0.3030 |
| Matrix Factorization | 0.0772 | 0.0575 | 0.0897 | 0.2372 |
| Two-Tower | 0.0670 | 0.0420 | 0.0756 | 0.1082 |
| Neural CF | 0.0702 | 0.0440 | 0.0781 | 0.1082 |

Full methodology, architecture diagrams, and honest discussion of why the
deep models don't outperform Item-CF on this dataset are in
`DOCUMENTATION.md`.
