# Product Comparison: Universal Recommender vs. YouTube's Recommendation System

Our system's architecture and UI are both directly inspired by YouTube's publicly
documented recommendation approach (Covington et al., 2016, and later updates).
This document compares the two honestly, including where ours falls short.

## Similarities

| Aspect | YouTube | Our System |
|---|---|---|
| Architecture | Two-stage funnel: candidate generation → ranking | Same: Two-Tower+FAISS retrieval → Neural CF ranking |
| Candidate generation | Extreme multiclass classification, ANN search at inference | Two-Tower embeddings, FAISS exact/approximate search |
| Ranking objective | Predicts expected watch time, not just click probability | Predicts implicit-feedback ranking (BPR loss); doesn't model watch time (no such signal in our data) |
| Cold start | Falls back to fresh/popular content | Falls back to Popularity model |
| UI | Dark theme, red accent, card-based feed | Same visual language |

## Differences

| Aspect | YouTube | Our System |
|---|---|---|
| Scale | Billions of videos, billions of users | 3,706 items, 6,040 users (MovieLens-1M) |
| Candidate generation | Negative sampling over a full softmax (too expensive to do exactly at their scale) | Exact FAISS search (IndexFlatIP) — feasible at our scale, would need approximate indexing (IVF/HNSW) at YouTube's |
| Ranking signal | Watch time, click-through, impression frequency, freshness | Star ratings converted to implicit positive/negative feedback only |
| Personalization signals | Watch history, search history, demographics, session context | Rating history only (demographics available in data but not yet used in features) |
| Explanations | Not natural-language; UI shows terse reasons ("Because you watched X") | Full LLM-generated natural-language explanations (RAG layer) — arguably a richer UX than YouTube's own |
| Online learning | Continuously retrains on streaming data | Batch-trained only; no online updates |
| A/B testing | Core to their deployment process | Not implemented — offline metrics only |

## Current Limitations

- **Dataset staleness**: MovieLens-1M interactions are from the late 1990s; no notion of "freshness" or trending content, unlike YouTube
- **No session context**: each recommendation is based on full rating history, not what the user is doing *right now* (YouTube heavily weights recent session activity)
- **Item-CF outperforms our deep models** on this dataset (see evaluation table) — the opposite of what you'd expect from a "modern" system, and worth stating plainly rather than hiding: deep models need substantially more data than MovieLens-1M provides to show their real advantage
- **No true online/streaming retraining** — everything is batch, unlike YouTube's continuous model updates
- **Single-domain demo** — while the architecture is domain-agnostic by design, we've only validated it on movies; e-commerce/music would need new item feature encoders to actually prove portability

## What We'd Build Next With More Time

1. **Session-based signals** — incorporate recency/sequence (e.g. a GRU4Rec-style sequential model) rather than treating all historical ratings equally
2. **Richer features** — bring in user demographics (age, occupation) and item content (actual plot text via TMDb, not just genres) into the towers
3. **Online evaluation** — simulate A/B testing via counterfactual/off-policy evaluation methods, since we can't run live A/B tests on a static dataset
4. **A second domain** — actually port the pipeline to a second dataset (e.g. an e-commerce click dataset) to empirically prove the "universal" claim, not just argue it architecturally
5. **Approximate FAISS indexing** — swap `IndexFlatIP` for `IndexIVFFlat` or HNSW to demonstrate the retrieval stage scales past exact search
6. **Streaming retraining** — even a simple scheduled retrain job would move this closer to how production systems actually operate
