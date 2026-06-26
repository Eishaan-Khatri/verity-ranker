# Copy these sections into your README.md (VS Code)

---

## Usage — Two-Step Pipeline

**Step 1: Precompute (Offline, LLM-based, ~20 minutes)**

Run once per JD + candidate dataset to extract skills, experience, and compute features:

```powershell
python precompute.py --jd data/jd.txt --candidates candidates.jsonl --cache-dir cache
```

This generates:
- `cache/jd_profile.json` — structured JD features
- `cache/candidate_features.jsonl` — 100K+ precomputed candidate profiles
- `cache/hyde_profiles.json`, `cache/manifest.json` — supporting data

**Step 2: Rank (Offline, deterministic, ~6 seconds)**

Load precomputed cache and rank candidates:

```powershell
python rank.py --jd data/jd.txt --candidates candidates.jsonl --cache-dir cache --output outputs/ranked_output.csv
```

Produces: `outputs/ranked_output.csv` with top 100 ranked candidates.

---

## For Hackathon Sandbox Evaluation

**If testing with the provided 100K candidates:**
- Cache is pre-baked and committed to repo (Git LFS)
- Run Step 2 only — completes in ~6 seconds
- Fully deterministic, reproducible, no network calls

**If testing with a NEW candidate dataset:**
1. Run Step 1 (precompute.py) on the new dataset first
2. Then run Step 2 (rank.py) with the new cache

---

## Robustness & Validation

 **Determinism:** Identical rankings across consecutive runs (100K candidates)  
 **Cache integrity:** Zero NaN/null anomalies in 100K feature vectors  
 **Graceful degradation:** Malformed data injection test — pipeline handles corrupted rows without crashing  
 **Guard effectiveness:** Off-domain and keyword-stuffed candidates explicitly demoted by verification guards  

---

## Tech Stack

- **Retrieval:** FAISS (dense), BM25 (sparse), Reciprocal Rank Fusion
- **Scoring:** Multi-layer rubric (skills, experience, role fit, honesty guards)
- **Validation:** Schema enforcement, claim verification, robustness testing
- **Output:** CSV submission + audit report (JSON)
- **Deployment:** FastAPI (backend), Streamlit (dashboard), HuggingFace Spaces

---

## Team

- **Eishaan Khatri** — Multi-agent scoring, claim verification, evaluation
- **Arushi Tripathi** — Architecture, guard layers, robustness testing
- **Naman Shrestha** - PPT and Presentations

---

## License

MIT