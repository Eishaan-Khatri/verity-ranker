# Hackathon Submission — Verity Ranker

## Two-script design (required for sandbox)

| Script | When to run | Internet | Purpose |
|--------|-------------|----------|---------|
| `precompute.py` | Once, before submission | Optional (LLM for JD only) | Full V2 intelligence → disk cache |
| `rank.py` | Judge sandbox / re-run | **NO** | Load cache → top 100 CSV in <5 min |

## One-command reproduction

```bash
pip install -r requirements.txt
python scripts/generate_hackathon_data.py   # local test data only
python precompute.py --jd data/hackathon/jd.txt --candidates data/hackathon/candidates.jsonl --cache-dir cache --force-fallback
python rank.py --jd data/hackathon/jd.txt --candidates data/hackathon/candidates.jsonl --cache-dir cache --output submission/ranked_output.csv
python scripts/validate_submission.py submission/ranked_output.csv
```

Replace `data/hackathon/candidates.jsonl` with the official 100K file for the real submission.

## What runs where

### `precompute.py` (heavy, once)

| Layer | Component |
|-------|-----------|
| 2 | JD Intelligence (`analyse_jd`) |
| 3 | HyDE ideal profiles → `cache/hyde_profiles.json` |
| 4 | Candidate profile extraction |
| 5 | **Skipped** — no GitHub crawl; uses `github_activity_score` |
| 6 | Evidence ledger from resume + platform score |
| 7 | Skill graph (via agent + retrieval helpers) |
| 9 | Multi-agent evaluation (rule-based, offline) |
| 10 | Rubric scoring |
| Guards | Honeypot, keyword-stuffer, engagement multipliers |

Writes: `cache/jd_profile.json`, `cache/candidate_features.jsonl`, `cache/manifest.json`

### `rank.py` (sandbox, offline)

| Step | Component |
|------|-----------|
| Load | Precomputed cache only — **no API calls** |
| 11 | Listwise re-rank top pool (rule-based tie-break) |
| 14 | CSV output + auto-validation |

## Output format

```csv
candidate_id,rank,score,reasoning
C00142,1,87.40,"Strong fit for Machine Learning Engineer: covers required skills Python, FastAPI ..."
```

Exactly **100 rows**. Scores **non-increasing** by rank. Ties broken by `candidate_id` ascending (validator requirement).

## Guards built into scoring

- **Honeypot down-rank** — impossible claims, title/experience conflicts, skill stuffing
- **Keyword-stuffer down-rank** — buzzword density vs profile depth
- **Engagement down-rank** — `last_active_date`, `recruiter_response_rate`

## Sandbox demo

Streamlit app (`streamlit run app.py`) demonstrates the full V2 pipeline on sample data for the portal link requirement.

## Offline self-test (before submitting)

```bash
# After precompute on the full 100K file:
python rank.py --jd data/jd.txt --candidates data/candidates.jsonl --cache-dir cache --output submission/ranked_output.csv
# Must finish in under 5 minutes with no network.
```
