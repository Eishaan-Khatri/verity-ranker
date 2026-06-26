# PRISM — Proof driven Ranking and Intelligent Selection Model

**A multi-layer, evidence-backed hiring pipeline that ranks 100,000 candidates against a job description — and proves its rankings aren't just keyword matching.**

Built for the **Redrob AI India Runs Data and AI Challenge**.

> [!NOTE]
> PRISM scores candidates the way a senior recruiter would — combining multi-layer retrieval (dense + sparse), LLM reasoning, skill graph analysis, and fraud-detection guards — not just keyword overlap.

---

## Table of Contents

- [Why PRISM](#why-prism)
- [Headline Result](#headline-result)
- [Tech Stack](#tech-stack)
- [Usage](#usage)
- [For Hackathon Sandbox Evaluation](#for-hackathon-sandbox-evaluation)
- [Evaluation & Ablation Study](#evaluation--ablation-study)
- [Robustness Testing](#robustness-testing)
- [Results Summary](#results-summary)
- [Team](#team)
- [License](#license)

---

## Why PRISM

Most hiring-ranker submissions for hackathons like this score candidates by keyword overlap with the JD — easy to game, and blind to actual domain fit. PRISM is built around a simple thesis: **a ranking pipeline is only trustworthy if you can show what happens when you strip each layer away.**

So instead of just shipping a ranked list, PRISM ships:
1. A ranking pipeline with 5 progressively richer scoring tiers
2. An **ablation study** that measures, layer by layer, what each tier actually changes about the top-100
3. **Guard mechanisms** that explicitly catch and demote keyword-stuffers, off-domain candidates, and undisclosed career transitions
4. **Robustness testing** — determinism checks, malformed-data injection, and a documented bug found-and-fixed during stress testing

---

## Headline Result

- Ranks **100,000** candidates against a single JD in **~5–8 seconds** (offline precompute excluded)
- **Zero overlap** with naive keyword-matching top-100; strong negative rank correlation (tau = -0.753)
- Guards demonstrably remove **all** off-domain and undisclosed-transition candidates from the top-100
- Fully deterministic, validated at 100K-candidate scale, stress-tested against malformed input

---

## Tech Stack

- **Retrieval:** FAISS (dense), BM25 (sparse), Reciprocal Rank Fusion (RRF)
- **Scoring:** Multi-layer rubric (skills, experience, role fit, honesty guards)
- **Validation:** Schema enforcement, claim verification, robustness testing
- **Output:** CSV submission + audit report (JSON)
- **Ranking Engine:** Python (deterministic, CPU-only, no network calls)

---

## Usage — Two-Step Pipeline

### Step 1: Precompute (Offline, LLM-based, ~20 minutes)

Run once per JD + candidate dataset to extract skills, experience, and compute features:

```powershell
python precompute.py --jd data/jd.txt --candidates candidates.jsonl --cache-dir cache
```

This generates:
- `cache/jd_profile.json` — structured JD features
- `cache/candidate_features.jsonl` — 100K+ precomputed candidate profiles
- `cache/hyde_profiles.json`, `cache/manifest.json` — supporting data

### Step 2: Rank (Offline, deterministic, ~6 seconds)

Load precomputed cache and rank candidates:

```powershell
python rank.py --jd data/jd.txt --candidates candidates.jsonl --cache-dir cache --output outputs/ranked_output.csv
```

Produces: `outputs/ranked_output.csv` with top 100 ranked candidates.

---

## For Hackathon Sandbox Evaluation

### Same Dataset (Provided 100K Candidates)

- Cache is pre-baked and committed to repo (Git LFS)
- Run **Step 2 only** — completes in ~6 seconds
- Fully deterministic, reproducible, no network calls
- **This is what sandbox evaluation will use**

### New/Unseen Dataset

If judges evaluate with a different candidate pool:

1. Run **Step 1** (precompute.py) on the new dataset first (~20 minutes, LLM calls)
2. Then run **Step 2** (rank.py) with the new cache (~6 seconds)
3. Document this two-step requirement clearly in submission

---

## Evaluation & Ablation Study

To isolate what each pipeline stage actually contributes, we reconstructed 5 progressively richer rankings from the *same* precompute cache — no new LLM calls — and measured each tier's top-100 composition (n = 100,000 candidates):

| Tier | Off-domain titles | Self-disclosed transitions | Avg. stuffer risk |
|---|---|---|---|
| Keyword-only | 0 | 0 | 0.050 |
| + Skill graph | 10 | 17 | 0.042 |
| + Agent rubric | 54 | 90 | 0.007 |
| + Retrieval blend | 55 | 91 | 0.007 |
| + Guards (full pipeline) | **0** | **0** | 0.024 |

**Reading this table:** keyword-only matching can't surface off-domain candidates at all — but it's the easiest to game (highest stuffer-risk). As richer signals are layered in, the ranker starts surfacing nominally well-matched but actually off-domain or recently-transitioned candidates. The final guard layer explicitly catches and removes these, collapsing both counts to zero while keeping stuffer-risk less than half the keyword-only baseline.

**This is direct, measurable evidence the guards are not cosmetic** — without them, roughly half of the top-100 would be wrong-fit candidates.

---

## Robustness Testing

Beyond accuracy, the pipeline was stress-tested for production-grade reliability:

**Determinism** — `rank.py` run twice against the identical JD + cache produced an **identical ranking order and identical top-100 set**, both times. Confirms the offline ranking stage is fully reproducible.

**Score sanity at scale** — all 100,000 cached `final_score` values checked for `NaN`/`None`/out-of-range values. **Zero anomalies.**

**Malformed-data injection** — four synthetic corrupted candidates (empty dimensions, empty skill matches, explicit `None` score, missing risk fields) were injected into the cache. The first run **crashed**:

```
TypeError: float() argument must be a string or a real number, not 'NoneType'
```

Root cause: a `.get(key, default)` pattern that only falls back when a key is *missing*, not when its value is explicitly `None`. Fixed with a null-safe coercion helper applied across all numeric reads in the ranking path. After the fix, the same corrupted cache (100,004 rows) ranked successfully in 6.1s, and **none of the 4 corrupted candidates appeared in the final top-100** — confirming the pipeline degrades gracefully on partial data corruption instead of crashing or silently promoting bad data.

---

## Results Summary

- Ranks **100,000** candidates against a single JD in **~5–8 seconds** (offline precompute excluded)
- **Zero overlap** with naive keyword-matching top-100; strong negative rank correlation (tau = -0.753)
- Guards demonstrably remove **all** off-domain and undisclosed-transition candidates from the top-100, where they'd otherwise make up over half the list
- Fully deterministic, validated at 100K-candidate scale, stress-tested against malformed input

---

## Team

- **Arushi Tripathi** ([@ArrushiTripathi2429](https://github.com/ArrushiTripathi2429)) — Architecture, guard layers, robustness testing, sandbox integration
- **Eishaan Khatri** ([@Eishaan-Khatri](https://github.com/Eishaan-Khatri)) — Multi-agent scoring, claim verification, evaluation

---

## License

MIT License

Copyright (c) 2026 Arushi Tripathi, Eishaan Khatri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.