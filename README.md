# PRISM — Proof driven Ranking and Intelligent Selection Model

**A multi-layer, evidence-backed hiring pipeline that ranks 100,000 candidates against a job description — and proves its rankings aren't just keyword matching.**

Built for the **Redrob AI India Runs Data and AI Challenge**.

> [!NOTE]
> <!-- TODO: one-line pitch, e.g. "PRISM scores candidates the way a senior recruiter would — combining retrieval, LLM reasoning, and fraud-detection guards, not just keyword overlap." -->

---

## Table of Contents

- [Why PRISM](#why-prism)
- [Headline Result](#headline-result)
- [Architecture](#architecture)
- [The Six-Dimension Rubric](#the-six-dimension-rubric)
- [Guard Layers (Anti-Gaming)](#guard-layers-anti-gaming)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
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

## Verity Ranker V1

V1 is the minimum working candidate ranking pipeline.

It is intentionally limited. It does not claim full agentic evaluation, proof-of-work verification, graph reasoning, or fairness guarantees. Its job is to prove that the project can:

- read a job description
- read candidate profiles/resumes
- extract basic role and candidate signals
- rank candidates with a deterministic scoring rubric
- produce a valid ranked output file
- generate a small audit report for review

## Why V1 Exists

Most advanced hiring-ranker ideas fail if the basic output file is wrong. V1 creates the clean baseline before adding claim verification, graph retrieval, multi-agent scoring, and reranking.

## Run Sample

From this folder:

```bash
python run_v1.py run --jd data/sample/jd.txt --candidates data/sample/candidates --output outputs/final/ranked_output.csv --report outputs/final/audit_report.json
python run_v1.py validate --file outputs/final/ranked_output.csv
```

## Web Demo

V1 includes a Streamlit app because GitHub Pages cannot run the Python ranker server-side. Streamlit Community Cloud can deploy this repository from GitHub and run the existing Python pipeline.

Run locally:

```bash
pip install -r requirements.txt

# TODO: list required environment variables (API keys etc.)
# e.g. GEMINI_API_KEY=...
# e.g. GROQ_API_KEY=...
```

---

## Usage

**1. Precompute (offline, run once per JD + candidate pool):**

```powershell
python precompute.py --jd data/hackathon/jd.txt --candidates candidates.jsonl --cache-dir cache
```

**2. Rank (online, fast, deterministic):**

```powershell
python rank.py --jd data/hackathon/jd.txt --candidates candidates.jsonl --cache-dir cache --output docs/submission.csv
```

Runtime: ~5–8 seconds for 100,000 candidates → top 100, on CPU, no network calls.

**3. Run the ablation/evaluation report:**

```powershell
python ablation_report.py --cache-dir cache --output docs/ablation_report.json
```

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

<!-- TODO: confirm names/roles/links -->
- **Arushi Tripathi** —
- **Eishaan Khatri** ([@Eishaan-Khatri](https://github.com/Eishaan-Khatri)) —

---

## License

<!-- TODO: add license, e.g. MIT -->
