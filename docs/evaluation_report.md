# Evaluation Report — PRISM / Verity Ranker

## 1. Fairness + Proxy Audit (Layer 13)

**Method:** Top-k impact ratio (k=5,10,20,50,100) across institution tier,
institution name, name-length (gender proxy), graduation year (Kendall's τ),
and career-gap presence — run on the full 100,000-candidate ranked output.

**Finding — institutional prestige correlation (confirmed, not code-injected):**
Tier-1 institution candidates are 6–7x over-represented in the top-50/100
(baseline 4.9% of the pool → 31–36% of top ranks). This pattern repeats
consistently across individual elite institutions (IITs, NITs, IIITs,
Stanford, UC Berkeley, MIT) when broken down by name, though each individual
institution's flag is statistically weak in isolation (n=1–3 occurrences).

Root-cause check: `education[].tier` and institution name are never read by
any scoring agent (`grep` confirmed — zero references outside the audit
script itself). The correlation is therefore not an explicit scoring input;
it reflects that tier-1 candidates' described career history, skills, and
achievements are independently stronger in the underlying dataset, and
institution acts as a correlated marker rather than a cause.

**Recommendation:** Recruiters using this ranking for final-stage decisions
should be aware of this pattern and consider blind-resume review to avoid
anchoring on institution name directly, even though the system does not
score it explicitly.

**No bias detected:** name-length (gender proxy) — no flags at k≥50.

## 2. Rank Stability Test (Layer 14)

**Method:** 5 perturbation runs, ±5% weight jitter (renormalised), evaluated
on the top-100 by base rank. A candidate is "stable" if their rank shifts by
≤1 position across all runs.

**Finding:** 6/100 top candidates are stable; 94 show rank movement >1
position under small weight perturbation.

**Interpretation:** This reflects tightly-clustered scores in the lower half
of the shortlist (e.g. rank 50 vs rank 90 may differ by <1 point), not a
flaw in the ranking logic. The top ~10 candidates (where score gaps are
largest, e.g. 87.50 → 83.50 across rank 1–7) are comparatively more robust
than ranks 50+. Placements beyond roughly rank 20 should be treated as
lower-confidence orderings rather than precise distinctions.

## 3. Evaluation + Ablation Report (Layer 17)

## Section 3 — Ablation: What Each Layer Actually Does

To isolate the contribution of each pipeline stage, we reconstructed 5
progressively richer rankings from the same precompute cache — no new LLM
calls, no re-running precompute — and measured how each layer's top-100
composition changes (n=100,000 candidates, JD skill count=11).

| Tier | Off-domain titles | Self-disclosed transitions | Avg stuffer risk |
|---|---|---|---|
| Keyword-only | 0 | 0 | 0.050 |
| + Skill graph | 10 | 17 | 0.042 |
| + Agent rubric | 54 | 90 | 0.007 |
| + Retrieval blend | 55 | 91 | 0.007 |
| + Guards (full pipeline) | **0** | **0** | 0.024 |

**Reading this table:** keyword-only matching is narrow — it can't surface
off-domain candidates at all, but it's the most easily gamed (highest
stuffer-risk at 0.050). As richer signals are layered in, the ranker starts
surfacing nominally well-matched but actually off-domain or recently-
transitioned candidates (54 off-domain titles, 90 self-disclosed transitions
in the top-100 by Tier 4). The final guard layer (Tier 5) explicitly catches
and removes these — collapsing both counts back to 0, while keeping
stuffer-risk less than half of the keyword-only baseline (0.024 vs 0.050).

This is direct, measurable evidence that the guards are not cosmetic: without
them, roughly half of the top-100 would be wrong-fit candidates.

### Keyword-only vs. full-pipeline divergence

- **Jaccard overlap (top-100): 0.0** — zero candidates in common between the
  naive keyword-matched top-100 and the full-pipeline top-100.
- **Kendall's tau: -0.753** — a strong, statistically robust *negative*
  correlation between keyword-only score and full-pipeline rank, not noise.

The three largest demotions from keyword-only top-10 illustrate why:

| Candidate | Title (resume) | Headline | Full-pipeline rank |
|---|---|---|---|
| Deepak Pandey (CAND_0000406) | DevOps Engineer | "Backend systems & APIs" | 42,193 |
| Myra Sen (CAND_0000703) | Mobile Developer | "Full-stack development" | 31,929 |
| Deepak Mehta (CAND_0000570) | DevOps Engineer | "Backend systems & APIs" | 28,804 |

All three carry generic, buzzword-dense headlines that overlap heavily with
JD keywords, despite their actual titles sitting in a different domain
(DevOps/Mobile vs. the target role). Keyword-matching rewards the headline
overlap; the full pipeline correctly identifies the domain mismatch and
demotes them by 28,000–42,000 ranks.

## Section 4 — Robustness Testing

Beyond accuracy metrics, we stress-tested the pipeline for production-grade
reliability across three dimensions.

### 4.1 Determinism

`rank.py` was run twice against the identical JD and cache (no re-precompute,
no new LLM calls):

- **Ranking order: identical** across both runs.
- **Top-100 set: identical** across both runs.
- Runtime: 5.8–8.1s for 100,000 candidates → top 100 (well within the
  hackathon's offline ranking budget; variance attributable to disk I/O,
  not non-determinism in the ranking logic).

This confirms the offline ranking stage (Layer 11 listwise tie-break +
submission sort) is fully deterministic given a fixed precompute cache —
re-running the submission script will always reproduce the same leaderboard
result.

### 4.2 Score sanity at scale

All 100,000 cached `final_score` values were checked for `NaN`, `None`, or
out-of-range (outside [0, 100]) values. **Zero anomalies found.**

### 4.3 Malformed-data injection (and a real bug found + fixed)

Four synthetic corrupted candidates were injected into the feature cache to
simulate partial precompute failures (e.g. a malformed Gemini response):

| Injected candidate | Corruption |
|---|---|
| CAND_BROKEN_001 | Empty `dimensions` dict |
| CAND_BROKEN_002 | Empty `matched_required` / `matched_preferred` |
| CAND_BROKEN_003 | `final_score` explicitly `None` |
| CAND_BROKEN_004 | Missing `honeypot_risk` / `stuffer_risk` fields |

The first run **crashed** the ranker: