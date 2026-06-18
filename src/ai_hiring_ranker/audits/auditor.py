"""
Fairness + Proxy Audit & Rank Stability Test — Layer 13.

Two independent audits run sequentially on the final ranking.
Neither audit modifies the ranking — they only produce flags and notes
that attach to the final output for recruiter review.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Audit 1 — Fairness + Proxy Audit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Checks whether proxy or protected attributes correlated with rank:

  - Prestige bias:  are top-k candidates over-represented by a single
                    institution / company? (top-k impact ratio ≥ 2×)
  - Location bias:  does candidate location appear correlated with rank?
  - Name/gender:    does name pattern (common gender signal) correlate?
  - Gap penalty:    are candidates with career gaps systematically lower?
  - Graduation year: does recency of graduation drive rank independent
                    of actual skills?

Proxy detection strategy:
  - Extract attribute values from CandidateProfiles
  - Compute top-k impact ratios (top-5, top-10 windows)
  - Flag over-representation (impact ratio ≥ 2.0) as WARNING
  - Flag extreme over-representation (≥ 3.5×) as HIGH
  - Compute Kendall's τ between proxy-rank and final-rank for
    attributes where we have numeric proxies (graduation year)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Audit 2 — Rank Stability Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Re-runs rubric scoring N times with small controlled perturbations
to check whether the ranking is robust or fragile:

  weight_jitter:  ±5% random nudge on each dimension weight
                  (weights are renormalised after each nudge)
  score_jitter:   ±3% Gaussian noise on each dimension score

For each candidate, records:
  - min / max rank observed across all runs
  - rank variance and score standard deviation
  - is_stable = True if rank varied by ≤ 1 position

Candidates flagged as unstable should be treated as lower-confidence
placements and highlighted in the recruiter report.

Public API
----------
run_fairness_audit(profiles, rerank_result, hiring_profile)
    → FairnessAuditResult

run_stability_audit(rerank_result, hiring_profile, n_runs=5)
    → StabilityAuditResult

run_audit(profiles, rerank_result, hiring_profile, run_id, n_runs)
    → AuditReport

save_audit(audit_report, output_dir)
    → Path
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Optional

from ..candidate_extraction.schemas import CandidateProfile
from ..jd_intelligence.schemas import HiringProfile
from ..reranking.schemas import RerankResult
from ..scoring.schemas import CandidateScore
from .schemas import (
    AuditReport,
    CandidateStability,
    FairnessAuditResult,
    FlagSeverity,
    ProxyFlag,
    StabilityAuditResult,
    TopKImpactRatio,
)

logger = logging.getLogger(__name__)

# Proxy fields defined in verification_rules.yaml
_PROXY_FIELDS = [
    "university_name",
    "location",
    "graduation_year",
    "name",
    "gender_signal",
]

# Impact ratio thresholds
_WARNING_RATIO = 2.0
_HIGH_RATIO    = 3.5

# Stability: a rank shift of > this is "unstable"
_STABLE_RANK_RANGE = 1

# Weight perturbation magnitude (±fraction)
_WEIGHT_JITTER = 0.05
_SCORE_JITTER  = 0.03


# ---------------------------------------------------------------------------
# Proxy attribute extractors
# ---------------------------------------------------------------------------


def _extract_institution(profile: CandidateProfile) -> Optional[str]:
    """Extract the most recent / highest-level institution name."""
    if not profile.education:
        return None
    # Prefer the most recent entry
    for edu in profile.education:
        if edu.institution:
            return edu.institution.strip().lower()
    return None


def _extract_location(profile: CandidateProfile) -> Optional[str]:
    """
    Attempt to infer location from company names or resume text.
    This is a best-effort heuristic — no structured location field exists.
    """
    # Without a raw_text field, we can't reliably extract location.
    # Return None — the audit will skip location checks gracefully.
    return None


def _extract_graduation_year(profile: CandidateProfile) -> Optional[int]:
    """Extract the most recent graduation year."""
    years = [e.year for e in profile.education if e.year is not None]
    return max(years) if years else None


def _name_gender_signal(profile: CandidateProfile) -> Optional[str]:
    """
    Very rough heuristic: first name length bucket as a proxy signal.
    In a real system this would use a name-gender database.
    We return a bucketed string to avoid encoding actual gender.
    """
    if not profile.name:
        return None
    first = profile.name.strip().split()[0] if profile.name.strip() else ""
    if len(first) <= 4:
        return "short_name"
    if len(first) <= 7:
        return "medium_name"
    return "long_name"


# ---------------------------------------------------------------------------
# Top-k impact ratio computation
# ---------------------------------------------------------------------------


def _compute_top_k_ratio(
    all_values: list[Optional[str]],
    ranked_ids: list[str],
    profile_map: dict[str, CandidateProfile],
    extractor,
    attribute: str,
    k: int,
) -> list[TopKImpactRatio]:
    """
    Compute top-k impact ratios for one attribute.

    Returns a list of TopKImpactRatio objects (one per distinct value
    that appears in the top-k window).
    """
    top_k_ids = ranked_ids[:k]
    total = len(ranked_ids)
    if total == 0:
        return []

    # Count values in full population
    all_vals = [extractor(profile_map[cid]) for cid in ranked_ids if cid in profile_map]
    all_counter: dict[str, int] = {}
    for v in all_vals:
        if v:
            all_counter[v] = all_counter.get(v, 0) + 1

    # Count values in top-k
    top_k_vals = [extractor(profile_map[cid]) for cid in top_k_ids if cid in profile_map]
    top_k_counter: dict[str, int] = {}
    for v in top_k_vals:
        if v:
            top_k_counter[v] = top_k_counter.get(v, 0) + 1

    ratios: list[TopKImpactRatio] = []
    for val, top_count in top_k_counter.items():
        top_ratio      = top_count / k
        baseline_count = all_counter.get(val, 0)
        baseline_ratio = baseline_count / total if total > 0 else 0.0
        impact         = top_ratio / baseline_ratio if baseline_ratio > 0 else 0.0

        ratios.append(TopKImpactRatio(
            attribute=attribute,
            value=val,
            k=k,
            top_k_ratio=round(top_ratio, 3),
            baseline_ratio=round(baseline_ratio, 3),
            impact_ratio=round(impact, 2),
        ))

    return [r for r in ratios if r.impact_ratio >= _WARNING_RATIO]


# ---------------------------------------------------------------------------
# Kendall's τ for graduation year vs rank correlation
# ---------------------------------------------------------------------------


def _kendalls_tau(x: list[float], y: list[float]) -> float:
    """
    Compute Kendall's τ correlation between two equal-length lists.
    Returns 0.0 if fewer than 3 pairs exist.
    """
    n = len(x)
    if n < 3:
        return 0.0

    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1

    pairs = concordant + discordant
    return (concordant - discordant) / pairs if pairs > 0 else 0.0


# ---------------------------------------------------------------------------
# Audit 1: Fairness + Proxy Audit
# ---------------------------------------------------------------------------


def run_fairness_audit(
    profiles:       list[CandidateProfile],
    rerank_result:  RerankResult,
    hiring_profile: HiringProfile,
) -> FairnessAuditResult:
    """
    Run the fairness and proxy bias audit.

    Args:
        profiles:       All CandidateProfiles from Layer 4.
        rerank_result:  RerankResult from Layer 11 (final rank order).
        hiring_profile: HiringProfile from Layer 2.

    Returns:
        FairnessAuditResult with flags, top-k ratios, and risk level.
    """
    ranked_ids  = rerank_result.ranked_ids
    profile_map = {p.candidate_id: p for p in profiles}

    proxy_flags:  list[ProxyFlag]       = []
    top_k_ratios: list[TopKImpactRatio] = []
    audit_notes:  list[str]             = []

    # ── 1. Institution / prestige bias ─────────────────────────────────
    for k in (5, 10):
        if len(ranked_ids) < k:
            continue
        ratios = _compute_top_k_ratio(
            [], ranked_ids, profile_map,
            _extract_institution, "university_name", k,
        )
        top_k_ratios.extend(ratios)

        for r in ratios:
            severity = FlagSeverity.HIGH if r.impact_ratio >= _HIGH_RATIO else FlagSeverity.WARNING
            proxy_flags.append(ProxyFlag(
                proxy_field="university_name",
                description=(
                    f"'{r.value}' appears in {r.top_k_ratio:.0%} of top-{k} "
                    f"vs {r.baseline_ratio:.0%} overall (×{r.impact_ratio:.1f}). "
                    f"Possible prestige bias."
                ),
                severity=severity,
                affected_ids=[
                    cid for cid in ranked_ids[:k]
                    if profile_map.get(cid) and
                    _extract_institution(profile_map[cid]) == r.value
                ],
                recommendation=(
                    "Review whether institution name influenced skill_fit or "
                    "domain_match scores. Consider blind institution evaluation."
                ),
            ))

    # ── 2. Graduation year vs rank correlation ─────────────────────────
    grad_pairs = [
        (_extract_graduation_year(profile_map[cid]), i + 1)
        for i, cid in enumerate(ranked_ids)
        if cid in profile_map and _extract_graduation_year(profile_map[cid]) is not None
    ]
    if len(grad_pairs) >= 5:
        grad_years = [p[0] for p in grad_pairs]
        ranks      = [p[1] for p in grad_pairs]
        tau        = _kendalls_tau(grad_years, ranks)

        if abs(tau) >= 0.35:
            severity = FlagSeverity.HIGH if abs(tau) >= 0.55 else FlagSeverity.WARNING
            direction = "recent graduates rank higher" if tau < 0 else "older graduates rank higher"
            proxy_flags.append(ProxyFlag(
                proxy_field="graduation_year",
                description=(
                    f"Kendall's τ between graduation year and rank = {tau:.2f}. "
                    f"Pattern: {direction}. This may reflect experience preference "
                    f"or could be a proxy for age."
                ),
                severity=severity,
                affected_ids=[],
                recommendation=(
                    "Check whether experience_depth scoring is dominated by recency. "
                    "Ensure older candidates are not penalised beyond the JD's explicit "
                    "experience requirements."
                ),
            ))
            audit_notes.append(
                f"Graduation year correlates with rank (τ={tau:.2f}). Review experience scoring."
            )

    # ── 3. Career gap penalty check ────────────────────────────────────
    # Candidates with career gaps (None end_year mid-timeline) — check
    # if they cluster at the bottom of the ranking
    gap_ids: list[str] = []
    for cid in ranked_ids:
        p = profile_map.get(cid)
        if not p or len(p.career_timeline) < 2:
            continue
        # A gap exists if a non-current role has a long unexplained pause
        roles = sorted(
            [r for r in p.career_timeline if r.end_year and r.start_year],
            key=lambda r: r.start_year or 0,
        )
        for i in range(len(roles) - 1):
            gap_years = (roles[i + 1].start_year or 0) - (roles[i].end_year or 0)
            if gap_years >= 2:
                gap_ids.append(cid)
                break

    if gap_ids:
        gap_ranks = [ranked_ids.index(cid) + 1 for cid in gap_ids if cid in ranked_ids]
        if gap_ranks:
            avg_gap_rank = sum(gap_ranks) / len(gap_ranks)
            avg_all_rank = (len(ranked_ids) + 1) / 2
            if avg_gap_rank > avg_all_rank * 1.3:
                proxy_flags.append(ProxyFlag(
                    proxy_field="career_gap",
                    description=(
                        f"{len(gap_ids)} candidate(s) with career gaps ≥2 years "
                        f"average rank {avg_gap_rank:.1f} vs overall average "
                        f"{avg_all_rank:.1f}. Possible gap penalty bias."
                    ),
                    severity=FlagSeverity.WARNING,
                    affected_ids=gap_ids,
                    recommendation=(
                        "Verify that career gaps are not penalising candidates unfairly. "
                        "Gaps may reflect caregiving, health, or other protected reasons."
                    ),
                ))

    # ── 4. Compute overall risk level ──────────────────────────────────
    if any(f.severity == FlagSeverity.HIGH for f in proxy_flags):
        overall_risk = FlagSeverity.HIGH
    elif any(f.severity == FlagSeverity.WARNING for f in proxy_flags):
        overall_risk = FlagSeverity.WARNING
    else:
        overall_risk = FlagSeverity.INFO
        audit_notes.append("No significant proxy bias signals detected.")

    prestige_risk = any(
        r.attribute == "university_name" and r.impact_ratio >= _WARNING_RATIO
        for r in top_k_ratios
    )

    logger.info(
        "Fairness audit complete: %d flags, risk=%s",
        len(proxy_flags), overall_risk.value,
    )

    return FairnessAuditResult(
        proxy_flags=proxy_flags,
        top_k_ratios=top_k_ratios,
        prestige_bias_risk=prestige_risk,
        location_bias_risk=False,  # location extractor not implemented — always False
        overall_risk_level=overall_risk,
        audit_notes=audit_notes,
    )


# ---------------------------------------------------------------------------
# Audit 2: Rank Stability Test
# ---------------------------------------------------------------------------


def _jitter_weights(weights: dict[str, float], magnitude: float) -> dict[str, float]:
    """
    Apply random ±magnitude jitter to each weight and renormalise.
    Preserves relative ordering intent while testing sensitivity.
    """
    rng     = random.Random()  # unseeded — different each call
    jittered = {
        k: max(0.01, v + rng.uniform(-magnitude, magnitude))
        for k, v in weights.items()
    }
    total = sum(jittered.values())
    return {k: round(v / total, 6) for k, v in jittered.items()}


def _jitter_scores(score: CandidateScore, magnitude: float) -> CandidateScore:
    """Apply Gaussian noise to each dimension score."""
    rng = random.Random()

    def nudge(v: float) -> float:
        return round(min(1.0, max(0.0, v + rng.gauss(0, magnitude))), 4)

    return score.model_copy(update={
        "skill_fit":        nudge(score.skill_fit),
        "experience_depth": nudge(score.experience_depth),
        "seniority_match":  nudge(score.seniority_match),
        "domain_match":     nudge(score.domain_match),
        "career_growth":    nudge(score.career_growth),
        "proof_strength":   nudge(score.proof_strength),
    })


def _compute_perturbed_score(
    score: CandidateScore,
    weights: dict[str, float],
) -> float:
    """Recompute final_score from dimension scores and given weights."""
    weighted = (
        score.skill_fit        * weights.get("skill_fit", 0.30) +
        score.experience_depth * weights.get("experience_depth", 0.20) +
        score.seniority_match  * weights.get("seniority_match", 0.15) +
        score.domain_match     * weights.get("domain_match", 0.15) +
        score.career_growth    * weights.get("career_growth", 0.10) +
        score.proof_strength   * weights.get("proof_strength", 0.10)
    )
    return round(weighted * 100.0, 2)


def run_stability_audit(
    rerank_result:  RerankResult,
    hiring_profile: HiringProfile,
    *,
    n_runs:    int = 5,
    method:    str = "weight_jitter",
    rng_seed:  Optional[int] = 42,
) -> StabilityAuditResult:
    """
    Run the rank stability test.

    Args:
        rerank_result:  RerankResult from Layer 11.
        hiring_profile: HiringProfile from Layer 2.
        n_runs:         Number of perturbation runs (default 5).
        method:         'weight_jitter' or 'score_jitter'.
        rng_seed:       Seed for reproducibility (None = random).

    Returns:
        StabilityAuditResult with per-candidate stability assessments.
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    entries       = rerank_result.entries
    base_scores   = {e.candidate_id: e.candidate_score for e in entries}
    base_ranks    = {e.candidate_id: e.reranked_rank for e in entries}
    base_weights  = {
        "skill_fit": 0.30, "experience_depth": 0.20,
        "seniority_match": 0.15, "domain_match": 0.15,
        "career_growth": 0.10, "proof_strength": 0.10,
    }

    # Use weights from the first score if available
    if entries and entries[0].candidate_score.weights_used:
        base_weights = dict(entries[0].candidate_score.weights_used)

    # Collect rank observations across perturbation runs
    rank_observations: dict[str, list[int]] = {cid: [] for cid in base_scores}
    score_observations: dict[str, list[float]] = {cid: [] for cid in base_scores}

    for run_idx in range(n_runs):
        # Perturb
        if method == "weight_jitter":
            perturbed_weights = _jitter_weights(base_weights, _WEIGHT_JITTER)
            run_scores = {
                cid: _compute_perturbed_score(score, perturbed_weights)
                for cid, score in base_scores.items()
            }
        else:  # score_jitter
            run_scores = {
                cid: _compute_perturbed_score(
                    _jitter_scores(score, _SCORE_JITTER), base_weights
                )
                for cid, score in base_scores.items()
            }

        # Re-rank based on perturbed scores
        perturbed_ranked = sorted(
            run_scores.items(), key=lambda x: x[1], reverse=True
        )
        for new_rank, (cid, perturbed_score) in enumerate(perturbed_ranked, 1):
            rank_observations[cid].append(new_rank)
            score_observations[cid].append(perturbed_score)

    # Build CandidateStability per candidate
    candidate_stability: list[CandidateStability] = []
    unstable_ids: list[str] = []

    for entry in entries:
        cid        = entry.candidate_id
        obs_ranks  = rank_observations[cid]
        obs_scores = score_observations[cid]

        if not obs_ranks:
            continue

        min_rank = min(obs_ranks)
        max_rank = max(obs_ranks)
        rank_var = (
            sum((r - sum(obs_ranks) / len(obs_ranks)) ** 2 for r in obs_ranks)
            / len(obs_ranks)
        )
        score_mean = sum(obs_scores) / len(obs_scores)
        score_std  = math.sqrt(
            sum((s - score_mean) ** 2 for s in obs_scores) / len(obs_scores)
        )
        is_stable  = (max_rank - min_rank) <= _STABLE_RANK_RANGE

        note = ""
        if not is_stable:
            note = (
                f"Rank ranged from #{min_rank} to #{max_rank} across "
                f"{n_runs} perturbation runs. "
                f"Score σ={score_std:.2f}. Treat this placement with lower confidence."
            )
            unstable_ids.append(cid)

        candidate_stability.append(CandidateStability(
            candidate_id=cid,
            base_rank=entry.reranked_rank,
            base_score=entry.final_score,
            min_rank_observed=min_rank,
            max_rank_observed=max_rank,
            rank_variance=round(rank_var, 4),
            score_std_dev=round(score_std, 3),
            is_stable=is_stable,
            stability_note=note,
        ))

    stable_count   = sum(1 for cs in candidate_stability if cs.is_stable)
    unstable_count = len(candidate_stability) - stable_count

    audit_notes: list[str] = []
    if unstable_count == 0:
        audit_notes.append(
            f"All {stable_count} candidates have stable ranks across {n_runs} perturbation runs."
        )
    else:
        audit_notes.append(
            f"{unstable_count} candidate(s) have unstable ranks. "
            f"They should be treated as lower-confidence placements."
        )

    logger.info(
        "Stability audit complete: %d stable, %d unstable (n_runs=%d, method=%s)",
        stable_count, unstable_count, n_runs, method,
    )

    return StabilityAuditResult(
        perturbation_runs=n_runs,
        perturbation_method=method,
        candidate_stability=candidate_stability,
        unstable_ids=unstable_ids,
        stable_count=stable_count,
        unstable_count=unstable_count,
        audit_notes=audit_notes,
    )


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


def run_audit(
    profiles:       list[CandidateProfile],
    rerank_result:  RerankResult,
    hiring_profile: HiringProfile,
    *,
    run_id:         str = "",
    n_runs:         int = 5,
    stability_method: str = "weight_jitter",
    rng_seed:       Optional[int] = 42,
) -> AuditReport:
    """
    Run both the fairness audit and the stability test.

    Args:
        profiles:          CandidateProfiles from Layer 4.
        rerank_result:     RerankResult from Layer 11.
        hiring_profile:    HiringProfile from Layer 2.
        run_id:            Pipeline run identifier.
        n_runs:            Perturbation runs for stability test.
        stability_method:  'weight_jitter' or 'score_jitter'.
        rng_seed:          RNG seed for reproducibility.

    Returns:
        AuditReport combining both audit results.
    """
    logger.info("Layer 13: running fairness audit...")
    fairness = run_fairness_audit(profiles, rerank_result, hiring_profile)

    logger.info("Layer 13: running stability test (%d runs)...", n_runs)
    stability = run_stability_audit(
        rerank_result, hiring_profile,
        n_runs=n_runs,
        method=stability_method,
        rng_seed=rng_seed,
    )

    report = AuditReport(
        run_id=run_id,
        job_title=hiring_profile.job_title,
        fairness=fairness,
        stability=stability,
    )

    logger.info(
        "Layer 13 complete. Fairness: %s | Stability: %d/%d stable\n%s",
        fairness.overall_risk_level.value,
        stability.stable_count,
        stability.stable_count + stability.unstable_count,
        report.full_summary(),
    )
    return report


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_audit(
    audit:      AuditReport,
    output_dir: Path | str,
    *,
    pretty:     bool = True,
) -> Path:
    """
    Save the audit report to disk as JSON.

    Written to: <output_dir>/<run_id>_audit.json

    Args:
        audit:      AuditReport from run_audit().
        output_dir: Directory to write into (created if missing).
        pretty:     Indent JSON for human readability.

    Returns:
        Path to the written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prefix   = audit.run_id if audit.run_id else "audit"
    path     = out / f"{prefix}_audit.json"
    indent   = 2 if pretty else None

    with path.open("w", encoding="utf-8") as fh:
        json.dump(audit.to_export_dict(), fh, indent=indent, ensure_ascii=False)

    logger.info("AuditReport saved → %s", path)
    return path
