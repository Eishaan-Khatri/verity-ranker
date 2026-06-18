"""
Evaluation & Ablation Engine — Layer 15.

Produces two reports from a completed PipelineResult:

  1. EvaluationReport — measures pipeline quality without ground truth:
       - Score distribution (mean, std, spread, tier counts)
       - Dimension discriminative power (σ per dimension)
       - Proof coverage (how many candidates are evidence-backed)
       - Retrieval recall (did shortlist miss strong candidates?)
       - Rank stability ratio (from Layer 13 audit)
       - Fairness risk level (from Layer 13 audit)
       - Narrative: strengths, concerns, recommendations

  2. AblationReport — measures each signal's contribution:
       - Removes each scoring dimension one at a time
       - Re-scores all candidates without that dimension
       - Computes Kendall's τ between ablated and full rankings
       - Low τ = signal was critical; high τ = signal is redundant

Both use only data already computed by Layers 1–14.
No new API calls, no new file reads. Pure Python + math.

Public API
----------
run_evaluation(pipeline_result, audit_report)
    → EvaluationReport

run_ablation(pipeline_result)
    → AblationReport

run_eval_ablation(pipeline_result, audit_report, run_id, output_dir)
    → EvalAblationBundle

save_eval_ablation(bundle, output_dir)
    → tuple[Path, Path]   (json_path, md_path)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from ..audits.schemas import AuditReport
from ..evaluation.schemas import PipelineResult
from .schemas import (
    AblationReport,
    AblationRun,
    DimensionSummary,
    EvalAblationBundle,
    EvaluationReport,
)

logger = logging.getLogger(__name__)

# Dimension name → default weight (mirrors scoring_weights.yaml)
_DIMENSIONS = {
    "skill_fit":        0.30,
    "experience_depth": 0.20,
    "seniority_match":  0.15,
    "domain_match":     0.15,
    "career_growth":    0.10,
    "proof_strength":   0.10,
}


# ---------------------------------------------------------------------------
# Kendall's τ
# ---------------------------------------------------------------------------


def _kendalls_tau(rank_a: list[int], rank_b: list[int]) -> float:
    """
    Compute Kendall's τ between two rank lists (concordance measure).
    1.0 = identical order, 0.0 = uncorrelated, -1.0 = reversed.
    """
    n = len(rank_a)
    if n < 2:
        return 1.0

    # Build position maps
    pos_a = {cid: i for i, cid in enumerate(rank_a)}
    pos_b = {cid: i for i, cid in enumerate(rank_b)}
    ids   = list(set(rank_a) & set(rank_b))

    concordant = discordant = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            da = pos_a[ids[i]] - pos_a[ids[j]]
            db = pos_b[ids[i]] - pos_b[ids[j]]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1

    pairs = concordant + discordant
    return round((concordant - discordant) / pairs, 4) if pairs > 0 else 1.0


# ---------------------------------------------------------------------------
# Score arithmetic helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return round(math.sqrt(variance), 4)


def _get_dim(row: dict, dim: str) -> float:
    return float(row.get(dim, 0.0))


# ---------------------------------------------------------------------------
# Ablation: remove one dimension, re-score, measure rank change
# ---------------------------------------------------------------------------


def _ablation_run(
    ranked_rows: list[dict],
    signal_removed: str,
    base_ranked_ids: list[str],
) -> AblationRun:
    """
    Re-score all candidates with one dimension zeroed out and renormalised.
    Returns an AblationRun with the Kendall's τ vs the full ranking.
    """
    remaining_dims = {k: v for k, v in _DIMENSIONS.items() if k != signal_removed}
    total_weight   = sum(remaining_dims.values())
    normed_weights = {k: v / total_weight for k, v in remaining_dims.items()}

    # Re-score
    ablated_scores: list[tuple[str, float]] = []
    for row in ranked_rows:
        score = sum(
            _get_dim(row, dim) * w
            for dim, w in normed_weights.items()
        ) * 100.0
        ablated_scores.append((row["candidate_id"], round(score, 4)))

    ablated_ranked = [
        cid for cid, _ in sorted(ablated_scores, key=lambda x: x[1], reverse=True)
    ]

    tau         = _kendalls_tau(base_ranked_ids, ablated_ranked)
    rank_map_base   = {cid: i + 1 for i, cid in enumerate(base_ranked_ids)}
    rank_map_ablated = {cid: i + 1 for i, cid in enumerate(ablated_ranked)}

    changes    = sum(
        1 for cid in base_ranked_ids
        if abs(rank_map_base.get(cid, 0) - rank_map_ablated.get(cid, 0)) > 0
    )
    shifts     = [
        abs(rank_map_base.get(cid, 0) - rank_map_ablated.get(cid, 0))
        for cid in base_ranked_ids
    ]
    avg_shift  = _mean([float(s) for s in shifts])

    # Check if top-3 changed
    top3_changed = any(
        rank_map_ablated.get(cid, 999) > 3
        for cid in base_ranked_ids[:3]
    )

    return AblationRun(
        signal_removed=signal_removed,
        kendalls_tau=tau,
        rank_changes=changes,
        avg_rank_shift=avg_shift,
        top3_changed=top3_changed,
        notes=(
            f"Remaining weight redistributed to: "
            f"{', '.join(f'{k}={v:.2f}' for k, v in normed_weights.items())}"
        ),
    )


# ---------------------------------------------------------------------------
# Audit 1: Evaluation Report
# ---------------------------------------------------------------------------


def run_evaluation(
    pipeline_result: PipelineResult,
    audit_report:    Optional[AuditReport] = None,
) -> EvaluationReport:
    """
    Compute the pipeline quality evaluation report.

    Args:
        pipeline_result: PipelineResult from Layer 14.
        audit_report:    AuditReport from Layer 13 (optional — adds
                         stability and fairness signals).

    Returns:
        EvaluationReport with quality metrics and narrative.
    """
    rows = pipeline_result.ranked_output
    if not rows:
        return EvaluationReport(
            run_id=pipeline_result.run_id,
            job_title=pipeline_result.job_title,
        )

    # ── Score distribution ─────────────────────────────────────────────
    scores = [float(r.get("final_score", 0.0)) for r in rows]
    score_mean   = _mean(scores)
    score_std    = _std(scores)
    score_min    = min(scores)
    score_max    = max(scores)
    score_spread = round(score_max - score_min, 2)

    # Tier counts
    exceptional = sum(1 for s in scores if s >= 80)
    strong      = sum(1 for s in scores if 65 <= s < 80)
    moderate    = sum(1 for s in scores if 45 <= s < 65)
    weak        = sum(1 for s in scores if 25 <= s < 45)
    poor        = sum(1 for s in scores if s < 25)

    # ── Dimension summaries ────────────────────────────────────────────
    dim_summaries: list[DimensionSummary] = []
    for dim, weight in _DIMENSIONS.items():
        dim_vals = [_get_dim(r, dim) for r in rows]
        dim_summaries.append(DimensionSummary(
            dimension=dim,
            weight=weight,
            mean=_mean(dim_vals),
            std_dev=_std(dim_vals),
            min_score=round(min(dim_vals), 4),
            max_score=round(max(dim_vals), 4),
            strong_count=sum(1 for v in dim_vals if v >= 0.80),
            weak_count=sum(1 for v in dim_vals if v < 0.30),
        ))

    # ── Proof coverage ─────────────────────────────────────────────────
    proof_vals = [_get_dim(r, "proof_strength") for r in rows]
    proof_coverage   = sum(1 for v in proof_vals if v >= 0.40) / len(proof_vals)
    avg_proof        = _mean(proof_vals)
    verified_counts  = [float(r.get("verified_claims", 0)) for r in rows]
    avg_verified     = _mean(verified_counts)

    # ── Retrieval recall ───────────────────────────────────────────────
    # top-10 candidates in final output — were they all in the shortlist?
    # We infer this from Layer 8 note in layer_records
    shortlist_record = pipeline_result.get_layer(8)
    shortlist_size   = 0
    shortlist_recall = 1.0  # assume perfect if we can't measure
    if shortlist_record and shortlist_record.notes:
        # Parse "Shortlisted X/Y candidates" from notes
        parts = shortlist_record.notes.split("/")
        if len(parts) == 2:
            try:
                shortlist_size = int(parts[0].split()[-1])
            except (ValueError, IndexError):
                pass

    top10_ids = [r["candidate_id"] for r in rows[:10]]
    # If shortlist was larger than 10, recall is guaranteed 1.0 for top-10
    if shortlist_size >= len(rows):
        shortlist_recall = 1.0
    elif shortlist_size > 0 and shortlist_size < len(top10_ids):
        # Some top-10 may have been outside shortlist
        shortlist_recall = round(shortlist_size / len(top10_ids), 3)

    # ── Audit signals ──────────────────────────────────────────────────
    stability_ratio = 1.0
    fairness_risk   = "info"
    unstable_count  = 0

    if audit_report:
        stability_ratio = audit_report.stability.stability_ratio
        fairness_risk   = audit_report.fairness.overall_risk_level.value
        unstable_count  = audit_report.stability.unstable_count

    # ── Narrative ──────────────────────────────────────────────────────
    strengths:       list[str] = []
    concerns:        list[str] = []
    recommendations: list[str] = []

    # Score spread
    if score_spread >= 30:
        strengths.append(
            f"Good candidate differentiation — score spread of {score_spread:.1f} pts "
            f"makes ranking decisions clear."
        )
    elif score_spread < 10 and len(rows) > 3:
        concerns.append(
            f"Low score spread ({score_spread:.1f} pts) — the pipeline struggles to "
            f"separate candidates. Consider adding more evidence sources."
        )

    # Proof coverage
    if proof_coverage >= 0.60:
        strengths.append(
            f"{proof_coverage:.0%} of candidates have verified claims — "
            f"high evidence quality."
        )
    elif proof_coverage < 0.30:
        concerns.append(
            f"Only {proof_coverage:.0%} of candidates have meaningful claim verification. "
            f"Most rankings are based on resume text only."
        )
        recommendations.append(
            "Ask candidates to provide GitHub / portfolio links to enable claim verification."
        )

    # Discriminative power
    low_power_dims = [
        ds.dimension for ds in dim_summaries
        if ds.discriminative_power == "low"
    ]
    if low_power_dims:
        concerns.append(
            f"Low discriminative power in: {', '.join(low_power_dims)}. "
            f"These dimensions are not separating candidates."
        )
        recommendations.append(
            f"Review scoring logic for {', '.join(low_power_dims)} — "
            f"scores may be too uniform."
        )

    # Stability
    if stability_ratio >= 0.90:
        strengths.append(
            f"High rank stability ({stability_ratio:.0%}) — "
            f"rankings are robust to small score changes."
        )
    elif stability_ratio < 0.60:
        concerns.append(
            f"Low rank stability ({stability_ratio:.0%}) — "
            f"{unstable_count} candidate(s) have fragile ranks."
        )
        recommendations.append(
            "Increase shortlist size or add more evidence to stabilise borderline candidates."
        )

    # Fairness
    if fairness_risk == "high":
        concerns.append("High fairness risk detected — possible proxy bias in ranking.")
        recommendations.append(
            "Review audit report for proxy bias details. "
            "Consider blind evaluation for flagged attributes."
        )
    elif fairness_risk == "warning":
        concerns.append("Fairness warning detected — review audit report.")

    logger.info(
        "Evaluation complete: quality=%s spread=%.1f proof=%.0%% stable=%.0%%",
        _get_quality_label(score_spread, proof_coverage, stability_ratio),
        score_spread,
        proof_coverage,
        stability_ratio,
    )

    return EvaluationReport(
        run_id=pipeline_result.run_id,
        job_title=pipeline_result.job_title,
        candidate_count=len(rows),
        score_mean=round(score_mean, 2),
        score_std=round(score_std, 2),
        score_min=round(score_min, 2),
        score_max=round(score_max, 2),
        score_spread=score_spread,
        exceptional_count=exceptional,
        strong_count=strong,
        moderate_count=moderate,
        weak_count=weak,
        poor_count=poor,
        dimension_summaries=dim_summaries,
        proof_coverage=round(proof_coverage, 3),
        avg_proof_strength=round(avg_proof, 3),
        avg_verified_claims=round(avg_verified, 2),
        shortlist_size=shortlist_size,
        shortlist_recall=shortlist_recall,
        stability_ratio=round(stability_ratio, 3),
        fairness_risk=fairness_risk,
        unstable_count=unstable_count,
        strengths=strengths,
        concerns=concerns,
        recommendations=recommendations,
    )


def _get_quality_label(spread: float, proof: float, stability: float) -> str:
    if spread < 10 or stability < 0.60:
        return "limited"
    if proof >= 0.50 and spread >= 30:
        return "strong"
    return "adequate"


# ---------------------------------------------------------------------------
# Audit 2: Ablation Report
# ---------------------------------------------------------------------------


def run_ablation(pipeline_result: PipelineResult) -> AblationReport:
    """
    Run ablation study — remove each dimension one at a time and measure impact.

    Args:
        pipeline_result: PipelineResult from Layer 14.

    Returns:
        AblationReport with one AblationRun per dimension.
    """
    rows = pipeline_result.ranked_output
    if not rows:
        return AblationReport(
            run_id=pipeline_result.run_id,
            job_title=pipeline_result.job_title,
        )

    base_ranked_ids = [r["candidate_id"] for r in rows]
    ablation_runs: list[AblationRun] = []

    for dim in _DIMENSIONS:
        try:
            run = _ablation_run(rows, dim, base_ranked_ids)
            ablation_runs.append(run)
            logger.debug(
                "Ablation [−%s]: τ=%.3f  changes=%d  importance=%s",
                dim, run.kendalls_tau, run.rank_changes, run.importance_label,
            )
        except Exception as exc:
            logger.warning("Ablation run failed for %s: %s", dim, exc)
            ablation_runs.append(AblationRun(
                signal_removed=dim,
                kendalls_tau=1.0,
                notes=f"Ablation failed: {exc}",
            ))

    logger.info(
        "Ablation complete. Most critical: %s  Least impactful: %s",
        min(ablation_runs, key=lambda r: r.kendalls_tau).signal_removed
        if ablation_runs else "n/a",
        max(ablation_runs, key=lambda r: r.kendalls_tau).signal_removed
        if ablation_runs else "n/a",
    )

    return AblationReport(
        run_id=pipeline_result.run_id,
        job_title=pipeline_result.job_title,
        ablation_runs=ablation_runs,
    )


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


def run_eval_ablation(
    pipeline_result: PipelineResult,
    audit_report:    Optional[AuditReport] = None,
    *,
    run_id:          str = "",
) -> EvalAblationBundle:
    """
    Run both the evaluation and ablation reports.

    Args:
        pipeline_result: PipelineResult from Layer 14.
        audit_report:    AuditReport from Layer 13 (optional).
        run_id:          Override run_id (uses pipeline_result.run_id if empty).

    Returns:
        EvalAblationBundle with both reports.
    """
    resolved_id = run_id or pipeline_result.run_id

    logger.info("Layer 15: running evaluation...")
    evaluation = run_evaluation(pipeline_result, audit_report)

    logger.info("Layer 15: running ablation study...")
    ablation   = run_ablation(pipeline_result)

    return EvalAblationBundle(
        run_id=resolved_id,
        evaluation=evaluation,
        ablation=ablation,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_eval_ablation(
    bundle:     EvalAblationBundle,
    output_dir: Path | str,
    *,
    pretty:     bool = True,
) -> tuple[Path, Path]:
    """
    Save the evaluation + ablation bundle as JSON and Markdown.

    Files written:
      <output_dir>/<run_id>_eval_ablation.json
      <output_dir>/<run_id>_eval_ablation.md

    Returns:
        (json_path, md_path)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prefix   = bundle.run_id or "eval"
    json_path = out / f"{prefix}_eval_ablation.json"
    md_path   = out / f"{prefix}_eval_ablation.md"
    indent    = 2 if pretty else None

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(bundle.to_export_dict(), fh, indent=indent, ensure_ascii=False)

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(bundle.to_markdown())

    logger.info("EvalAblation saved → JSON: %s  | MD: %s", json_path, md_path)
    return json_path, md_path
