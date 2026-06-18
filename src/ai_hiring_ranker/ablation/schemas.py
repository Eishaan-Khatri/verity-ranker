"""
Output schemas for Layer 15 — Evaluation & Ablation Report.

Layer 15 answers two questions:

  1. How good is the final ranking overall?
     → EvaluationReport: quality metrics on the ranked output, pipeline
       health summary, dimension score distributions, and a narrative
       explanation of what the system did well and where it struggled.

  2. What is the contribution of each component?
     → AblationReport: re-runs the rubric scorer with individual signals
       removed (one at a time) and measures how much each signal changed
       the ranking (Kendall's τ). Signals that change ranking a lot when
       removed are high-importance. Signals that change nothing are
       redundant or dominated.

Together they form the EvalAblationBundle, written to:
  outputs/final/<run_id>_eval_ablation.json
  outputs/final/<run_id>_eval_ablation.md

Consumed by:
  - Developers   — understand what's working, tune weights
  - Recruiters   — understand how much to trust the ranking
  - Presentations — show that the system is evidence-backed and measurable
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dimension quality summary
# ---------------------------------------------------------------------------


class DimensionSummary(BaseModel):
    """Statistical summary for one scoring dimension across all candidates."""

    dimension:    str   = Field(...)
    weight:       float = Field(description="Configured weight (0–1).")
    mean:         float = Field(description="Mean raw score across all candidates.")
    std_dev:      float = Field(description="Standard deviation of raw scores.")
    min_score:    float = Field(description="Lowest raw score observed.")
    max_score:    float = Field(description="Highest raw score observed.")
    strong_count: int   = Field(default=0, description="Candidates scoring ≥ 0.80.")
    weak_count:   int   = Field(default=0, description="Candidates scoring < 0.30.")

    @property
    def spread(self) -> float:
        return round(self.max_score - self.min_score, 4)

    @property
    def discriminative_power(self) -> str:
        """How well this dimension separates candidates."""
        if self.std_dev >= 0.20:
            return "high"
        if self.std_dev >= 0.10:
            return "medium"
        return "low"

    def row(self) -> str:
        return (
            f"  {self.dimension:<20} "
            f"μ={self.mean:.3f}  σ={self.std_dev:.3f}  "
            f"range=[{self.min_score:.2f}–{self.max_score:.2f}]  "
            f"power={self.discriminative_power}"
        )


# ---------------------------------------------------------------------------
# Pipeline quality evaluation
# ---------------------------------------------------------------------------


class EvaluationReport(BaseModel):
    """
    Quality evaluation of one complete pipeline run.

    Does not require ground-truth labels — metrics are computed from
    internal signals:
      - Score distribution shape (spread, std dev)
      - Dimension discriminative power
      - Proof coverage (fraction of candidates with verified claims)
      - Retrieval recall (did shortlist cover high-scoring candidates?)
      - Rank stability ratio (from Layer 13)
      - Fairness risk level (from Layer 13)
    """

    run_id:              str                   = Field(default="")
    job_title:           str                   = Field(default="")
    candidate_count:     int                   = Field(default=0)
    evaluated_at:        datetime              = Field(default_factory=datetime.utcnow)

    # Score distribution
    score_mean:          float                 = Field(default=0.0)
    score_std:           float                 = Field(default=0.0)
    score_min:           float                 = Field(default=0.0)
    score_max:           float                 = Field(default=0.0)
    score_spread:        float                 = Field(default=0.0,
        description="max − min. Low spread means the system can't separate candidates well.")

    # Tier distribution
    exceptional_count:   int                   = Field(default=0)  # ≥ 80
    strong_count:        int                   = Field(default=0)   # 65–79
    moderate_count:      int                   = Field(default=0)   # 45–64
    weak_count:          int                   = Field(default=0)   # 25–44
    poor_count:          int                   = Field(default=0)   # < 25

    # Dimension summaries
    dimension_summaries: list[DimensionSummary] = Field(default_factory=list)

    # Proof coverage
    proof_coverage:      float                 = Field(
        default=0.0,
        description=(
            "Fraction of candidates where proof_strength ≥ 0.40. "
            "Low values mean most candidates are unverified."
        ),
    )
    avg_proof_strength:  float                 = Field(default=0.0)
    avg_verified_claims: float                 = Field(default=0.0)

    # Retrieval quality
    shortlist_size:      int                   = Field(default=0)
    shortlist_recall:    float                 = Field(
        default=0.0,
        description=(
            "Fraction of top-10 final candidates that were in the shortlist. "
            "1.0 = retrieval didn't miss any strong candidates."
        ),
    )

    # Audit signals
    stability_ratio:     float                 = Field(default=1.0)
    fairness_risk:       str                   = Field(default="info")
    unstable_count:      int                   = Field(default=0)

    # Narrative
    strengths:           list[str]             = Field(default_factory=list)
    concerns:            list[str]             = Field(default_factory=list)
    recommendations:     list[str]             = Field(default_factory=list)

    @property
    def overall_quality(self) -> str:
        """High-level pipeline quality label."""
        if self.score_spread < 10 and self.candidate_count > 3:
            return "poor"      # can't differentiate candidates
        if self.proof_coverage < 0.20:
            return "limited"   # mostly unverified
        if self.stability_ratio < 0.60:
            return "unstable"  # ranks not reliable
        if self.score_spread >= 30 and self.proof_coverage >= 0.50:
            return "strong"
        return "adequate"

    def to_markdown(self) -> str:
        lines = [
            "## Evaluation Report",
            f"**Run:** {self.run_id}  | **Job:** {self.job_title}  "
            f"| **Candidates:** {self.candidate_count}",
            f"**Overall pipeline quality:** {self.overall_quality.upper()}",
            "",
            "### Score Distribution",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean score     | {self.score_mean:.1f} |",
            f"| Std deviation  | {self.score_std:.1f} |",
            f"| Spread (max−min) | {self.score_spread:.1f} pts |",
            f"| Exceptional (≥80) | {self.exceptional_count} |",
            f"| Strong (65–79)    | {self.strong_count} |",
            f"| Moderate (45–64)  | {self.moderate_count} |",
            f"| Weak/Poor (<45)   | {self.weak_count + self.poor_count} |",
            "",
            "### Dimension Discriminative Power",
        ]
        for ds in sorted(self.dimension_summaries, key=lambda d: -d.std_dev):
            lines.append(ds.row())

        lines += [
            "",
            "### Evidence Coverage",
            f"- Proof coverage (≥0.40 proof strength): {self.proof_coverage:.0%}",
            f"- Average proof strength: {self.avg_proof_strength:.2f}",
            f"- Average verified claims per candidate: {self.avg_verified_claims:.1f}",
            "",
            "### Retrieval Quality",
            f"- Shortlist size: {self.shortlist_size}",
            f"- Shortlist recall (top-10 coverage): {self.shortlist_recall:.0%}",
            "",
            "### Audit Summary",
            f"- Rank stability ratio: {self.stability_ratio:.0%} "
            f"({self.unstable_count} unstable)",
            f"- Fairness risk level: {self.fairness_risk.upper()}",
        ]

        if self.strengths:
            lines += ["", "### What Worked Well"]
            for s in self.strengths:
                lines.append(f"- ✓ {s}")
        if self.concerns:
            lines += ["", "### Concerns"]
            for c in self.concerns:
                lines.append(f"- ⚠ {c}")
        if self.recommendations:
            lines += ["", "### Recommendations"]
            for r in self.recommendations:
                lines.append(f"- → {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ablation: one signal removed
# ---------------------------------------------------------------------------


class AblationRun(BaseModel):
    """
    Result of removing one signal from the pipeline and re-ranking.

    signal_removed:   which dimension/signal was zeroed out
    kendalls_tau:     rank correlation with the full-signal ranking
                      (1.0 = identical, 0.0 = random, -1.0 = reversed)
    rank_changes:     number of candidates whose rank changed
    avg_rank_shift:   mean absolute rank shift across all candidates
    importance_label: derived from kendalls_tau
    """

    signal_removed:  str   = Field(...)
    kendalls_tau:    float = Field(description="Kendall's τ with full-signal ranking.")
    rank_changes:    int   = Field(default=0)
    avg_rank_shift:  float = Field(default=0.0)
    top3_changed:    bool  = Field(
        default=False,
        description="True if any of the top-3 candidates changed rank.",
    )
    notes:           str   = Field(default="")

    @property
    def importance_label(self) -> str:
        """How important is this signal to the final ranking?"""
        tau = abs(self.kendalls_tau)
        if tau < 0.70:
            return "critical"    # removing it radically changes ranking
        if tau < 0.85:
            return "important"   # removing it changes some ranks
        if tau < 0.95:
            return "moderate"    # minor effect
        return "low"             # almost no effect — possibly redundant

    def row(self) -> str:
        top3_flag = " ⚠TOP3" if self.top3_changed else ""
        return (
            f"  {self.signal_removed:<20} "
            f"τ={self.kendalls_tau:.3f}  "
            f"changes={self.rank_changes}  "
            f"avg_shift={self.avg_rank_shift:.2f}  "
            f"[{self.importance_label}]{top3_flag}"
        )


# ---------------------------------------------------------------------------
# Full ablation report
# ---------------------------------------------------------------------------


class AblationReport(BaseModel):
    """
    Complete ablation study for one pipeline run.

    Tests the contribution of each scoring dimension by removing it and
    measuring the impact on the final ranking order.
    """

    run_id:          str              = Field(default="")
    job_title:       str              = Field(default="")
    ablation_runs:   list[AblationRun] = Field(default_factory=list)
    evaluated_at:    datetime         = Field(default_factory=datetime.utcnow)

    @property
    def most_important(self) -> Optional[AblationRun]:
        """The signal whose removal changes ranking the most."""
        if not self.ablation_runs:
            return None
        return min(self.ablation_runs, key=lambda r: r.kendalls_tau)

    @property
    def least_important(self) -> Optional[AblationRun]:
        """The signal whose removal changes ranking the least."""
        if not self.ablation_runs:
            return None
        return max(self.ablation_runs, key=lambda r: r.kendalls_tau)

    def to_markdown(self) -> str:
        lines = [
            "## Ablation Report",
            f"**Run:** {self.run_id}  | **Job:** {self.job_title}",
            "",
            "Each row shows what happens when one dimension is removed from scoring.",
            "τ closer to 1.0 = signal had little effect (redundant).",
            "τ closer to 0.0 = signal was critical for ranking.",
            "",
            f"| Dimension Removed | Kendall's τ | Rank Changes | Avg Shift | Importance |",
            f"|-------------------|-------------|--------------|-----------|------------|",
        ]
        for run in sorted(self.ablation_runs, key=lambda r: r.kendalls_tau):
            top3 = " ⚠" if run.top3_changed else ""
            lines.append(
                f"| {run.signal_removed:<18} "
                f"| {run.kendalls_tau:.3f}       "
                f"| {run.rank_changes:<12} "
                f"| {run.avg_rank_shift:.2f}      "
                f"| {run.importance_label}{top3} |"
            )
        if self.most_important:
            lines += [
                "",
                f"**Most critical signal:** {self.most_important.signal_removed} "
                f"(τ={self.most_important.kendalls_tau:.3f})",
            ]
        if self.least_important:
            lines.append(
                f"**Least impactful signal:** {self.least_important.signal_removed} "
                f"(τ={self.least_important.kendalls_tau:.3f})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Combined bundle
# ---------------------------------------------------------------------------


class EvalAblationBundle(BaseModel):
    """
    Combined Layer 15 output: evaluation + ablation for one run.
    Written to outputs/final/<run_id>_eval_ablation.json and .md
    """

    run_id:     str             = Field(default="")
    evaluation: EvaluationReport = Field(default_factory=EvaluationReport)
    ablation:   AblationReport  = Field(default_factory=AblationReport)

    def to_markdown(self) -> str:
        return "\n\n".join([
            f"# Evaluation & Ablation Report",
            f"**Run:** {self.run_id}",
            "",
            self.evaluation.to_markdown(),
            "---",
            self.ablation.to_markdown(),
        ])

    def to_export_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "evaluation": {
                "overall_quality":   self.evaluation.overall_quality,
                "candidate_count":   self.evaluation.candidate_count,
                "score_mean":        round(self.evaluation.score_mean, 2),
                "score_std":         round(self.evaluation.score_std, 2),
                "score_spread":      round(self.evaluation.score_spread, 2),
                "proof_coverage":    round(self.evaluation.proof_coverage, 3),
                "avg_proof_strength": round(self.evaluation.avg_proof_strength, 3),
                "shortlist_recall":  round(self.evaluation.shortlist_recall, 3),
                "stability_ratio":   round(self.evaluation.stability_ratio, 3),
                "fairness_risk":     self.evaluation.fairness_risk,
                "unstable_count":    self.evaluation.unstable_count,
                "tier_distribution": {
                    "exceptional": self.evaluation.exceptional_count,
                    "strong":      self.evaluation.strong_count,
                    "moderate":    self.evaluation.moderate_count,
                    "weak":        self.evaluation.weak_count,
                    "poor":        self.evaluation.poor_count,
                },
                "dimension_summaries": [
                    {
                        "dimension":           ds.dimension,
                        "weight":              ds.weight,
                        "mean":                round(ds.mean, 4),
                        "std_dev":             round(ds.std_dev, 4),
                        "spread":              round(ds.spread, 4),
                        "discriminative_power": ds.discriminative_power,
                    }
                    for ds in self.evaluation.dimension_summaries
                ],
                "strengths":        self.evaluation.strengths,
                "concerns":         self.evaluation.concerns,
                "recommendations":  self.evaluation.recommendations,
            },
            "ablation": {
                "runs": [
                    {
                        "signal_removed": r.signal_removed,
                        "kendalls_tau":   round(r.kendalls_tau, 4),
                        "rank_changes":   r.rank_changes,
                        "avg_rank_shift": round(r.avg_rank_shift, 3),
                        "top3_changed":   r.top3_changed,
                        "importance":     r.importance_label,
                        "notes":          r.notes,
                    }
                    for r in sorted(self.ablation.ablation_runs, key=lambda r: r.kendalls_tau)
                ],
                "most_critical":    self.ablation.most_important.signal_removed
                                    if self.ablation.most_important else None,
                "least_impactful":  self.ablation.least_important.signal_removed
                                    if self.ablation.least_important else None,
            },
        }
