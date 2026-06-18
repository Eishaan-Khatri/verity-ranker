"""
Output schemas for Layer 13 — Fairness + Proxy Audit & Rank Stability Test.

Layer 13 runs two independent audits on the final ranking:

  1. Fairness + Proxy Audit
     Checks whether protected or proxy attributes (university prestige,
     location, name, graduation year) measurably influenced ranking.
     Reports top-k impact ratios, overdependence flags, and whether
     any proxy feature correlated with rank.

  2. Rank Stability Test
     Re-runs the rubric scoring with small controlled perturbations
     (±5% weight nudges, slight score jitter) to check whether the
     ranking is stable or sensitive to minor changes.
     Candidates with unstable ranks get a lower_confidence flag.

Both audits are non-blocking — they produce warnings and flags that
attach to the final output, but never alter the ranking itself.

Consumed by:
  - Layer 12 (Recruiter Report) — unstable_rank_warnings section
  - Final Output                — audit_summary field in ranked JSON
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Proxy flag severity
# ---------------------------------------------------------------------------


class FlagSeverity(str, Enum):
    INFO    = "info"     # noted but not a concern
    WARNING = "warning"  # potential bias — should be reviewed
    HIGH    = "high"     # strong signal of proxy influence — act on this


# ---------------------------------------------------------------------------
# Single proxy flag
# ---------------------------------------------------------------------------


class ProxyFlag(BaseModel):
    """
    One detected proxy bias signal.

    A proxy attribute is a field that should NOT influence ranking
    (university prestige, location, name, graduation year) but may
    have done so indirectly through the scoring pipeline.
    """

    proxy_field:    str          = Field(
        ...,
        description="The field that may have acted as a proxy (e.g. 'university_name').",
    )
    description:    str          = Field(
        ...,
        description="Plain-language explanation of the detected pattern.",
    )
    severity:       FlagSeverity = Field(default=FlagSeverity.INFO)
    affected_ids:   list[str]    = Field(
        default_factory=list,
        description="Candidate IDs whose rank may have been affected.",
    )
    recommendation: str          = Field(
        default="",
        description="Suggested action to mitigate this bias risk.",
    )


# ---------------------------------------------------------------------------
# Top-k impact ratio
# ---------------------------------------------------------------------------


class TopKImpactRatio(BaseModel):
    """
    What fraction of the top-k ranked candidates share a specific
    attribute value, compared to what would be expected by chance?

    Example: if 80% of top-5 candidates went to the same university
    but only 20% of all candidates did, that's a 4× over-representation.
    """

    attribute:       str   = Field(..., description="The attribute being measured.")
    value:           str   = Field(..., description="The specific value (e.g. 'MIT').")
    k:               int   = Field(..., description="The top-k window size.")
    top_k_ratio:     float = Field(..., ge=0.0, le=1.0,
                                   description="Fraction in top-k with this value.")
    baseline_ratio:  float = Field(..., ge=0.0, le=1.0,
                                   description="Fraction in all candidates with this value.")
    impact_ratio:    float = Field(
        default=1.0,
        description="top_k_ratio / baseline_ratio. >2.0 = over-represented.",
    )

    @property
    def is_over_represented(self) -> bool:
        return self.impact_ratio >= 2.0

    def summary(self) -> str:
        return (
            f"{self.attribute}='{self.value}': "
            f"{self.top_k_ratio:.0%} of top-{self.k} vs "
            f"{self.baseline_ratio:.0%} baseline "
            f"(×{self.impact_ratio:.1f})"
        )


# ---------------------------------------------------------------------------
# Fairness audit result
# ---------------------------------------------------------------------------


class FairnessAuditResult(BaseModel):
    """
    Complete fairness and proxy bias audit result for one pipeline run.
    """

    proxy_flags:        list[ProxyFlag]       = Field(default_factory=list)
    top_k_ratios:       list[TopKImpactRatio] = Field(default_factory=list)
    prestige_bias_risk: bool                  = Field(
        default=False,
        description="True if top-k over-representation of a single institution was detected.",
    )
    location_bias_risk: bool                  = Field(
        default=False,
        description="True if location appears correlated with rank.",
    )
    overall_risk_level: FlagSeverity          = Field(default=FlagSeverity.INFO)
    audit_notes:        list[str]             = Field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return self.overall_risk_level in (FlagSeverity.WARNING, FlagSeverity.HIGH)

    def summary_lines(self) -> list[str]:
        lines = [f"Fairness Audit  [{self.overall_risk_level.value.upper()}]"]
        for flag in self.proxy_flags:
            lines.append(f"  [{flag.severity.value}] {flag.proxy_field}: {flag.description}")
        for ratio in self.top_k_ratios:
            if ratio.is_over_represented:
                lines.append(f"  [ratio] {ratio.summary()}")
        return lines


# ---------------------------------------------------------------------------
# Rank stability result — one candidate
# ---------------------------------------------------------------------------


class CandidateStability(BaseModel):
    """Rank stability assessment for one candidate."""

    candidate_id:       str   = Field(...)
    base_rank:          int   = Field(ge=1, description="Rank in the final output.")
    base_score:         float = Field(description="Final score from Layer 10.")
    min_rank_observed:  int   = Field(ge=1, description="Lowest rank seen across perturbation runs.")
    max_rank_observed:  int   = Field(ge=1, description="Highest rank seen across perturbation runs.")
    rank_variance:      float = Field(
        default=0.0,
        description="Variance of rank across perturbation runs.",
    )
    score_std_dev:      float = Field(
        default=0.0,
        description="Standard deviation of score across perturbation runs.",
    )
    is_stable:          bool  = Field(
        default=True,
        description="True if rank varied by ≤ 1 position across all perturbations.",
    )
    stability_note:     str   = Field(default="")

    @property
    def rank_range(self) -> int:
        return self.max_rank_observed - self.min_rank_observed

    def summary(self) -> str:
        stable_str = "✓ stable" if self.is_stable else "⚠ unstable"
        return (
            f"{self.candidate_id:<12} rank={self.base_rank}  "
            f"range=[{self.min_rank_observed}–{self.max_rank_observed}]  "
            f"score_σ={self.score_std_dev:.2f}  {stable_str}"
        )


# ---------------------------------------------------------------------------
# Rank stability audit result
# ---------------------------------------------------------------------------


class StabilityAuditResult(BaseModel):
    """
    Complete rank stability audit result for one pipeline run.
    """

    perturbation_runs:    int                       = Field(
        default=0,
        description="Number of perturbation runs executed.",
    )
    perturbation_method:  str                       = Field(
        default="weight_jitter",
        description="How scores were perturbed: 'weight_jitter' or 'score_jitter'.",
    )
    candidate_stability:  list[CandidateStability]  = Field(default_factory=list)
    unstable_ids:         list[str]                 = Field(
        default_factory=list,
        description="Candidate IDs whose rank was unstable across perturbations.",
    )
    stable_count:         int                       = Field(default=0)
    unstable_count:       int                       = Field(default=0)
    audit_notes:          list[str]                 = Field(default_factory=list)

    @property
    def stability_ratio(self) -> float:
        total = self.stable_count + self.unstable_count
        return round(self.stable_count / total, 3) if total > 0 else 1.0

    def summary_lines(self) -> list[str]:
        lines = [
            f"Stability Audit  "
            f"({self.perturbation_runs} runs, method={self.perturbation_method})  "
            f"stable={self.stable_count}  unstable={self.unstable_count}"
        ]
        for cs in sorted(self.candidate_stability, key=lambda x: x.base_rank):
            lines.append(f"  {cs.summary()}")
        return lines


# ---------------------------------------------------------------------------
# Combined audit report
# ---------------------------------------------------------------------------


class AuditReport(BaseModel):
    """
    Combined Layer 13 audit report — fairness + stability.

    Attached to the final pipeline output as an audit_summary.
    Never modifies the ranking — only flags concerns.
    """

    run_id:    str                  = Field(default="")
    job_title: str                  = Field(default="")
    fairness:  FairnessAuditResult  = Field(default_factory=FairnessAuditResult)
    stability: StabilityAuditResult = Field(default_factory=StabilityAuditResult)

    @property
    def has_any_warnings(self) -> bool:
        return self.fairness.has_warnings or self.stability.unstable_count > 0

    def full_summary(self) -> str:
        lines = [f"=== Layer 13 Audit Report: {self.job_title} ==="]
        lines += self.fairness.summary_lines()
        lines.append("")
        lines += self.stability.summary_lines()
        return "\n".join(lines)

    def to_export_dict(self) -> dict:
        return {
            "run_id":              self.run_id,
            "job_title":           self.job_title,
            "fairness_risk_level": self.fairness.overall_risk_level.value,
            "fairness_flags":      [
                {
                    "proxy_field":    f.proxy_field,
                    "description":    f.description,
                    "severity":       f.severity.value,
                    "affected_ids":   f.affected_ids,
                    "recommendation": f.recommendation,
                }
                for f in self.fairness.proxy_flags
            ],
            "top_k_ratios": [
                {
                    "attribute":      r.attribute,
                    "value":          r.value,
                    "k":              r.k,
                    "top_k_ratio":    round(r.top_k_ratio, 3),
                    "baseline_ratio": round(r.baseline_ratio, 3),
                    "impact_ratio":   round(r.impact_ratio, 2),
                }
                for r in self.fairness.top_k_ratios
                if r.is_over_represented
            ],
            "stability_ratio":   round(self.stability.stability_ratio, 3),
            "unstable_candidates": self.stability.unstable_ids,
            "candidate_stability": [
                {
                    "candidate_id": cs.candidate_id,
                    "base_rank":    cs.base_rank,
                    "rank_range":   cs.rank_range,
                    "score_std":    round(cs.score_std_dev, 3),
                    "is_stable":    cs.is_stable,
                }
                for cs in self.stability.candidate_stability
            ],
        }
