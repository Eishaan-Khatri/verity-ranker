"""
Output schemas for Layer 10 — Rubric-Based Scoring.

Takes the six DimensionScores from Layer 9 (Multi-Agent Evaluation),
applies the configured weights from scoring_weights.yaml, and produces
a CandidateScore with a final_score on a 0–100 scale.

Every score must cite evidence — the per-dimension breakdown and the
evidence citations from Layer 9 verdicts are preserved here so the
recruiter report (Layer 12) can explain every number.

Consumed by:
  - Layer 11 (Listwise Re-Ranking)  — uses final_score as the base rank
  - Layer 12 (Recruiter Report)     — uses dimension breakdown + evidence
  - Final Output                    — matches schemas/ranked_output.schema.json exactly

Design decisions:
  - final_score is 0–100 (not 0–1) to match the ranked_output.schema.json contract
    and to make the numbers human-readable for recruiters.
  - ScoringWeights is a Pydantic model that validates the config at load time
    and enforces that all weights sum to 1.0 (±0.01 tolerance).
  - DimensionBreakdown stores both the raw 0–1 score and the weighted
    contribution so it's always clear why a candidate scored X.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Scoring weights (loaded from scoring_weights.yaml)
# ---------------------------------------------------------------------------


class ScoringWeights(BaseModel):
    """
    Six rubric dimension weights that must sum to 1.0.
    Loaded from configs/v2/scoring_weights.yaml.
    """

    skill_fit:          float = Field(default=0.30, ge=0.0, le=1.0)
    experience_depth:   float = Field(default=0.20, ge=0.0, le=1.0)
    seniority_match:    float = Field(default=0.15, ge=0.0, le=1.0)
    domain_match:       float = Field(default=0.15, ge=0.0, le=1.0)
    career_growth:      float = Field(default=0.10, ge=0.0, le=1.0)
    proof_strength:     float = Field(default=0.10, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "ScoringWeights":
        total = (
            self.skill_fit + self.experience_depth + self.seniority_match +
            self.domain_match + self.career_growth + self.proof_strength
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.4f}. "
                "Update configs/v2/scoring_weights.yaml."
            )
        return self

    def as_dict(self) -> dict[str, float]:
        return {
            "skill_fit":        self.skill_fit,
            "experience_depth": self.experience_depth,
            "seniority_match":  self.seniority_match,
            "domain_match":     self.domain_match,
            "career_growth":    self.career_growth,
            "proof_strength":   self.proof_strength,
        }


# ---------------------------------------------------------------------------
# Per-dimension breakdown — one row in the scoring table
# ---------------------------------------------------------------------------


class DimensionBreakdown(BaseModel):
    """
    Detailed breakdown for one scoring dimension.

    raw_score       — the 0–1 score from Layer 9 agents
    weight          — the configured weight for this dimension
    weighted_score  — raw_score × weight (contribution to final_score)
    evidence        — cited evidence from the agent verdict
    reasoning       — the agent's explanation for this score
    """

    dimension:       str   = Field(...)
    raw_score:       float = Field(ge=0.0, le=1.0)
    weight:          float = Field(ge=0.0, le=1.0)
    weighted_score:  float = Field(ge=0.0)
    evidence:        list[str] = Field(default_factory=list)
    reasoning:       str       = Field(default="")

    @property
    def score_label(self) -> str:
        if self.raw_score >= 0.80:
            return "strong"
        if self.raw_score >= 0.55:
            return "moderate"
        if self.raw_score >= 0.30:
            return "weak"
        return "poor"

    def row(self) -> str:
        """One-line summary for logging / debug output."""
        return (
            f"  {self.dimension:<20} "
            f"raw={self.raw_score:.3f}  "
            f"w={self.weight:.2f}  "
            f"contrib={self.weighted_score:.3f}  "
            f"[{self.score_label}]"
        )


# ---------------------------------------------------------------------------
# Final scored candidate — the main output of Layer 10
# ---------------------------------------------------------------------------


class CandidateScore(BaseModel):
    """
    Complete rubric-based score for one candidate.

    final_score is on a 0–100 scale to match ranked_output.schema.json.
    All dimension breakdowns preserve the evidence chain from Layer 9.
    """

    candidate_id:    str                     = Field(...)
    candidate_name:  str                     = Field(default="")

    # Primary output
    final_score:     float                   = Field(
        ge=0.0, le=100.0,
        description="Weighted rubric score, 0–100. Primary sort key for ranking.",
    )

    # Per-dimension breakdown (one per dimension)
    breakdowns:      list[DimensionBreakdown] = Field(default_factory=list)

    # Raw 0–1 dimension scores (convenience — same values as in breakdowns)
    skill_fit:          float = Field(default=0.0, ge=0.0, le=1.0)
    experience_depth:   float = Field(default=0.0, ge=0.0, le=1.0)
    seniority_match:    float = Field(default=0.0, ge=0.0, le=1.0)
    domain_match:       float = Field(default=0.0, ge=0.0, le=1.0)
    career_growth:      float = Field(default=0.0, ge=0.0, le=1.0)
    proof_strength:     float = Field(default=0.0, ge=0.0, le=1.0)

    # Claim counts (from evidence ledger — set by scorer if ledger available)
    verified_claims:    int   = Field(default=0)
    unverified_claims:  int   = Field(default=0)

    # Pass-through from Layer 9
    strengths:          list[str] = Field(default_factory=list)
    risks:              list[str] = Field(default_factory=list)

    # Scoring metadata
    weights_used:       dict[str, float] = Field(
        default_factory=dict,
        description="Snapshot of the weights applied in this run.",
    )
    score_notes:        list[str] = Field(
        default_factory=list,
        description="Non-critical notes about score adjustments (e.g. bonuses applied).",
    )

    @property
    def score_label(self) -> str:
        if self.final_score >= 80:
            return "exceptional"
        if self.final_score >= 65:
            return "strong"
        if self.final_score >= 45:
            return "moderate"
        if self.final_score >= 25:
            return "weak"
        return "poor"

    @property
    def proof_ratio(self) -> float:
        """Verified / total claims ratio (0–1). 0 if no claims."""
        total = self.verified_claims + self.unverified_claims
        return round(self.verified_claims / total, 3) if total > 0 else 0.0

    def breakdown_table(self) -> str:
        lines = [
            f"{self.candidate_id}  final_score={self.final_score:.1f}/100  [{self.score_label}]"
        ]
        for bd in self.breakdowns:
            lines.append(bd.row())
        return "\n".join(lines)

    def to_export_dict(self) -> dict:
        """Serialise to the format required by schemas/ranked_output.schema.json."""
        return {
            "candidate_id":    self.candidate_id,
            "candidate_name":  self.candidate_name,
            "final_score":     round(self.final_score, 2),
            "skill_fit":       round(self.skill_fit, 4),
            "experience_depth": round(self.experience_depth, 4),
            "seniority_match": round(self.seniority_match, 4),
            "domain_match":    round(self.domain_match, 4),
            "career_growth":   round(self.career_growth, 4),
            "proof_strength":  round(self.proof_strength, 4),
            "verified_claims": self.verified_claims,
            "unverified_claims": self.unverified_claims,
            "strengths":       self.strengths,
            "risks":           self.risks,
        }


# ---------------------------------------------------------------------------
# Ranked output — all candidates sorted and serialisable
# ---------------------------------------------------------------------------


class RankedOutput(BaseModel):
    """
    Full ranked list of all scored candidates for one pipeline run.

    Sorted by final_score descending. Rank is 1-based.
    Serialised to outputs/final/<run_id>_ranked.json, matching
    schemas/ranked_output.schema.json exactly.
    """

    job_title:  str                  = Field(default="")
    run_id:     str                  = Field(default="")
    scores:     list[CandidateScore] = Field(default_factory=list)

    @property
    def ranked(self) -> list[CandidateScore]:
        """Candidates sorted by final_score descending."""
        return sorted(self.scores, key=lambda s: s.final_score, reverse=True)

    def get(self, candidate_id: str) -> Optional[CandidateScore]:
        return next((s for s in self.scores if s.candidate_id == candidate_id), None)

    def to_export_list(self) -> list[dict]:
        """
        Produce a list matching schemas/ranked_output.schema.json.
        Adds the rank field and sorts by final_score.
        """
        output: list[dict] = []
        for rank, score in enumerate(self.ranked, start=1):
            row = score.to_export_dict()
            row["rank"] = rank
            output.append(row)
        return output

    def summary_table(self) -> str:
        lines = [
            f"{'Rank':<5} {'Candidate':<12} {'Score':>6}  "
            f"{'skl':>4} {'exp':>4} {'sen':>4} {'dom':>4} {'trj':>4} {'prf':>4}  Label"
        ]
        for rank, s in enumerate(self.ranked, 1):
            lines.append(
                f"{rank:<5} {s.candidate_id:<12} {s.final_score:>6.1f}  "
                f"{s.skill_fit:.2f} {s.experience_depth:.2f} {s.seniority_match:.2f} "
                f"{s.domain_match:.2f} {s.career_growth:.2f} {s.proof_strength:.2f}  "
                f"[{s.score_label}]"
            )
        return "\n".join(lines)
