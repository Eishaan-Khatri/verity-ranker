"""
Layer 10 — Rubric-Based Scoring.

Applies the six configured dimension weights to Layer 9's EvaluationResults
and produces a final_score (0–100) for every candidate. Every score cites
the evidence chain from the agent verdicts.

Weights (from configs/v2/scoring_weights.yaml):
  skill_fit        0.30
  experience_depth 0.20
  seniority_match  0.15
  domain_match     0.15
  career_growth    0.10
  proof_strength   0.10

Score adjustments applied on top of the weighted sum:
  +3 pts  proof_strength ≥ 0.80 with ≥ 3 verified claims
  +1 pt   proof_strength ≥ 0.65
  ×0.90   severe seniority mismatch (seniority_match < 0.25)
  −10 pts missing a hard-required JD skill (dealbreaker)

Public API
----------
from ai_hiring_ranker.scoring import (
    score_candidates,     # BatchEvaluationResult → RankedOutput
    score_one,            # EvaluationResult → CandidateScore
    save_ranked_output,   # RankedOutput → JSON file
    load_ranked_output,   # JSON file → list[dict]
    get_scoring_weights,  # () → ScoringWeights
    RankedOutput,
    CandidateScore,
    ScoringWeights,
    DimensionBreakdown,
)
"""

from .scorer import (
    get_scoring_weights,
    load_ranked_output,
    save_ranked_output,
    score_candidates,
    score_one,
)
from .schemas import (
    CandidateScore,
    DimensionBreakdown,
    RankedOutput,
    ScoringWeights,
)

__all__ = [
    # Functions
    "score_candidates",
    "score_one",
    "save_ranked_output",
    "load_ranked_output",
    "get_scoring_weights",
    # Schema types
    "RankedOutput",
    "CandidateScore",
    "ScoringWeights",
    "DimensionBreakdown",
]
