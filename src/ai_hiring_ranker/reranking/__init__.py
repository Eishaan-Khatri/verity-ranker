"""
Layer 11 — Listwise Re-Ranking.

Takes the Layer 10 RankedOutput and produces a final globally-consistent
ranking by comparing all top candidates together rather than in isolation.

Why this matters
----------------
Layer 10 scores every candidate independently — a 0.4-point gap between
two candidates is decided without any direct comparison. Listwise
re-ranking fixes this by showing the full candidate slate at once and
resolving close ties using evidence-based reasoning.

Two modes
---------
LLM mode (GPT-4o-mini)
  Receives the full candidate slate with scores, strengths, and risks.
  Returns a corrected rank order with pairwise justifications.

Rules mode (fallback, no API key needed)
  Deterministic tie-breaking: proof_strength → skill_fit →
  seniority_match → career_growth → experience_depth.

Output
------
RerankResult contains:
  - Final ordered list (RerankEntry per candidate)
  - rank_delta per candidate (how many positions they moved)
  - rank_confidence: HIGH / MEDIUM / LOW / UNSTABLE
  - PairwiseJustifications for close or swapped pairs
    (feeds the "why A above B" section of Layer 12 recruiter report)

Public API
----------
from ai_hiring_ranker.reranking import (
    rerank,               # main entry point → RerankResult
    RerankResult,
    RerankEntry,
    PairwiseJustification,
    RankConfidence,
)
"""

from .reranker import rerank
from .schemas import (
    PairwiseJustification,
    RankConfidence,
    RerankEntry,
    RerankResult,
)

__all__ = [
    "rerank",
    "RerankResult",
    "RerankEntry",
    "PairwiseJustification",
    "RankConfidence",
]
