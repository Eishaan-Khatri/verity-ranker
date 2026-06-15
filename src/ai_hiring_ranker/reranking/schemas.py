"""
Output schemas for Layer 11 — Listwise Re-Ranking.

Layer 10 scores candidates independently using a weighted rubric.
That produces a good first-order ranking, but it has a blind spot:
candidates are evaluated in isolation, so a score of 71 vs 70 is
decided without ever comparing the two candidates directly.

Listwise re-ranking fixes this by showing the LLM (or the rule engine)
the full top-N candidate slate at the same time and asking:
"given everything you can see, what is the correct global order?"

This layer produces:
  - A revised rank order (may match Layer 10 exactly, or swap close pairs)
  - A rank_delta per candidate (how many positions they moved)
  - A confidence flag per candidate (stable vs unstable rank)
  - Pairwise justifications for any rank swaps

Consumed by:
  - Layer 12 (Recruiter Report) — uses reranked order and justifications
  - Final Output                — the reranked order is the final answer

Design decisions:
  - RerankEntry wraps a CandidateScore with the new rank and delta.
  - RerankResult is the primary output — sorted by reranked_rank.
  - PairwiseJustification stores why candidate A was placed above B.
    This feeds the "why A above B" section of the recruiter report.
  - rank_confidence: HIGH if the score gap to adjacent candidates is
    large, LOW if the margin was narrow and the LLM/rules disagree.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from ..scoring.schemas import CandidateScore


# ---------------------------------------------------------------------------
# Rank confidence
# ---------------------------------------------------------------------------


class RankConfidence(str, Enum):
    HIGH     = "high"      # clear separation from adjacent candidates
    MEDIUM   = "medium"    # some separation, rank is reliable
    LOW      = "low"       # very close scores — rank could flip on re-run
    UNSTABLE = "unstable"  # rank changed significantly from Layer 10 order


# ---------------------------------------------------------------------------
# Single entry in the reranked list
# ---------------------------------------------------------------------------


class RerankEntry(BaseModel):
    """
    One candidate in the final reranked list.

    Carries the original Layer 10 score plus the new rank, delta,
    and confidence assessment.
    """

    candidate_score:   CandidateScore  = Field(...)
    original_rank:     int             = Field(ge=1, description="Rank from Layer 10.")
    reranked_rank:     int             = Field(ge=1, description="Final rank after listwise re-rank.")
    rank_delta:        int             = Field(
        default=0,
        description=(
            "How many positions the candidate moved. "
            "Positive = moved up, negative = moved down, 0 = unchanged."
        ),
    )
    rank_confidence:   RankConfidence  = Field(default=RankConfidence.MEDIUM)
    rerank_note:       str             = Field(
        default="",
        description="Short explanation of why this rank was assigned or changed.",
    )

    @property
    def candidate_id(self) -> str:
        return self.candidate_score.candidate_id

    @property
    def candidate_name(self) -> str:
        return self.candidate_score.candidate_name

    @property
    def final_score(self) -> float:
        return self.candidate_score.final_score

    @property
    def moved_up(self) -> bool:
        return self.rank_delta > 0

    @property
    def moved_down(self) -> bool:
        return self.rank_delta < 0

    @property
    def is_stable(self) -> bool:
        return self.rank_confidence in (RankConfidence.HIGH, RankConfidence.MEDIUM)

    def summary_line(self) -> str:
        delta_str = (
            f"↑{self.rank_delta}"  if self.rank_delta > 0 else
            f"↓{abs(self.rank_delta)}" if self.rank_delta < 0 else
            "="
        )
        return (
            f"#{self.reranked_rank:<3} {self.candidate_id:<12} "
            f"score={self.final_score:.1f}  "
            f"was=#{self.original_rank}  {delta_str}  "
            f"[{self.rank_confidence.value}]"
        )


# ---------------------------------------------------------------------------
# Pairwise justification — why A is ranked above B
# ---------------------------------------------------------------------------


class PairwiseJustification(BaseModel):
    """
    Explains why candidate A is ranked above candidate B.

    Generated only for pairs where the margin is close (< 5 pts) or
    where a rank swap occurred relative to Layer 10.
    """

    candidate_a:       str        = Field(..., description="The higher-ranked candidate.")
    candidate_b:       str        = Field(..., description="The lower-ranked candidate.")
    score_gap:         float      = Field(description="final_score(A) − final_score(B).")
    deciding_factors:  list[str]  = Field(
        default_factory=list,
        description="What evidence tipped the decision toward A.",
    )
    uncertainty:       str        = Field(
        default="",
        description="What remains uncertain or could flip this decision.",
    )
    was_swapped:       bool       = Field(
        default=False,
        description="True if A was ranked below B in Layer 10 but above B here.",
    )


# ---------------------------------------------------------------------------
# Full rerank result
# ---------------------------------------------------------------------------


class RerankResult(BaseModel):
    """
    Complete listwise re-ranking result for one pipeline run.

    entries is the final ordered list — this is what Layer 12 and the
    final output use. The original Layer 10 order is preserved in
    each RerankEntry.original_rank for full auditability.
    """

    job_title:              str                        = Field(default="")
    entries:                list[RerankEntry]          = Field(
        default_factory=list,
        description="Final ranked list, sorted by reranked_rank ascending.",
    )
    pairwise_justifications: list[PairwiseJustification] = Field(
        default_factory=list,
        description="Justifications for close-margin or swapped pairs.",
    )
    rerank_method:          str = Field(
        default="rules",
        description="'llm' or 'rules' — which path was used.",
    )
    unstable_count:         int = Field(
        default=0,
        description="Number of candidates with LOW or UNSTABLE rank confidence.",
    )

    @property
    def top_candidate(self) -> Optional[RerankEntry]:
        return self.entries[0] if self.entries else None

    @property
    def ranked_ids(self) -> list[str]:
        return [e.candidate_id for e in self.entries]

    def get_entry(self, candidate_id: str) -> Optional[RerankEntry]:
        return next((e for e in self.entries if e.candidate_id == candidate_id), None)

    def get_justification(
        self, candidate_a: str, candidate_b: str
    ) -> Optional[PairwiseJustification]:
        return next(
            (j for j in self.pairwise_justifications
             if j.candidate_a == candidate_a and j.candidate_b == candidate_b),
            None,
        )

    def summary_table(self) -> str:
        lines = [
            f"{'Rank':<5} {'Candidate':<12} {'Score':>6}  "
            f"{'Was':>4}  {'Δ':>3}  Confidence"
        ]
        for e in self.entries:
            delta_str = (
                f"+{e.rank_delta}" if e.rank_delta > 0 else
                str(e.rank_delta) if e.rank_delta < 0 else
                "  ="
            )
            lines.append(
                f"{e.reranked_rank:<5} {e.candidate_id:<12} "
                f"{e.final_score:>6.1f}  "
                f"#{e.original_rank:>3}  {delta_str:>3}  "
                f"[{e.rank_confidence.value}]"
            )
        if self.unstable_count:
            lines.append(
                f"\n⚠  {self.unstable_count} candidate(s) with low/unstable rank confidence."
            )
        return "\n".join(lines)
