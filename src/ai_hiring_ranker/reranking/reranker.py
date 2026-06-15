"""
Listwise Re-Ranker — Layer 11.

Takes the RankedOutput from Layer 10 and produces a final re-ordered
RerankResult by comparing all top candidates together — not in isolation.

Why listwise matters
--------------------
Layer 10 scores every candidate independently:
  - Candidate A scores 71.2
  - Candidate B scores 70.8

The 0.4-point gap may be noise. Layer 10 has no way to ask:
"but given that A has more verified claims and B has higher seniority,
which one would the recruiter actually prefer?"

Listwise re-ranking answers that by showing all candidates at once.

Two execution modes
-------------------
LLM mode (preferred)
  Builds a compact "slate" of all shortlisted candidates and asks the
  LLM to produce a corrected rank order with brief justifications.
  Generates PairwiseJustifications for close or swapped pairs.
  Requires an API key.

Rules mode (fallback — no API key needed)
  Applies a set of deterministic tie-breaking rules to nudge the
  Layer 10 order when scores are within the close-margin threshold:

  Rule priority (applied in order):
    1. Proof strength  — higher verified claim ratio wins
    2. Skill fit       — more required skills covered wins
    3. Seniority match — closer to the JD level wins
    4. Career growth   — stronger upward trajectory wins

  If no rule separates a pair → order is preserved from Layer 10.

Confidence assignment
---------------------
  HIGH     → score gap to nearest neighbour > 5 pts
  MEDIUM   → score gap 2–5 pts
  LOW      → score gap < 2 pts (nearly tied)
  UNSTABLE → rank changed more than 2 positions from Layer 10 order

Public API
----------
rerank(ranked_output, hiring_profile, eval_results, force_fallback)
    → RerankResult
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from ..agents.schemas import BatchEvaluationResult
from ..jd_intelligence.schemas import HiringProfile
from ..scoring.schemas import CandidateScore, RankedOutput
from .schemas import (
    PairwiseJustification,
    RankConfidence,
    RerankEntry,
    RerankResult,
)

logger = logging.getLogger(__name__)

# Scores within this band are treated as "close" for tie-breaking
_CLOSE_MARGIN_PTS = 5.0
# Rank shifts larger than this are flagged as UNSTABLE
_UNSTABLE_DELTA   = 2


# ---------------------------------------------------------------------------
# Confidence assignment
# ---------------------------------------------------------------------------


def _assign_confidence(
    entry: RerankEntry,
    all_entries: list[RerankEntry],
) -> RankConfidence:
    """Assign rank confidence based on score gap to adjacent candidates."""
    rank = entry.reranked_rank
    scores = {e.reranked_rank: e.final_score for e in all_entries}

    above_score = scores.get(rank - 1)
    below_score = scores.get(rank + 1)

    gaps: list[float] = []
    if above_score is not None:
        gaps.append(abs(entry.final_score - above_score))
    if below_score is not None:
        gaps.append(abs(entry.final_score - below_score))

    min_gap = min(gaps) if gaps else 999.0

    # Check stability relative to Layer 10
    if abs(entry.rank_delta) > _UNSTABLE_DELTA:
        return RankConfidence.UNSTABLE

    if min_gap > 5.0:
        return RankConfidence.HIGH
    if min_gap >= 2.0:
        return RankConfidence.MEDIUM
    return RankConfidence.LOW


# ---------------------------------------------------------------------------
# Pairwise justification builder
# ---------------------------------------------------------------------------


def _build_pairwise_justification(
    score_a: CandidateScore,
    score_b: CandidateScore,
    *,
    was_swapped: bool = False,
) -> PairwiseJustification:
    """
    Build a rule-based pairwise justification for why A is above B.
    Used when the score gap is close or a swap occurred.
    """
    deciding_factors: list[str] = []
    score_gap = round(score_a.final_score - score_b.final_score, 2)

    # Which dimensions clearly separate them?
    dim_diffs = {
        "skill_fit":        score_a.skill_fit        - score_b.skill_fit,
        "experience_depth": score_a.experience_depth - score_b.experience_depth,
        "proof_strength":   score_a.proof_strength   - score_b.proof_strength,
        "seniority_match":  score_a.seniority_match  - score_b.seniority_match,
        "career_growth":    score_a.career_growth    - score_b.career_growth,
        "domain_match":     score_a.domain_match     - score_b.domain_match,
    }

    for dim, diff in sorted(dim_diffs.items(), key=lambda x: -abs(x[1])):
        if abs(diff) >= 0.10:
            direction = "higher" if diff > 0 else "lower"
            deciding_factors.append(
                f"{score_a.candidate_id} has {direction} {dim.replace('_', ' ')} "
                f"({getattr(score_a, dim):.2f} vs {getattr(score_b, dim):.2f})"
            )
        if len(deciding_factors) >= 3:
            break

    if not deciding_factors:
        deciding_factors.append(
            f"Marginal score difference ({score_gap:+.1f} pts); "
            f"order preserved from rubric scoring."
        )

    # What remains uncertain?
    low_proof_a = score_a.proof_strength < 0.50
    low_proof_b = score_b.proof_strength < 0.50
    uncertainty_parts: list[str] = []
    if low_proof_a or low_proof_b:
        uncertainty_parts.append(
            "Both candidates have weakly verified claims — "
            "interview performance could reverse this order."
        )
    if abs(score_gap) < 3.0:
        uncertainty_parts.append(
            f"Score gap is only {abs(score_gap):.1f} pts — ranking is sensitive to re-evaluation."
        )

    return PairwiseJustification(
        candidate_a=score_a.candidate_id,
        candidate_b=score_b.candidate_id,
        score_gap=score_gap,
        deciding_factors=deciding_factors,
        uncertainty=" ".join(uncertainty_parts),
        was_swapped=was_swapped,
    )


# ---------------------------------------------------------------------------
# Rules-based reranker
# ---------------------------------------------------------------------------


def _sort_key_rules(score: CandidateScore) -> tuple:
    """
    Multi-key sort tuple for rule-based tie-breaking.
    Primary: final_score (desc), then dimensions in priority order.
    All values negated for descending sort.
    """
    return (
        -score.final_score,
        -score.proof_strength,
        -score.skill_fit,
        -score.seniority_match,
        -score.career_growth,
        -score.experience_depth,
    )


def _rerank_rules(ranked_output: RankedOutput) -> list[CandidateScore]:
    """Apply deterministic tie-breaking rules to the ranked list."""
    return sorted(ranked_output.ranked, key=_sort_key_rules)


def _build_result_from_order(
    reranked_scores: list[CandidateScore],
    original_ranked: list[CandidateScore],
    method: str,
    job_title: str,
) -> RerankResult:
    """Assemble a RerankResult from a reranked score list."""
    original_rank_map = {s.candidate_id: i + 1 for i, s in enumerate(original_ranked)}

    entries: list[RerankEntry] = []
    for new_rank, score in enumerate(reranked_scores, start=1):
        orig_rank = original_rank_map.get(score.candidate_id, new_rank)
        delta     = orig_rank - new_rank  # positive = moved up

        entries.append(RerankEntry(
            candidate_score=score,
            original_rank=orig_rank,
            reranked_rank=new_rank,
            rank_delta=delta,
            rank_confidence=RankConfidence.MEDIUM,  # will be updated below
            rerank_note="",
        ))

    # Assign confidence now that all ranks are set
    for entry in entries:
        entry.rank_confidence = _assign_confidence(entry, entries)
        if entry.rank_delta != 0:
            entry.rerank_note = (
                f"Moved {'up' if entry.rank_delta > 0 else 'down'} "
                f"{abs(entry.rank_delta)} position(s) via {method} tie-breaking."
            )

    # Build pairwise justifications for close or swapped pairs
    justifications: list[PairwiseJustification] = []
    score_map = {s.candidate_id: s for s in reranked_scores}

    for i in range(len(entries) - 1):
        a = entries[i]
        b = entries[i + 1]
        gap = a.final_score - b.final_score

        was_swapped = original_rank_map.get(a.candidate_id, 999) > \
                      original_rank_map.get(b.candidate_id, 999)

        if gap < _CLOSE_MARGIN_PTS or was_swapped:
            score_a = score_map.get(a.candidate_id)
            score_b = score_map.get(b.candidate_id)
            if score_a and score_b:
                justifications.append(
                    _build_pairwise_justification(score_a, score_b, was_swapped=was_swapped)
                )

    unstable_count = sum(
        1 for e in entries
        if e.rank_confidence in (RankConfidence.LOW, RankConfidence.UNSTABLE)
    )

    return RerankResult(
        job_title=job_title,
        entries=entries,
        pairwise_justifications=justifications,
        rerank_method=method,
        unstable_count=unstable_count,
    )


# ---------------------------------------------------------------------------
# LLM-based reranker
# ---------------------------------------------------------------------------


def _build_slate_text(
    ranked_scores: list[CandidateScore],
    hiring_profile: HiringProfile,
    eval_results: Optional[BatchEvaluationResult],
) -> str:
    """Build a compact candidate slate text for the LLM prompt."""
    lines: list[str] = [
        f"Job: {hiring_profile.job_title} ({hiring_profile.seniority.value})",
        f"Required skills: {', '.join(hiring_profile.all_required_skill_names[:8])}",
        "",
        "Current ranking (from rubric scoring):",
    ]

    eval_map = {}
    if eval_results:
        eval_map = {r.candidate_id: r for r in eval_results.results}

    for i, score in enumerate(ranked_scores, 1):
        eval_r = eval_map.get(score.candidate_id)
        summary = eval_r.summary if eval_r else ""
        strengths = "; ".join((eval_r.strengths or [])[:2]) if eval_r else ""
        risks     = "; ".join((eval_r.risks     or [])[:1]) if eval_r else ""

        lines.append(
            f"\n#{i} {score.candidate_id} | score={score.final_score:.1f} | "
            f"skl={score.skill_fit:.2f} exp={score.experience_depth:.2f} "
            f"sen={score.seniority_match:.2f} prf={score.proof_strength:.2f}"
        )
        if summary:
            lines.append(f"   Summary: {summary[:150]}")
        if strengths:
            lines.append(f"   Strengths: {strengths}")
        if risks:
            lines.append(f"   Risks: {risks}")

    return "\n".join(lines)


def _rerank_llm(
    ranked_output: RankedOutput,
    hiring_profile: HiringProfile,
    eval_results: Optional[BatchEvaluationResult],
) -> RerankResult:
    """Use LLM to produce a globally-consistent reranked order."""
    from ..llm_provider import chat_completion

    ranked_scores = ranked_output.ranked
    slate_text    = _build_slate_text(ranked_scores, hiring_profile, eval_results)

    system_prompt = (
        "You are a Final Ranking Agent performing listwise re-ranking. "
        "You are shown all candidates at once. Your job is to decide the correct "
        "global rank order, resolving any close ties using the full context. "
        "Return ONLY a JSON object with keys:\n"
        "  reranked_ids: list of candidate_id strings in final rank order (best first)\n"
        "  justifications: list of objects with keys: "
        "    candidate_a, candidate_b, deciding_factors (list of strings), uncertainty (string)\n"
        "  notes: dict mapping candidate_id to a short rerank note (1 sentence)\n"
        "Do not change the order unless there is a clear evidence-based reason. "
        "Preserve the rubric order when in doubt."
    )

    user_prompt = f"{slate_text}\n\nProduce the final rank order. Return JSON."

    raw = chat_completion(system_prompt, user_prompt)
    data = json.loads(raw.strip().strip("```json").strip("```"))

    reranked_ids: list[str] = data.get("reranked_ids", [])
    llm_justifications: list[dict] = data.get("justifications", [])
    llm_notes: dict[str, str] = data.get("notes", {})

    # Map score objects
    score_map = {s.candidate_id: s for s in ranked_scores}
    original_rank_map = {s.candidate_id: i + 1 for i, s in enumerate(ranked_scores)}

    # Build reranked order — fall back to rubric order for any missing IDs
    seen: set[str] = set()
    reranked_scores: list[CandidateScore] = []
    for cid in reranked_ids:
        if cid in score_map and cid not in seen:
            reranked_scores.append(score_map[cid])
            seen.add(cid)
    # Append any candidates the LLM dropped
    for s in ranked_scores:
        if s.candidate_id not in seen:
            reranked_scores.append(s)

    result = _build_result_from_order(
        reranked_scores,
        ranked_scores,
        method="llm",
        job_title=hiring_profile.job_title,
    )

    # Overwrite rerank notes from LLM
    for entry in result.entries:
        note = llm_notes.get(entry.candidate_id)
        if note:
            entry.rerank_note = note

    # Replace pairwise justifications with LLM's richer versions
    llm_pairwise: list[PairwiseJustification] = []
    score_map2 = {s.candidate_id: s for s in reranked_scores}
    for j in llm_justifications[:10]:
        cid_a = j.get("candidate_a", "")
        cid_b = j.get("candidate_b", "")
        sa = score_map2.get(cid_a)
        sb = score_map2.get(cid_b)
        if sa and sb:
            gap = sa.final_score - sb.final_score
            orig_a = original_rank_map.get(cid_a, 999)
            orig_b = original_rank_map.get(cid_b, 999)
            llm_pairwise.append(PairwiseJustification(
                candidate_a=cid_a,
                candidate_b=cid_b,
                score_gap=round(gap, 2),
                deciding_factors=list(j.get("deciding_factors", [])),
                uncertainty=str(j.get("uncertainty", "")),
                was_swapped=orig_a > orig_b,
            ))

    if llm_pairwise:
        result = result.model_copy(update={"pairwise_justifications": llm_pairwise})

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rerank(
    ranked_output: RankedOutput,
    hiring_profile: HiringProfile,
    eval_results: Optional[BatchEvaluationResult] = None,
    *,
    force_fallback: bool = False,
    top_n: int = 30,
) -> RerankResult:
    """
    Perform listwise re-ranking on the top-N candidates from Layer 10.

    Args:
        ranked_output:  RankedOutput from Layer 10 (score_candidates).
        hiring_profile: HiringProfile from Layer 2.
        eval_results:   BatchEvaluationResult from Layer 9 (adds context
                        for LLM mode — summaries, strengths, risks).
        force_fallback: Always use rule-based re-ranking.
        top_n:          Only rerank the top N candidates (default 30).
                        Candidates outside top_n keep their Layer 10 rank.

    Returns:
        RerankResult with final ordered list, confidence flags,
        and pairwise justifications for close/swapped pairs.
    """
    if not ranked_output.scores:
        return RerankResult(job_title=hiring_profile.job_title)

    # Take top-N from the Layer 10 ranking
    all_ranked = ranked_output.ranked
    to_rerank  = all_ranked[:top_n]
    remainder  = all_ranked[top_n:]

    logger.info(
        "Layer 11 reranking: %d candidates (top_n=%d, force_fallback=%s)",
        len(to_rerank), top_n, force_fallback,
    )

    # Run reranking
    has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY"))

    if not force_fallback and has_key:
        try:
            # Wrap top_n into a temporary RankedOutput for the LLM path
            top_ranked_output = RankedOutput(
                job_title=ranked_output.job_title,
                run_id=ranked_output.run_id,
                scores=to_rerank,
            )
            result = _rerank_llm(top_ranked_output, hiring_profile, eval_results)
            logger.info("Layer 11 used LLM re-ranking.")
        except Exception as exc:
            logger.warning("LLM reranking failed (%s) — falling back to rules.", exc)
            top_ranked_output = RankedOutput(
                job_title=ranked_output.job_title,
                run_id=ranked_output.run_id,
                scores=to_rerank,
            )
            reranked_scores = _rerank_rules(top_ranked_output)
            result = _build_result_from_order(
                reranked_scores, to_rerank, "rules", hiring_profile.job_title
            )
    else:
        logger.info("Layer 11 using rule-based re-ranking.")
        top_ranked_output = RankedOutput(
            job_title=ranked_output.job_title,
            run_id=ranked_output.run_id,
            scores=to_rerank,
        )
        reranked_scores = _rerank_rules(top_ranked_output)
        result = _build_result_from_order(
            reranked_scores, to_rerank, "rules", hiring_profile.job_title
        )

    # Append remainder candidates beyond top_n (unchanged rank)
    if remainder:
        base_rank = len(result.entries) + 1
        for i, score in enumerate(remainder):
            orig_rank = top_n + i + 1
            result.entries.append(RerankEntry(
                candidate_score=score,
                original_rank=orig_rank,
                reranked_rank=base_rank + i,
                rank_delta=0,
                rank_confidence=RankConfidence.MEDIUM,
                rerank_note="Outside top-N window — rank unchanged from Layer 10.",
            ))

    logger.info(
        "Layer 11 complete. %d unstable ranks.\n%s",
        result.unstable_count,
        result.summary_table(),
    )
    return result
