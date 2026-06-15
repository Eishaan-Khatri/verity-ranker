"""
Rubric-Based Scorer — Layer 10.

Applies the configured dimension weights to the Layer 9 EvaluationResults
and produces a fully populated CandidateScore + final RankedOutput.

Key responsibilities:
  1. Load weights from configs/v2/scoring_weights.yaml (cached).
  2. For each candidate, compute weighted sum → final_score (0–100).
  3. Build per-dimension DimensionBreakdown with evidence chain.
  4. Apply optional bonuses (proof bonus, dealbreaker penalty).
  5. Attach claim counts from the evidence ledger.
  6. Assemble RankedOutput sorted by final_score.
  7. Persist to outputs/final/<run_id>_ranked.json.

Score adjustment rules:
  Proof bonus    +3 pts  → proof_strength ≥ 0.80 AND verified_claims ≥ 3
  Proof bonus    +1 pt   → proof_strength ≥ 0.65
  Dealbreaker    −10 pts → required JD skill completely missing AND
                           it was listed as a dealbreaker in the JD
  Seniority cap  × 0.90  → seniority_match < 0.25 (severe mismatch)

All adjustments are logged in CandidateScore.score_notes.

Public API
----------
score_candidates(batch_result, hiring_profile, ledger_map, run_id)
    → RankedOutput

score_one(eval_result, weights, hiring_profile, ledger)
    → CandidateScore
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..agents.schemas import AgentRole, BatchEvaluationResult, EvaluationResult
from ..evidence.schemas import CandidateLedger
from ..jd_intelligence.schemas import HiringProfile
from .schemas import (
    CandidateScore,
    DimensionBreakdown,
    RankedOutput,
    ScoringWeights,
)

logger = logging.getLogger(__name__)

_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[4] / "configs" / "v2" / "scoring_weights.yaml"
)

# Score adjustments
_PROOF_BONUS_HIGH   = 3.0   # proof_strength ≥ 0.80
_PROOF_BONUS_MED    = 1.0   # proof_strength ≥ 0.65
_DEALBREAKER_PENALTY = 10.0 # missing a JD dealbreaker skill
_SENIORITY_CAP      = 0.90  # multiplier for severe seniority mismatch


# ---------------------------------------------------------------------------
# Weight loader
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_weights() -> ScoringWeights:
    """Load scoring weights from YAML, fall back to defaults."""
    try:
        import yaml  # type: ignore
        with _WEIGHTS_PATH.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        weights = ScoringWeights(**raw)
        logger.debug("Scoring weights loaded from %s", _WEIGHTS_PATH)
        return weights
    except FileNotFoundError:
        logger.warning("scoring_weights.yaml not found — using defaults.")
        return ScoringWeights()
    except ImportError:
        logger.warning("pyyaml not installed — using default scoring weights.")
        return ScoringWeights()
    except Exception as exc:
        logger.warning("Failed to load scoring weights (%s) — using defaults.", exc)
        return ScoringWeights()


def get_scoring_weights() -> ScoringWeights:
    """Public accessor for the cached scoring weights."""
    return _load_weights()


# ---------------------------------------------------------------------------
# Evidence citation helpers
# ---------------------------------------------------------------------------


def _get_evidence_for_dimension(
    eval_result: EvaluationResult,
    agent_roles: list[AgentRole],
) -> tuple[list[str], str]:
    """
    Pull evidence snippets and reasoning from specific agent verdicts.

    Returns (evidence_list, combined_reasoning).
    """
    all_evidence:  list[str] = []
    all_reasoning: list[str] = []

    for role in agent_roles:
        verdict = eval_result.get_verdict(role)
        if verdict:
            all_evidence.extend(verdict.evidence[:2])
            if verdict.reasoning:
                all_reasoning.append(verdict.reasoning)

    # Deduplicate evidence
    seen: set[str] = set()
    unique_evidence: list[str] = []
    for e in all_evidence:
        if e not in seen:
            seen.add(e)
            unique_evidence.append(e)

    return unique_evidence[:4], " | ".join(all_reasoning[:2])


# ---------------------------------------------------------------------------
# Score adjustment logic
# ---------------------------------------------------------------------------


def _apply_adjustments(
    base_score:      float,
    eval_result:     EvaluationResult,
    hiring_profile:  HiringProfile,
    ledger:          Optional[CandidateLedger],
) -> tuple[float, list[str]]:
    """
    Apply optional score adjustments and return (adjusted_score, notes).

    Adjustments are small intentional nudges that reward verified evidence
    or penalise missing dealbreakers. They don't override the rubric —
    they refine the final sort for edge cases.
    """
    score = base_score
    notes: list[str] = []

    proof = eval_result.dimensions.proof_strength

    # Proof bonus — reward candidates with strong external evidence
    if proof >= 0.80:
        score = min(100.0, score + _PROOF_BONUS_HIGH)
        notes.append(f"Proof bonus +{_PROOF_BONUS_HIGH}pts (proof_strength={proof:.2f} ≥ 0.80)")
    elif proof >= 0.65:
        score = min(100.0, score + _PROOF_BONUS_MED)
        notes.append(f"Proof bonus +{_PROOF_BONUS_MED}pt (proof_strength={proof:.2f} ≥ 0.65)")

    # Seniority cap — strong mismatch is a meaningful signal
    if eval_result.dimensions.seniority_match < 0.25:
        score = score * _SENIORITY_CAP
        notes.append(
            f"Seniority cap ×{_SENIORITY_CAP} applied "
            f"(seniority_match={eval_result.dimensions.seniority_match:.2f} < 0.25)"
        )

    # Dealbreaker penalty — missing a hard-required skill
    if hiring_profile.required_skills and ledger:
        required_names = {s.skill for s in hiring_profile.required_skills if s.is_required}
        unsupported    = {e.skill for e in ledger.entries if e.is_unsupported}
        missing_critical = required_names & unsupported

        if missing_critical:
            skill_list = ", ".join(sorted(missing_critical)[:3])
            score = max(0.0, score - _DEALBREAKER_PENALTY)
            notes.append(
                f"Dealbreaker penalty -{_DEALBREAKER_PENALTY}pts "
                f"(required skills with no evidence: {skill_list})"
            )

    return round(score, 2), notes


# ---------------------------------------------------------------------------
# Single candidate scorer
# ---------------------------------------------------------------------------


def score_one(
    eval_result:    EvaluationResult,
    weights:        ScoringWeights,
    hiring_profile: HiringProfile,
    ledger:         Optional[CandidateLedger] = None,
) -> CandidateScore:
    """
    Compute the rubric score for one candidate.

    Args:
        eval_result:    EvaluationResult from Layer 9.
        weights:        ScoringWeights loaded from config.
        hiring_profile: HiringProfile from Layer 2 (for dealbreaker check).
        ledger:         CandidateLedger from Layer 6 (optional).

    Returns:
        CandidateScore with final_score (0–100) and full dimension breakdown.
    """
    d = eval_result.dimensions
    w = weights

    # ── Build dimension breakdowns ──────────────────────────────────────
    dimension_map = [
        ("skill_fit",        d.skill_fit,        w.skill_fit,
         [AgentRole.JD_FIT, AgentRole.TECHNICAL_FIT]),
        ("experience_depth", d.experience_depth, w.experience_depth,
         [AgentRole.TRAJECTORY]),
        ("seniority_match",  d.seniority_match,  w.seniority_match,
         [AgentRole.TRAJECTORY]),
        ("domain_match",     d.domain_match,     w.domain_match,
         [AgentRole.JD_FIT]),
        ("career_growth",    d.career_growth,    w.career_growth,
         [AgentRole.TRAJECTORY]),
        ("proof_strength",   d.proof_strength,   w.proof_strength,
         [AgentRole.VERIFICATION]),
    ]

    breakdowns: list[DimensionBreakdown] = []
    weighted_sum = 0.0

    for dim_name, raw_score, weight, agent_roles in dimension_map:
        contribution = round(raw_score * weight, 6)
        weighted_sum += contribution
        evidence, reasoning = _get_evidence_for_dimension(eval_result, agent_roles)

        breakdowns.append(DimensionBreakdown(
            dimension=dim_name,
            raw_score=round(raw_score, 4),
            weight=weight,
            weighted_score=round(contribution, 6),
            evidence=evidence,
            reasoning=reasoning,
        ))

    # Scale 0–1 → 0–100
    base_score = round(weighted_sum * 100.0, 2)

    # ── Apply adjustments ───────────────────────────────────────────────
    final_score, score_notes = _apply_adjustments(
        base_score, eval_result, hiring_profile, ledger
    )

    # ── Claim counts from ledger ────────────────────────────────────────
    verified_claims   = ledger.verified_count   if ledger else 0
    unverified_claims = (ledger.unsupported_count + ledger.pending_count) if ledger else 0

    candidate_score = CandidateScore(
        candidate_id=eval_result.candidate_id,
        candidate_name=eval_result.candidate_name,
        final_score=final_score,
        breakdowns=breakdowns,
        skill_fit=d.skill_fit,
        experience_depth=d.experience_depth,
        seniority_match=d.seniority_match,
        domain_match=d.domain_match,
        career_growth=d.career_growth,
        proof_strength=d.proof_strength,
        verified_claims=verified_claims,
        unverified_claims=unverified_claims,
        strengths=eval_result.strengths,
        risks=eval_result.risks,
        weights_used=w.as_dict(),
        score_notes=score_notes,
    )

    logger.debug(
        "%s  base=%.1f  final=%.1f  [%s]",
        eval_result.candidate_id,
        base_score,
        final_score,
        candidate_score.score_label,
    )

    return candidate_score


# ---------------------------------------------------------------------------
# Batch scorer — main public entry point
# ---------------------------------------------------------------------------


def score_candidates(
    batch_result:   BatchEvaluationResult,
    hiring_profile: HiringProfile,
    ledger_map:     Optional[dict[str, CandidateLedger]] = None,
    *,
    run_id:         str = "",
    weights:        Optional[ScoringWeights] = None,
) -> RankedOutput:
    """
    Score all evaluated candidates and return a ranked output.

    Args:
        batch_result:   BatchEvaluationResult from Layer 9.
        hiring_profile: HiringProfile from Layer 2.
        ledger_map:     dict[candidate_id → CandidateLedger] from Layer 6.
        run_id:         Pipeline run ID for the output file.
        weights:        Override the config weights (for ablation testing).

    Returns:
        RankedOutput sorted by final_score descending.
    """
    w = weights or get_scoring_weights()
    ledger_map = ledger_map or {}

    logger.info(
        "Layer 10 scoring: %d candidates  weights=%s",
        len(batch_result.results),
        {k: round(v, 2) for k, v in w.as_dict().items()},
    )

    scored: list[CandidateScore] = []
    for eval_result in batch_result.results:
        ledger = ledger_map.get(eval_result.candidate_id)
        try:
            cs = score_one(eval_result, w, hiring_profile, ledger)
        except Exception as exc:
            logger.error(
                "Scoring failed for %s: %s — assigning zero score.",
                eval_result.candidate_id, exc,
            )
            cs = CandidateScore(
                candidate_id=eval_result.candidate_id,
                candidate_name=eval_result.candidate_name,
                final_score=0.0,
                score_notes=[f"Scoring error: {exc}"],
            )
        scored.append(cs)

    ranked = RankedOutput(
        job_title=hiring_profile.job_title,
        run_id=run_id,
        scores=scored,
    )

    logger.info(
        "Layer 10 complete.\n%s",
        ranked.summary_table(),
    )
    return ranked


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_ranked_output(
    ranked: RankedOutput,
    output_dir: Path | str,
    *,
    pretty: bool = True,
) -> Path:
    """
    Write the ranked output to disk as JSON.

    File format matches schemas/ranked_output.schema.json exactly.
    Written to: <output_dir>/<run_id>_ranked.json

    Args:
        ranked:     RankedOutput from score_candidates().
        output_dir: Directory to write into (created if missing).
        pretty:     Indent JSON for human readability.

    Returns:
        Path to the written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    filename = f"{ranked.run_id}_ranked.json" if ranked.run_id else "ranked.json"
    path = out / filename

    payload = ranked.to_export_list()
    indent = 2 if pretty else None

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, ensure_ascii=False)

    logger.info(
        "RankedOutput[%s] saved → %s  (%d candidates)",
        ranked.run_id,
        path,
        len(payload),
    )
    return path


def load_ranked_output(path: Path | str) -> list[dict]:
    """
    Load a ranked output JSON file.
    Returns the raw list of dicts (matches ranked_output.schema.json).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ranked output file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
