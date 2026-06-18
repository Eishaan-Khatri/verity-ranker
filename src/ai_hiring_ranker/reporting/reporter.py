"""
Recruiter Audit Report Generator — Layer 12.

Assembles the complete recruiter-facing report from all upstream layers:

  Layer 9  (agents)     → strengths, risks, summary per candidate
  Layer 10 (scoring)    → final_score, dimension scores, breakdowns
  Layer 11 (reranking)  → final rank order, pairwise justifications,
                          rank confidence, score gaps
  Layer 6  (ledger)     → verified/unverified claim counts, skill lists

Two execution modes
-------------------
LLM mode
  Generates richer interview questions tailored to each candidate's
  specific gaps using the LLM. Also enriches "why above next" prose.
  Requires an API key.

Rules mode (fallback, no API key needed)
  Generates interview questions using a static template library keyed
  by skill category and verification status:
    - Unverified required skills → high-priority probing questions
    - Weak/inferred skills       → medium-priority questions
    - Seniority gaps             → career trajectory questions
    - Missing domain experience  → domain depth questions

Public API
----------
generate_report(rerank_result, eval_results, ledger_map,
                hiring_profile, run_id, force_fallback)
    → RecruiterReport

save_report(report, output_dir)
    → tuple[Path, Path]  (JSON path, Markdown path)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from ..agents.schemas import BatchEvaluationResult, EvaluationResult
from ..evidence.schemas import CandidateLedger
from ..jd_intelligence.schemas import HiringProfile
from ..reranking.schemas import RankConfidence, RerankEntry, RerankResult
from ..scoring.schemas import CandidateScore
from .schemas import CandidateCard, InterviewQuestion, RecruiterReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static interview question templates
# ---------------------------------------------------------------------------

_QUESTION_TEMPLATES: dict[str, list[dict]] = {
    # Unverified technical skill
    "unverified_skill": [
        {
            "template": "Can you walk me through a project where you used {skill} in production?",
            "priority": "high",
            "rationale": "This skill appears on the resume but has no external verification.",
        },
        {
            "template": "What was the most complex problem you solved using {skill}?",
            "priority": "high",
            "rationale": "Probes depth of knowledge for an unverified claim.",
        },
    ],
    # Weak/inferred skill
    "weak_skill": [
        {
            "template": "How extensively have you used {skill}, and in what context?",
            "priority": "medium",
            "rationale": "Evidence for this skill is indirect or limited.",
        },
    ],
    # Missing required skill
    "missing_required": [
        {
            "template": (
                "The role requires {skill}. While we didn't see direct evidence, "
                "do you have experience here? Can you give a specific example?"
            ),
            "priority": "high",
            "rationale": "Required skill has no evidence in the profile.",
        },
    ],
    # Seniority gap
    "seniority_gap": [
        {
            "template": (
                "This role is at the {jd_seniority} level. "
                "Can you describe a situation where you operated at that level of ownership?"
            ),
            "priority": "medium",
            "rationale": "Candidate seniority signal is lower than the JD requirement.",
        },
    ],
    # Career gap
    "career_gap": [
        {
            "template": "Can you tell me about the period between your roles at {company_a} and {company_b}?",
            "priority": "medium",
            "rationale": "Career timeline has a gap that may need clarification.",
        },
    ],
    # Low proof strength
    "low_proof": [
        {
            "template": (
                "Your profile is strong on paper. Can you share a GitHub link or "
                "portfolio piece that demonstrates your most recent work?"
            ),
            "priority": "high",
            "rationale": "Overall proof strength is low — most claims are unverified.",
        },
    ],
    # Generic technical depth
    "technical_depth": [
        {
            "template": "What's the largest-scale system you've built or maintained using {skill}?",
            "priority": "medium",
            "rationale": "Probes production-scale experience.",
        },
        {
            "template": "How do you stay current with {skill}? Any recent work or projects?",
            "priority": "low",
            "rationale": "Checks recency and continued engagement.",
        },
    ],
}


def _generate_questions_rules(
    card_score: CandidateScore,
    ledger: Optional[CandidateLedger],
    hiring_profile: HiringProfile,
) -> list[InterviewQuestion]:
    """Generate interview questions using static templates."""
    questions: list[InterviewQuestion] = []

    # Unverified required skills — highest priority
    required_set = {s.strip().title() for s in hiring_profile.all_required_skill_names}

    if ledger:
        unverified_required = [
            e.skill for e in ledger.entries
            if e.skill in required_set and e.is_unsupported
        ]
        for skill in unverified_required[:3]:
            for tmpl in _QUESTION_TEMPLATES["missing_required"]:
                questions.append(InterviewQuestion(
                    question=tmpl["template"].format(skill=skill),
                    skill=skill,
                    priority=tmpl["priority"],
                    rationale=tmpl["rationale"],
                ))

        # Weak/inferred skills
        weak_skills = [
            e.skill for e in ledger.entries
            if e.verification_status.value in ("weak", "inferred")
            and e.skill in required_set
        ][:3]
        for skill in weak_skills:
            for tmpl in _QUESTION_TEMPLATES["weak_skill"]:
                questions.append(InterviewQuestion(
                    question=tmpl["template"].format(skill=skill),
                    skill=skill,
                    priority=tmpl["priority"],
                    rationale=tmpl["rationale"],
                ))

        # Low proof strength
        if ledger.proof_strength < 0.35:
            tmpl = _QUESTION_TEMPLATES["low_proof"][0]
            questions.append(InterviewQuestion(
                question=tmpl["template"],
                skill="general",
                priority=tmpl["priority"],
                rationale=tmpl["rationale"],
            ))

    # Seniority mismatch
    if card_score.seniority_match < 0.45:
        tmpl = _QUESTION_TEMPLATES["seniority_gap"][0]
        questions.append(InterviewQuestion(
            question=tmpl["template"].format(
                jd_seniority=hiring_profile.seniority.value
            ),
            skill="seniority",
            priority=tmpl["priority"],
            rationale=tmpl["rationale"],
        ))

    # Top required skills — always add depth probes for top 2
    for skill in hiring_profile.all_required_skill_names[:2]:
        tmpl = _QUESTION_TEMPLATES["technical_depth"][0]
        questions.append(InterviewQuestion(
            question=tmpl["template"].format(skill=skill),
            skill=skill,
            priority="medium",
            rationale=tmpl["rationale"],
        ))

    # Deduplicate by question text
    seen: set[str] = set()
    unique: list[InterviewQuestion] = []
    for q in questions:
        if q.question not in seen:
            seen.add(q.question)
            unique.append(q)

    # Sort: high → medium → low
    priority_order = {"high": 0, "medium": 1, "low": 2}
    unique.sort(key=lambda q: priority_order.get(q.priority, 1))

    return unique[:6]


def _generate_questions_llm(
    card_score: CandidateScore,
    ledger: Optional[CandidateLedger],
    hiring_profile: HiringProfile,
    eval_result: Optional[EvaluationResult],
) -> list[InterviewQuestion]:
    """Use LLM to generate tailored interview questions."""
    from ..llm_provider import chat_completion

    unverified = (
        ", ".join(ledger.unsupported_skills[:5]) if ledger else "unknown"
    )
    risks = "; ".join(card_score.risks[:3])
    strengths = "; ".join(card_score.strengths[:3])

    system_prompt = (
        "You are a technical recruiter. Generate 5–6 targeted interview questions "
        "for a candidate based on their profile gaps and unverified claims. "
        "Return ONLY a JSON array of objects with keys: "
        "question (string), skill (string), priority (high/medium/low), rationale (string)."
    )
    user_prompt = f"""Job: {hiring_profile.job_title} ({hiring_profile.seniority.value})
Required skills: {', '.join(hiring_profile.all_required_skill_names[:8])}

Candidate: {card_score.candidate_name or card_score.candidate_id}
Score: {card_score.final_score:.1f}/100
Strengths: {strengths or 'none listed'}
Risks: {risks or 'none listed'}
Unverified required skills: {unverified}
Proof strength: {card_score.proof_strength:.2f}

Generate targeted interview questions to verify the weakest claims."""

    raw = chat_completion(system_prompt, user_prompt)
    data = json.loads(raw.strip().strip("```json").strip("```"))
    if not isinstance(data, list):
        data = data.get("questions", [])

    return [
        InterviewQuestion(
            question=str(q.get("question", "")),
            skill=str(q.get("skill", "")),
            priority=str(q.get("priority", "medium")),
            rationale=str(q.get("rationale", "")),
        )
        for q in data[:6]
        if q.get("question")
    ]


# ---------------------------------------------------------------------------
# "Why above next" prose builder
# ---------------------------------------------------------------------------


def _build_why_above_next(
    entry: RerankEntry,
    next_entry: Optional[RerankEntry],
    rerank_result: RerankResult,
) -> tuple[str, Optional[float]]:
    """
    Build the "why ranked above the next candidate" text.
    Returns (prose_text, score_gap).
    """
    if next_entry is None:
        return "This is the lowest-ranked candidate in the shortlist.", None

    score_gap = round(entry.final_score - next_entry.final_score, 2)

    # Try to get a pairwise justification from Layer 11
    justification = rerank_result.get_justification(
        entry.candidate_id, next_entry.candidate_id
    )

    if justification and justification.deciding_factors:
        factors_text = "; ".join(justification.deciding_factors[:3])
        uncertainty  = f" Uncertainty: {justification.uncertainty}" if justification.uncertainty else ""
        return (
            f"Ranked above {next_entry.candidate_id} "
            f"(score gap: {score_gap:+.1f} pts). "
            f"Key factors: {factors_text}.{uncertainty}"
        ), score_gap

    # Fallback: compare dimension scores
    a = entry.candidate_score
    b = next_entry.candidate_score
    dim_diffs = {
        "skill fit":        a.skill_fit        - b.skill_fit,
        "proof strength":   a.proof_strength   - b.proof_strength,
        "seniority match":  a.seniority_match  - b.seniority_match,
        "career growth":    a.career_growth    - b.career_growth,
        "experience depth": a.experience_depth - b.experience_depth,
    }
    top_diff = max(dim_diffs.items(), key=lambda x: x[1])

    if top_diff[1] >= 0.05:
        return (
            f"Ranked above {next_entry.candidate_id} "
            f"(score gap: {score_gap:+.1f} pts). "
            f"Primary differentiator: higher {top_diff[0]} "
            f"({getattr(a, top_diff[0].replace(' ', '_')):.2f} vs "
            f"{getattr(b, top_diff[0].replace(' ', '_')):.2f})."
        ), score_gap

    return (
        f"Marginally ranked above {next_entry.candidate_id} "
        f"(score gap: {score_gap:+.1f} pts). "
        f"Scores are very close — rank confidence is low."
    ), score_gap


# ---------------------------------------------------------------------------
# CandidateCard builder
# ---------------------------------------------------------------------------


def _build_card(
    entry:          RerankEntry,
    next_entry:     Optional[RerankEntry],
    rerank_result:  RerankResult,
    eval_result:    Optional[EvaluationResult],
    ledger:         Optional[CandidateLedger],
    hiring_profile: HiringProfile,
    force_fallback: bool,
) -> CandidateCard:
    """Assemble a CandidateCard from all upstream layer outputs."""
    score = entry.candidate_score

    why_above, score_gap = _build_why_above_next(entry, next_entry, rerank_result)

    # Interview questions
    has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY"))
    if not force_fallback and has_key:
        try:
            questions = _generate_questions_llm(score, ledger, hiring_profile, eval_result)
        except Exception as exc:
            logger.warning("LLM questions failed for %s (%s) — using templates.", score.candidate_id, exc)
            questions = _generate_questions_rules(score, ledger, hiring_profile)
    else:
        questions = _generate_questions_rules(score, ledger, hiring_profile)

    verified_skills   = ledger.verified_skills[:10]   if ledger else []
    unverified_skills = ledger.unsupported_skills[:6] if ledger else []

    return CandidateCard(
        rank=entry.reranked_rank,
        candidate_id=score.candidate_id,
        candidate_name=score.candidate_name,
        final_score=score.final_score,
        score_label=score.score_label,
        skill_fit=score.skill_fit,
        experience_depth=score.experience_depth,
        seniority_match=score.seniority_match,
        domain_match=score.domain_match,
        career_growth=score.career_growth,
        proof_strength=score.proof_strength,
        verified_claims=score.verified_claims,
        unverified_claims=score.unverified_claims,
        verified_skills=verified_skills,
        unverified_skills=unverified_skills,
        strengths=eval_result.strengths if eval_result else score.strengths,
        risks=eval_result.risks         if eval_result else score.risks,
        summary=eval_result.summary     if eval_result else "",
        why_above_next=why_above,
        score_gap_to_next=score_gap,
        rank_confidence=entry.rank_confidence.value,
        interview_questions=questions,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_report(
    rerank_result:  RerankResult,
    hiring_profile: HiringProfile,
    eval_results:   Optional[BatchEvaluationResult] = None,
    ledger_map:     Optional[dict[str, CandidateLedger]] = None,
    *,
    run_id:         str = "",
    force_fallback: bool = False,
) -> RecruiterReport:
    """
    Generate the complete recruiter audit report.

    Args:
        rerank_result:  RerankResult from Layer 11.
        hiring_profile: HiringProfile from Layer 2.
        eval_results:   BatchEvaluationResult from Layer 9 (optional, enriches cards).
        ledger_map:     dict[candidate_id → CandidateLedger] from Layer 6.
        run_id:         Pipeline run ID.
        force_fallback: Use template-based question generation only.

    Returns:
        RecruiterReport with CandidateCards, stability warnings, and run notes.
    """
    ledger_map  = ledger_map or {}
    eval_map    = {r.candidate_id: r for r in eval_results.results} if eval_results else {}
    entries     = rerank_result.entries

    logger.info(
        "Layer 12: generating report for %d candidates (run_id=%s)",
        len(entries), run_id,
    )

    cards: list[CandidateCard] = []
    for i, entry in enumerate(entries):
        next_entry  = entries[i + 1] if i + 1 < len(entries) else None
        eval_result = eval_map.get(entry.candidate_id)
        ledger      = ledger_map.get(entry.candidate_id)

        try:
            card = _build_card(
                entry, next_entry, rerank_result,
                eval_result, ledger, hiring_profile, force_fallback,
            )
        except Exception as exc:
            logger.error("Card generation failed for %s: %s", entry.candidate_id, exc)
            card = CandidateCard(
                rank=entry.reranked_rank,
                candidate_id=entry.candidate_id,
                candidate_name=entry.candidate_name,
                final_score=entry.final_score,
                score_label=entry.candidate_score.score_label,
                summary=f"Report generation failed: {exc}",
            )
        cards.append(card)

    # Stability warnings
    warnings: list[str] = []
    for entry in entries:
        if entry.rank_confidence in (RankConfidence.LOW, RankConfidence.UNSTABLE):
            warnings.append(
                f"#{entry.reranked_rank} {entry.candidate_id}: "
                f"rank confidence is {entry.rank_confidence.value} "
                f"(score gap < 5 pts or rank shifted {abs(entry.rank_delta)} positions). "
                f"Consider re-evaluating manually."
            )

    # Run notes
    run_notes: list[str] = []
    if rerank_result.rerank_method == "rules":
        run_notes.append("Listwise re-ranking used rule-based fallback (no LLM API key).")
    if rerank_result.unstable_count:
        run_notes.append(
            f"{rerank_result.unstable_count} candidate(s) have low/unstable rank confidence."
        )

    report = RecruiterReport(
        run_id=run_id,
        job_title=hiring_profile.job_title,
        total_evaluated=len(entries),
        cards=cards,
        unstable_rank_warnings=warnings,
        run_notes=run_notes,
    )

    logger.info("Layer 12 complete: %d candidate cards generated.", len(cards))
    return report


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_report(
    report:     RecruiterReport,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """
    Save the report as both JSON and Markdown.

    Files written:
      <output_dir>/<run_id>_report.json
      <output_dir>/<run_id>_report.md

    Returns:
        (json_path, md_path)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prefix = report.run_id if report.run_id else "report"

    json_path = out / f"{prefix}_report.json"
    md_path   = out / f"{prefix}_report.md"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report.to_export_dict(), fh, indent=2, ensure_ascii=False)

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(report.to_markdown())

    logger.info(
        "RecruiterReport saved → JSON: %s  | MD: %s",
        json_path, md_path,
    )
    return json_path, md_path
