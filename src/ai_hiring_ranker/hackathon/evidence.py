"""Build hackathon evidence ledgers without live GitHub crawling."""

from __future__ import annotations

from typing import Optional

from ..candidate_extraction.schemas import CandidateProfile, SkillConfidence
from ..evidence.ledger import make_claim_id
from ..evidence.schemas import CandidateLedger, ClaimSource, LedgerEntry
from ..ingestion.schemas import VerificationStatus


def _status_for_skill(
    skill_name: str,
    *,
    confidence: SkillConfidence,
    has_snippet: bool,
    github_activity_score: float,
) -> VerificationStatus:
    """Map resume skill signals + platform score to verification status."""
    if confidence == SkillConfidence.EXPLICIT and has_snippet:
        return VerificationStatus.VERIFIED

    if confidence == SkillConfidence.EXPLICIT:
        if github_activity_score >= 0.65:
            return VerificationStatus.VERIFIED
        if github_activity_score >= 0.35:
            return VerificationStatus.WEAK
        return VerificationStatus.INFERRED

    if github_activity_score >= 0.50:
        return VerificationStatus.WEAK
    return VerificationStatus.INFERRED


def build_hackathon_ledger(
    profile: CandidateProfile,
    *,
    github_activity_score: float = 0.0,
    run_id: str = "precompute",
) -> CandidateLedger:
    """
    Create a lightweight evidence ledger from resume extraction only.

    Uses ``github_activity_score`` from the dataset as platform proof —
    no GitHub API calls.
    """
    entries: list[LedgerEntry] = []

    for skill in profile.skills:
        snippet = skill.evidence_snippets[0] if skill.evidence_snippets else ""
        has_snippet = bool(snippet.strip())
        status = _status_for_skill(
            skill.skill,
            confidence=skill.confidence,
            has_snippet=has_snippet,
            github_activity_score=github_activity_score,
        )
        source = ClaimSource.GITHUB if (
            status == VerificationStatus.VERIFIED and github_activity_score >= 0.50
        ) else ClaimSource.RESUME

        claim_text = f"Claims proficiency in {skill.skill}"
        entries.append(
            LedgerEntry(
                claim_id=make_claim_id(profile.candidate_id, skill.skill, claim_text),
                candidate_id=profile.candidate_id,
                skill=skill.skill,
                claim_text=claim_text,
                source=source,
                verification_status=status,
                confidence=min(
                    1.0,
                    0.55 + (0.25 if has_snippet else 0.0) + github_activity_score * 0.20,
                ),
                evidence_snippet=snippet[:200],
                reasoning=(
                    f"Resume extraction ({skill.confidence.value}); "
                    f"github_activity_score={github_activity_score:.2f}"
                ),
            )
        )

    if github_activity_score >= 0.40 and not entries:
        entries.append(
            LedgerEntry(
                claim_id=make_claim_id(
                    profile.candidate_id,
                    "Platform Activity",
                    "Active on GitHub",
                ),
                candidate_id=profile.candidate_id,
                skill="Platform Activity",
                claim_text="Demonstrates GitHub platform activity",
                source=ClaimSource.GITHUB,
                verification_status=VerificationStatus.WEAK,
                confidence=github_activity_score,
                evidence_snippet=f"github_activity_score={github_activity_score:.2f}",
                reasoning="Platform activity score from dataset (no live crawl).",
            )
        )

    return CandidateLedger(
        candidate_id=profile.candidate_id,
        candidate_name=profile.name or profile.candidate_id,
        entries=entries,
        run_id=run_id,
    )
