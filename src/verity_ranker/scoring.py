from dataclasses import dataclass

from .candidates import CandidateProfile
from .jd import RoleProfile
from .verification import ClaimVerification


@dataclass(frozen=True)
class CandidateScore:
    rank: int
    candidate_id: str
    candidate_name: str
    score: float
    matched_required_skills: list[str]
    matched_preferred_skills: list[str]
    missing_required_skills: list[str]
    proof_strength: float
    evidence_summary: dict[str, list[str]]
    claim_verification: list[dict[str, str | float | list[str]]]


def _coverage(candidate_skills: set[str], target_skills: list[str]) -> float:
    if not target_skills:
        return 1.0
    return len(candidate_skills.intersection(target_skills)) / len(target_skills)


def _proof_adjusted_coverage(
    candidate_id: str,
    target_skills: list[str],
    verification_index: dict[str, dict[str, ClaimVerification]],
) -> float:
    if not target_skills:
        return 1.0
    candidate_verifications = verification_index.get(candidate_id, {})
    proof_total = sum(candidate_verifications.get(skill).proof_strength if skill in candidate_verifications else 0.0 for skill in target_skills)
    return proof_total / len(target_skills)


def _average_proof_strength(
    candidate_id: str,
    skills: list[str],
    verification_index: dict[str, dict[str, ClaimVerification]],
) -> float:
    if not skills:
        return 0.0
    candidate_verifications = verification_index.get(candidate_id, {})
    return sum(candidate_verifications.get(skill).proof_strength if skill in candidate_verifications else 0.0 for skill in skills) / len(skills)


def score_candidate(
    candidate: CandidateProfile,
    role: RoleProfile,
    weights: dict[str, float],
    verification_index: dict[str, dict[str, ClaimVerification]],
) -> CandidateScore:
    candidate_skills = set(candidate.skills)
    matched_required = sorted(candidate_skills.intersection(role.required_skills))
    matched_preferred = sorted(candidate_skills.intersection(role.preferred_skills))
    missing_required = sorted(set(role.required_skills) - candidate_skills)

    required_score = _proof_adjusted_coverage(candidate.candidate_id, role.required_skills, verification_index)
    preferred_score = _proof_adjusted_coverage(candidate.candidate_id, role.preferred_skills, verification_index)

    evidence_count = sum(len(candidate.evidence_snippets.get(skill, [])) for skill in matched_required)
    experience_depth = min(evidence_count / max(len(role.required_skills), 1), 1.0)
    proof_strength = _average_proof_strength(candidate.candidate_id, matched_required + matched_preferred, verification_index)

    total = (
        weights["required_skill_fit"] * required_score
        + weights["preferred_skill_fit"] * preferred_score
        + weights["experience_depth"] * experience_depth
        + weights["seniority_signal"] * candidate.seniority_signal
        + weights["achievement_signal"] * candidate.achievement_signal
        + weights["proof_strength"] * proof_strength
    )

    evidence_summary = {
        skill: candidate.evidence_snippets.get(skill, [])[:2]
        for skill in matched_required + matched_preferred
    }

    return CandidateScore(
        rank=0,
        candidate_id=candidate.candidate_id,
        candidate_name=candidate.name,
        score=round(total * 100, 2),
        matched_required_skills=matched_required,
        matched_preferred_skills=matched_preferred,
        missing_required_skills=missing_required,
        proof_strength=round(proof_strength, 3),
        evidence_summary=evidence_summary,
        claim_verification=[
            {
                "skill": verification.skill,
                "status": verification.status,
                "proof_strength": verification.proof_strength,
                "source": verification.evidence_source,
            }
            for verification in verification_index.get(candidate.candidate_id, {}).values()
        ],
    )


def rank_candidates(
    candidates: list[CandidateProfile],
    role: RoleProfile,
    weights: dict[str, float],
    verification_index: dict[str, dict[str, ClaimVerification]],
) -> list[CandidateScore]:
    scored = [score_candidate(candidate, role, weights, verification_index) for candidate in candidates]
    scored.sort(key=lambda item: (-item.score, item.candidate_id))
    return [
        CandidateScore(
            rank=index,
            candidate_id=item.candidate_id,
            candidate_name=item.candidate_name,
            score=item.score,
            matched_required_skills=item.matched_required_skills,
            matched_preferred_skills=item.matched_preferred_skills,
            missing_required_skills=item.missing_required_skills,
            proof_strength=item.proof_strength,
            evidence_summary=item.evidence_summary,
            claim_verification=item.claim_verification,
        )
        for index, item in enumerate(scored, start=1)
    ]
