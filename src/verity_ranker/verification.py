from dataclasses import dataclass

from .candidates import CandidateProfile
from .jd import RoleProfile


STRONG_ACTION_MARKERS = [
    "built",
    "created",
    "deployed",
    "added tests",
    "converted",
    "implemented",
    "pipeline",
    "api",
    "docker-based",
    "docker images",
    "ci pipelines",
]


@dataclass(frozen=True)
class ClaimVerification:
    candidate_id: str
    candidate_name: str
    skill: str
    claim: str
    status: str
    proof_strength: float
    confidence: float
    evidence_source: str
    evidence_snippets: list[str]
    score_dimension: str


@dataclass(frozen=True)
class EvidenceLedgerEntry:
    evidence_id: str
    candidate_id: str
    candidate_name: str
    skill: str
    claim: str
    status: str
    proof_strength: float
    confidence: float
    source: str
    evidence_snippet: str
    score_dimension: str


def _has_strong_action(snippets: list[str]) -> bool:
    joined = " ".join(snippets).lower()
    return any(marker in joined for marker in STRONG_ACTION_MARKERS)


def _dimension_for_skill(skill: str, role: RoleProfile) -> str:
    if skill in role.required_skills:
        return "required_skill_fit"
    if skill in role.preferred_skills:
        return "preferred_skill_fit"
    return "candidate_claim"


def verify_candidate_claims(candidate: CandidateProfile, role: RoleProfile) -> list[ClaimVerification]:
    target_skills = sorted(
        set(role.required_skills)
        | set(role.preferred_skills)
        | set(candidate.declared_skills)
        | set(candidate.body_skills)
        | set(candidate.negated_skills)
    )
    verifications: list[ClaimVerification] = []

    for skill in target_skills:
        positive = candidate.evidence_snippets.get(skill, [])
        negative = candidate.negated_snippets.get(skill, [])
        declared = skill in candidate.declared_skills
        dimension = _dimension_for_skill(skill, role)

        if negative:
            status = "unsupported"
            proof_strength = 0.0
            confidence = 0.95
            source = "negated_resume_text"
            snippets = negative[:2]
        elif positive and _has_strong_action(positive):
            status = "verified"
            proof_strength = 1.0
            confidence = 0.85
            source = "resume_body_evidence"
            snippets = positive[:2]
        elif positive:
            status = "weakly_supported"
            proof_strength = 0.65
            confidence = 0.70
            source = "resume_body_evidence"
            snippets = positive[:2]
        elif declared:
            status = "weakly_supported"
            proof_strength = 0.15
            confidence = 0.55
            source = "skills_line_only"
            snippets = []
        else:
            status = "unverifiable"
            proof_strength = 0.0
            confidence = 0.40
            source = "no_candidate_evidence"
            snippets = []

        verifications.append(
            ClaimVerification(
                candidate_id=candidate.candidate_id,
                candidate_name=candidate.name,
                skill=skill,
                claim=f"Candidate claims or is evaluated for {skill}.",
                status=status,
                proof_strength=proof_strength,
                confidence=confidence,
                evidence_source=source,
                evidence_snippets=snippets,
                score_dimension=dimension,
            )
        )

    return verifications


def build_claim_verifications(candidates: list[CandidateProfile], role: RoleProfile) -> list[ClaimVerification]:
    verifications: list[ClaimVerification] = []
    for candidate in candidates:
        verifications.extend(verify_candidate_claims(candidate, role))
    return verifications


def build_verification_index(verifications: list[ClaimVerification]) -> dict[str, dict[str, ClaimVerification]]:
    index: dict[str, dict[str, ClaimVerification]] = {}
    for verification in verifications:
        index.setdefault(verification.candidate_id, {})[verification.skill] = verification
    return index


def build_evidence_ledger(verifications: list[ClaimVerification]) -> list[EvidenceLedgerEntry]:
    ledger: list[EvidenceLedgerEntry] = []
    counters: dict[str, int] = {}

    for verification in verifications:
        snippets = verification.evidence_snippets or [""]
        for snippet in snippets:
            counters[verification.candidate_id] = counters.get(verification.candidate_id, 0) + 1
            evidence_id = f"{verification.candidate_id}-E{counters[verification.candidate_id]:03d}"
            ledger.append(
                EvidenceLedgerEntry(
                    evidence_id=evidence_id,
                    candidate_id=verification.candidate_id,
                    candidate_name=verification.candidate_name,
                    skill=verification.skill,
                    claim=verification.claim,
                    status=verification.status,
                    proof_strength=verification.proof_strength,
                    confidence=verification.confidence,
                    source=verification.evidence_source,
                    evidence_snippet=snippet,
                    score_dimension=verification.score_dimension,
                )
            )

    return ledger
