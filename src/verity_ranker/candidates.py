from dataclasses import dataclass
from pathlib import Path
import re

from .skills import extract_skills
from .text import split_sentences


NEGATION_MARKERS = [
    "no direct experience",
    "no experience",
    "limited direct",
    "limited deployment",
    "without",
    "not experienced",
]


@dataclass(frozen=True)
class CandidateProfile:
    candidate_id: str
    name: str
    raw_text: str
    declared_skills: list[str]
    body_skills: list[str]
    negated_skills: list[str]
    skills: list[str]
    evidence_snippets: dict[str, list[str]]
    negated_snippets: dict[str, list[str]]
    seniority_signal: float
    achievement_signal: float


def _field(text: str, field_name: str) -> str | None:
    match = re.search(rf"^{re.escape(field_name)}:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _is_negated_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(marker in lowered for marker in NEGATION_MARKERS)


def parse_candidate(path: Path) -> CandidateProfile:
    text = path.read_text(encoding="utf-8")
    candidate_id = _field(text, "Candidate ID") or path.stem
    name = _field(text, "Name") or candidate_id
    body_text = "\n".join(
        line
        for line in text.splitlines()
        if not re.match(r"^\s*(Candidate ID|Name|Skills):", line, re.IGNORECASE)
    )

    skills = set()
    declared_skills = set()
    skills_line = _field(text, "Skills")
    if skills_line:
        declared_skills.update(extract_skills(skills_line))
        skills.update(declared_skills)

    positive_sentences = [sentence for sentence in split_sentences(body_text) if not _is_negated_sentence(sentence)]
    negative_sentences = [sentence for sentence in split_sentences(body_text) if _is_negated_sentence(sentence)]
    body_skills = set()
    negated_skills = set()

    for sentence in positive_sentences:
        sentence_skills = extract_skills(sentence)
        body_skills.update(sentence_skills)
        skills.update(sentence_skills)

    snippets: dict[str, list[str]] = {skill: [] for skill in sorted(skills)}
    for sentence in positive_sentences:
        sentence_skills = extract_skills(sentence)
        for skill in sentence_skills:
            snippets.setdefault(skill, []).append(sentence)

    negated_snippets: dict[str, list[str]] = {}
    for sentence in negative_sentences:
        sentence_skills = extract_skills(sentence)
        negated_skills.update(sentence_skills)
        for skill in sentence_skills:
            negated_snippets.setdefault(skill, []).append(sentence)

    skills -= negated_skills

    lowered = text.lower()
    seniority_signal = 1.0 if any(term in lowered for term in ["lead", "senior", "3 years", "4 years", "5 years"]) else 0.45
    achievement_signal = 1.0 if any(term in lowered for term in ["built", "created", "deployed", "added tests", "converted"]) else 0.35

    return CandidateProfile(
        candidate_id=candidate_id,
        name=name,
        raw_text=text,
        declared_skills=sorted(declared_skills),
        body_skills=sorted(body_skills),
        negated_skills=sorted(negated_skills),
        skills=sorted(skills),
        evidence_snippets=snippets,
        negated_snippets=negated_snippets,
        seniority_signal=seniority_signal,
        achievement_signal=achievement_signal,
    )


def load_candidates(directory: Path) -> list[CandidateProfile]:
    files = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".txt", ".md"})
    if not files:
        raise ValueError(f"No .txt or .md candidate files found in {directory}")
    return [parse_candidate(path) for path in files]
