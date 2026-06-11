"""
JD Intelligence Agent — Layer 2.

Takes a raw JDInput and returns a validated HiringProfile.

Two execution modes:
  1. LLM mode  — calls the configured LLM with a structured prompt.
                 Requires OPENAI_API_KEY (or configured provider key).
  2. Fallback  — pure rule-based extraction using regex + keyword lists.
                 Used when no API key is present or during offline testing.

The fallback is intentionally conservative and less capable than the LLM
mode, but it ensures the pipeline always produces *something* rather than
hard-crashing when a key is missing.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from ..ingestion.schemas import JDInput
from ..llm_provider import structured_completion
from .schemas import (
    AmbiguityFlag,
    EmploymentType,
    HiddenExpectation,
    HiringProfile,
    SeniorityLevel,
    SkillEntry,
)

logger = logging.getLogger(__name__)

# Path to the prompt file
_PROMPT_PATH = Path(__file__).resolve().parents[4] / "prompts" / "jd_intelligence.md"


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------


def _load_system_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    # Inline fallback if the file is somehow missing
    return (
        "You are a technical recruiter. Extract a structured JSON hiring profile "
        "from the job description. Return ONLY valid JSON."
    )


# ---------------------------------------------------------------------------
# Rule-based fallback (no LLM required)
# ---------------------------------------------------------------------------

# Seniority keyword map — ordered most-specific first
_SENIORITY_MAP: list[tuple[re.Pattern, SeniorityLevel]] = [
    (re.compile(r"\bprincipal\b", re.I), SeniorityLevel.PRINCIPAL),
    (re.compile(r"\bstaff\b", re.I), SeniorityLevel.STAFF),
    (re.compile(r"\blead\b", re.I), SeniorityLevel.LEAD),
    (re.compile(r"\bsenior\b|\bsr\b", re.I), SeniorityLevel.SENIOR),
    (re.compile(r"\bjunior\b|\bjr\b", re.I), SeniorityLevel.JUNIOR),
    (re.compile(r"\bintern\b", re.I), SeniorityLevel.INTERN),
    (re.compile(r"\bmanager\b", re.I), SeniorityLevel.MANAGER),
    (re.compile(r"\bmid[\s\-]?level\b|\bmedior\b", re.I), SeniorityLevel.MID),
    (re.compile(r"\b[3-5]\+?\s*years?\b", re.I), SeniorityLevel.MID),
    (re.compile(r"\b[5-9]\+?\s*years?\b", re.I), SeniorityLevel.SENIOR),
]

_EMPLOYMENT_MAP: list[tuple[re.Pattern, EmploymentType]] = [
    (re.compile(r"\bfull[\s\-]?time\b", re.I), EmploymentType.FULL_TIME),
    (re.compile(r"\bpart[\s\-]?time\b", re.I), EmploymentType.PART_TIME),
    (re.compile(r"\bcontract\b|\bcontractor\b", re.I), EmploymentType.CONTRACT),
    (re.compile(r"\bfreelance\b", re.I), EmploymentType.FREELANCE),
    (re.compile(r"\binternship\b|\bintern\b", re.I), EmploymentType.INTERNSHIP),
]

_AMBIGUITY_MARKERS = [
    (r"\betc\.?\b", "Trailing 'etc.' leaves requirements open-ended."),
    (r"\band more\b", "'And more' is unspecified — what exactly?"),
    (r"\bstrong background\b", "'Strong background' is subjective — no concrete bar set."),
    (r"\bfamiliar(?:ity)? with\b", "'Familiar with' is ambiguous — no depth specified."),
    (r"\bexperience (?:with|in)\b(?! \d)", "Experience level is not quantified."),
    (r"\bknowledge of\b", "'Knowledge of' does not specify depth or recency."),
    (r"\bgood understanding\b", "'Good understanding' is vague."),
]

_PREFERRED_MARKERS = ["preferred", "nice to have", "bonus", "plus", "ideally", "desirable", "a plus"]
_RESPONSIBILITY_MARKERS = [
    "build", "develop", "design", "deploy", "maintain", "own", "lead",
    "collaborate", "evaluate", "implement", "create", "manage", "work with",
    "drive", "improve", "support", "write", "test", "monitor",
]

# Common skills to look for — extended list for robust rule-based extraction
_SKILL_ALIASES: dict[str, list[str]] = {
    "Python": ["python"],
    "Machine Learning": ["machine learning", " ml ", "ml,", "ml."],
    "Deep Learning": ["deep learning", "neural network", "pytorch", "tensorflow"],
    "FastAPI": ["fastapi", "fast api"],
    "Docker": ["docker", "containerization", "containers"],
    "Kubernetes": ["kubernetes", "k8s"],
    "SQL": ["sql", "postgres", "mysql", "sqlite"],
    "NoSQL": ["nosql", "mongodb", "redis", "dynamodb"],
    "REST API": ["rest api", "restful", "api design"],
    "Cloud": ["aws", "gcp", "azure", "cloud deployment"],
    "CI/CD": ["ci/cd", "ci pipeline", "github actions", "jenkins"],
    "Model Evaluation": ["model evaluation", "evaluation pipeline", "metrics", "f1", "accuracy", "auc"],
    "Embeddings": ["embedding", "vector"],
    "Retrieval": ["retrieval", "vector search", "faiss", "bm25"],
    "LLM": ["llm", "large language model", "gpt", "language model"],
    "LangChain": ["langchain", "langgraph"],
    "Production Engineering": ["production", "production-ready", "production system", "reliable system"],
    "Testing": ["unit test", "pytest", "testing", "test coverage"],
    "Git": ["git", "github", "version control"],
    "Spark": ["spark", "pyspark", "distributed"],
    "Airflow": ["airflow", "workflow"],
    "MLflow": ["mlflow", "experiment tracking"],
    "NLP": ["nlp", "natural language processing", "text classification", "named entity"],
    "Computer Vision": ["computer vision", "cv", "image classification", "object detection"],
    "Statistics": ["statistics", "statistical", "probability", "hypothesis testing"],
    "Data Engineering": ["data pipeline", "etl", "data engineering"],
    "Communication": ["communicate", "stakeholder", "collaboration"],
}


def _extract_skills_fallback(text: str) -> tuple[list[SkillEntry], list[SkillEntry]]:
    """Return (required_skills, preferred_skills) using keyword matching."""
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    required: list[SkillEntry] = []
    preferred: list[SkillEntry] = []
    seen: set[str] = set()

    for sentence in sentences:
        lowered = sentence.lower()
        is_preferred_sentence = any(marker in lowered for marker in _PREFERRED_MARKERS)

        for skill_name, aliases in _SKILL_ALIASES.items():
            if skill_name in seen:
                continue
            if any(alias in lowered for alias in aliases):
                seen.add(skill_name)
                entry = SkillEntry(
                    skill=skill_name,
                    is_required=not is_preferred_sentence,
                    is_preferred=is_preferred_sentence,
                    context_snippet=sentence.strip()[:200],
                )
                if is_preferred_sentence:
                    preferred.append(entry)
                else:
                    required.append(entry)

    return required, preferred


def _extract_responsibilities_fallback(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    result = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(marker in lowered for marker in _RESPONSIBILITY_MARKERS):
            clean = sentence.strip()
            if 10 < len(clean) < 300:
                result.append(clean)
    return result[:10]  # cap at 10


def _detect_seniority(text: str) -> SeniorityLevel:
    for pattern, level in _SENIORITY_MAP:
        if pattern.search(text):
            return level
    return SeniorityLevel.UNKNOWN


def _detect_employment_type(text: str) -> EmploymentType:
    for pattern, emp_type in _EMPLOYMENT_MAP:
        if pattern.search(text):
            return emp_type
    return EmploymentType.UNKNOWN


def _detect_years(text: str) -> Optional[int]:
    match = re.search(r"(\d+)\+?\s*years?", text, re.I)
    return int(match.group(1)) if match else None


def _detect_ambiguity(text: str) -> list[AmbiguityFlag]:
    flags = []
    for pattern, reason in _AMBIGUITY_MARKERS:
        match = re.search(pattern, text, re.I)
        if match:
            flags.append(AmbiguityFlag(phrase=match.group(0), reason=reason))
    return flags


def _detect_hidden_expectations(text: str, required: list[SkillEntry]) -> list[HiddenExpectation]:
    """Infer hidden expectations from responsibility language."""
    hidden = []
    lowered = text.lower()

    rules = [
        (
            r"production|deploy|reliable|scalable",
            "Candidate must understand production engineering practices (monitoring, alerting, scaling).",
            "production/deployment language in JD",
        ),
        (
            r"build.*api|api.*build|serve.*model",
            "Candidate is expected to know API security, rate limiting, and error handling.",
            "API development responsibility",
        ),
        (
            r"team|collaborat|cross.functional|stakeholder",
            "Candidate needs strong written and verbal communication skills.",
            "team/collaboration language in JD",
        ),
        (
            r"notebook|jupyter|experiment",
            "Candidate is expected to move beyond exploratory notebooks to production code.",
            "'notebook' or 'experiment' language in JD",
        ),
        (
            r"lead|own|drive|architect",
            "Candidate may need to mentor junior engineers and make technical decisions independently.",
            "ownership/leadership language in JD",
        ),
    ]
    for pattern, description, inferred_from in rules:
        if re.search(pattern, lowered):
            hidden.append(
                HiddenExpectation(
                    description=description,
                    inferred_from=inferred_from,
                    confidence=0.75,
                )
            )

    return hidden


def _extract_job_title(text: str) -> str:
    """Try first non-empty line as job title."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) < 100:
            return stripped
    return "Unknown Role"


def _extract_domain(text: str) -> Optional[str]:
    domain_map = {
        "Machine Learning": ["machine learning", "ml engineer", "mlops"],
        "Data Engineering": ["data engineer", "data pipeline", "etl"],
        "Backend Engineering": ["backend", "server-side", "api engineer"],
        "NLP": ["natural language processing", "nlp engineer", "text"],
        "Computer Vision": ["computer vision", "image", "object detection"],
        "FinTech": ["fintech", "finance", "banking", "payments"],
        "Healthcare": ["health", "medical", "clinical", "pharma"],
        "Security": ["security", "infosec", "penetration testing"],
        "DevOps": ["devops", "platform engineer", "sre", "site reliability"],
    }
    lowered = text.lower()
    for domain, keywords in domain_map.items():
        if any(kw in lowered for kw in keywords):
            return domain
    return None


def _run_fallback(jd: JDInput) -> HiringProfile:
    """Pure rule-based extraction — no LLM, no API key required."""
    text = jd.raw_text
    required, preferred = _extract_skills_fallback(text)
    responsibilities = _extract_responsibilities_fallback(text)
    seniority = _detect_seniority(text)
    employment_type = _detect_employment_type(text)
    years = _detect_years(text)
    ambiguity = _detect_ambiguity(text)
    hidden = _detect_hidden_expectations(text, required)
    job_title = _extract_job_title(text)
    domain = _extract_domain(text)

    return HiringProfile(
        job_title=job_title,
        seniority=seniority,
        employment_type=employment_type,
        domain=domain,
        years_of_experience_min=years,
        required_skills=required,
        preferred_skills=preferred,
        key_responsibilities=responsibilities,
        hidden_expectations=hidden,
        ambiguity_flags=ambiguity,
    )


# ---------------------------------------------------------------------------
# LLM mode
# ---------------------------------------------------------------------------


def _run_llm(jd: JDInput) -> HiringProfile:
    """LLM-powered extraction using structured_completion."""
    system_prompt = _load_system_prompt()
    user_prompt = (
        f"Here is the Job Description to analyse:\n\n"
        f"---\n{jd.raw_text}\n---\n\n"
        "Return a single JSON object matching the schema in your instructions."
    )
    return structured_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=HiringProfile,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyse_jd(
    jd: JDInput,
    *,
    force_fallback: bool = False,
) -> HiringProfile:
    """
    Run the JD Intelligence Agent on a validated JDInput.

    Automatically selects LLM mode if an API key is available,
    otherwise falls back to rule-based extraction.

    Args:
        jd:             Validated JDInput from Layer 1.
        force_fallback: If True, always use rule-based mode (for testing).

    Returns:
        A validated HiringProfile.
    """
    has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY"))

    if force_fallback or not has_key:
        mode = "rule-based fallback (no API key)" if not has_key else "rule-based fallback (forced)"
        logger.info("JD Intelligence Agent running in %s mode", mode)
        return _run_fallback(jd)

    logger.info("JD Intelligence Agent running in LLM mode")
    try:
        return _run_llm(jd)
    except Exception as exc:
        logger.warning(
            "LLM mode failed (%s) — falling back to rule-based extraction.", exc
        )
        return _run_fallback(jd)
