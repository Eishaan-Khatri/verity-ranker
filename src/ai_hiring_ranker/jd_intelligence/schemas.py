"""
Output schemas for the JD Intelligence Agent.

These are the structured objects produced after the LLM analyses
a raw Job Description. Everything downstream (HyDE, retrieval,
scoring rubric) reads from HiringProfile.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SeniorityLevel(str, Enum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"
    LEAD = "lead"
    MANAGER = "manager"
    UNKNOWN = "unknown"


class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    INTERNSHIP = "internship"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Skill with context
# ---------------------------------------------------------------------------


class SkillEntry(BaseModel):
    """A skill extracted from the JD, with metadata about how it was mentioned."""

    skill: str = Field(..., description="Normalised skill name, e.g. 'Python', 'FastAPI'.")
    is_required: bool = Field(..., description="True if the JD marks this as required/must-have.")
    is_preferred: bool = Field(..., description="True if the JD marks this as preferred/nice-to-have.")
    context_snippet: Optional[str] = Field(
        default=None,
        description="The sentence from the JD where this skill was found.",
    )

    @field_validator("skill", mode="before")
    @classmethod
    def normalise_skill(cls, v: str) -> str:
        return v.strip().title()


# ---------------------------------------------------------------------------
# Hidden expectation / ambiguity
# ---------------------------------------------------------------------------


class HiddenExpectation(BaseModel):
    """A requirement that is implied but not stated explicitly in the JD."""

    description: str = Field(
        ...,
        description="What the expectation is, in plain language.",
    )
    inferred_from: str = Field(
        ...,
        description="The JD phrase or context that implies this expectation.",
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="How confident we are that this is a real hidden expectation (0–1).",
    )


class AmbiguityFlag(BaseModel):
    """A vague or underspecified requirement in the JD."""

    phrase: str = Field(..., description="The ambiguous phrase from the JD.")
    reason: str = Field(..., description="Why this phrase is ambiguous.")
    suggested_clarification: Optional[str] = Field(
        default=None,
        description="How a recruiter could clarify this.",
    )


# ---------------------------------------------------------------------------
# Core output: HiringProfile
# ---------------------------------------------------------------------------


class HiringProfile(BaseModel):
    """
    Structured hiring profile produced by the JD Intelligence Agent.

    This is the single source of truth about what the role needs.
    Every downstream agent reads from this, never from the raw JD text.
    """

    # Role metadata
    job_title: str = Field(..., description="Normalised job title extracted from the JD.")
    seniority: SeniorityLevel = Field(
        default=SeniorityLevel.UNKNOWN,
        description="Detected seniority level.",
    )
    employment_type: EmploymentType = Field(
        default=EmploymentType.UNKNOWN,
        description="Detected employment type.",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Industry or domain, e.g. 'Machine Learning', 'FinTech', 'Healthcare'.",
    )
    years_of_experience_min: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum years of experience explicitly or implicitly required.",
    )

    # Skills
    required_skills: list[SkillEntry] = Field(
        default_factory=list,
        description="Skills the JD marks as required / must-have.",
    )
    preferred_skills: list[SkillEntry] = Field(
        default_factory=list,
        description="Skills the JD marks as preferred / nice-to-have.",
    )

    # Responsibilities
    key_responsibilities: list[str] = Field(
        default_factory=list,
        description="Main responsibilities / duties extracted from the JD.",
    )

    # Hidden signal
    hidden_expectations: list[HiddenExpectation] = Field(
        default_factory=list,
        description="Implied requirements not stated explicitly.",
    )
    ambiguity_flags: list[AmbiguityFlag] = Field(
        default_factory=list,
        description="Vague or underspecified requirements that a recruiter should clarify.",
    )

    # Convenience accessors
    @property
    def all_required_skill_names(self) -> list[str]:
        return [s.skill for s in self.required_skills]

    @property
    def all_preferred_skill_names(self) -> list[str]:
        return [s.skill for s in self.preferred_skills]

    @property
    def all_skill_names(self) -> list[str]:
        return self.all_required_skill_names + self.all_preferred_skill_names
