"""
Pydantic v2 schemas for all ingestion-layer data models.

These are the validated, typed inputs that flow into every downstream agent.
Nothing enters the pipeline without passing these schemas.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FileFormat(str, Enum):
    TXT = "txt"
    MD = "md"
    PDF = "pdf"
    DOCX = "docx"
    UNKNOWN = "unknown"


class VerificationStatus(str, Enum):
    """Claim verification states used by the Evidence Ledger downstream."""
    VERIFIED = "verified"
    WEAK = "weak"
    INFERRED = "inferred"
    UNSUPPORTED = "unsupported"
    PENDING = "pending"


# ---------------------------------------------------------------------------
# Portfolio / external link schema
# ---------------------------------------------------------------------------


class PortfolioLinks(BaseModel):
    """All optional external profile links extracted from a resume."""

    github: Optional[str] = Field(
        default=None,
        description="GitHub profile or repository URL.",
    )
    kaggle: Optional[str] = Field(
        default=None,
        description="Kaggle profile URL.",
    )
    linkedin: Optional[str] = Field(
        default=None,
        description="LinkedIn profile URL.",
    )
    portfolio: list[str] = Field(
        default_factory=list,
        description="Any other portfolio / project URLs found in the resume.",
    )

    @field_validator("github", "kaggle", "linkedin", mode="before")
    @classmethod
    def strip_whitespace(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if isinstance(value, str) else value

    @property
    def has_any(self) -> bool:
        return bool(self.github or self.kaggle or self.linkedin or self.portfolio)


# ---------------------------------------------------------------------------
# Job Description input schema
# ---------------------------------------------------------------------------


class JDInput(BaseModel):
    """Validated, structured representation of a raw Job Description."""

    source_path: Optional[Path] = Field(
        default=None,
        description="Filesystem path the JD was loaded from (None for in-memory/upload).",
    )
    file_format: FileFormat = Field(
        default=FileFormat.TXT,
        description="Detected file format.",
    )
    raw_text: str = Field(
        ...,
        min_length=50,
        description="Full raw text of the JD, exactly as parsed.",
    )
    char_count: int = Field(
        default=0,
        description="Character count of raw_text, populated automatically.",
    )
    word_count: int = Field(
        default=0,
        description="Word count of raw_text, populated automatically.",
    )
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of ingestion.",
    )

    @model_validator(mode="after")
    def populate_counts(self) -> "JDInput":
        self.char_count = len(self.raw_text)
        self.word_count = len(self.raw_text.split())
        return self

    @field_validator("raw_text", mode="before")
    @classmethod
    def strip_raw_text(cls, value: str) -> str:
        return value.strip()


# ---------------------------------------------------------------------------
# Candidate input schema
# ---------------------------------------------------------------------------


class CandidateInput(BaseModel):
    """Validated, structured representation of a single candidate resume."""

    candidate_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier — derived from filename stem or an explicit 'Candidate ID:' field in the resume.",
    )
    source_path: Optional[Path] = Field(
        default=None,
        description="Filesystem path the resume was loaded from.",
    )
    file_format: FileFormat = Field(
        default=FileFormat.TXT,
        description="Detected file format.",
    )
    raw_text: str = Field(
        ...,
        min_length=20,
        description="Full raw text of the resume, exactly as parsed.",
    )
    char_count: int = Field(default=0)
    word_count: int = Field(default=0)
    portfolio_links: PortfolioLinks = Field(
        default_factory=PortfolioLinks,
        description="All external links extracted from the resume text.",
    )
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def populate_counts(self) -> "CandidateInput":
        self.char_count = len(self.raw_text)
        self.word_count = len(self.raw_text.split())
        return self

    @field_validator("raw_text", mode="before")
    @classmethod
    def strip_raw_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("candidate_id", mode="before")
    @classmethod
    def strip_id(cls, value: str) -> str:
        return value.strip()
