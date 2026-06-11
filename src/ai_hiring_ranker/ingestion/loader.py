"""
Ingestion loader — the single public entry point for Layer 1.

Usage
-----
    from ai_hiring_ranker.ingestion import ingest

    result = ingest(
        jd_path=Path("data/sample/jd.txt"),
        candidates_dir=Path("data/sample/candidates"),
    )
    # result.jd          → JDInput
    # result.candidates  → list[CandidateInput]
    # result.errors      → list[str]  (non-fatal parse failures)

Design
------
- Errors on individual candidate files are collected and reported rather than
  crashing the whole run.  A bad PDF from one candidate should not block the
  other ten.
- The JD is fatal: if it cannot be loaded the whole run should stop.
- candidate_id is resolved in this order:
    1. Explicit "Candidate ID: <value>" field in the resume text.
    2. File stem (e.g. "C001" from "C001.pdf").
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .link_extractor import extract_portfolio_links
from .parsers import extract_text
from .schemas import CandidateInput, FileFormat, JDInput

logger = logging.getLogger(__name__)

# Formats we will attempt to load
SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".txt", ".md", ".pdf", ".docx"})

# Regex to pull an explicit candidate ID from resume text
_CANDIDATE_ID_RE = re.compile(
    r"^\s*Candidate\s+ID\s*:\s*(.+)$",
    re.MULTILINE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Everything produced by :func:`ingest`."""

    jd: JDInput
    candidates: list[CandidateInput]
    errors: list[str] = field(default_factory=list)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def summary(self) -> str:
        lines = [
            f"JD ingested: {self.jd.word_count} words  ({self.jd.file_format.value})",
            f"Candidates ingested: {self.candidate_count}",
        ]
        for c in self.candidates:
            link_note = " [has links]" if c.portfolio_links.has_any else ""
            lines.append(f"  • {c.candidate_id} — {c.word_count} words{link_note}")
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  ✗ {err}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_format(path: Path) -> FileFormat:
    mapping = {
        ".txt": FileFormat.TXT,
        ".md": FileFormat.MD,
        ".pdf": FileFormat.PDF,
        ".docx": FileFormat.DOCX,
    }
    return mapping.get(path.suffix.lower(), FileFormat.UNKNOWN)


def _resolve_candidate_id(text: str, path: Path) -> str:
    """Return an explicit Candidate ID from the text, or fall back to the file stem."""
    match = _CANDIDATE_ID_RE.search(text)
    if match:
        return match.group(1).strip()
    return path.stem


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------


def load_jd(jd_path: Path) -> JDInput:
    """Load and validate a Job Description file.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the text is too short to be a real JD (< 50 chars after strip).
        pydantic.ValidationError: if schema validation fails.
    """
    raw_text = extract_text(jd_path)
    return JDInput(
        source_path=jd_path,
        file_format=_detect_format(jd_path),
        raw_text=raw_text,
    )


def load_candidate(path: Path) -> CandidateInput:
    """Load, parse, and validate a single candidate resume file.

    Raises:
        FileNotFoundError / ValueError: propagated from :func:`extract_text`.
        pydantic.ValidationError: if schema validation fails.
    """
    raw_text = extract_text(path)
    candidate_id = _resolve_candidate_id(raw_text, path)
    portfolio_links = extract_portfolio_links(raw_text)

    return CandidateInput(
        candidate_id=candidate_id,
        source_path=path,
        file_format=_detect_format(path),
        raw_text=raw_text,
        portfolio_links=portfolio_links,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def ingest(
    jd_path: Path,
    candidates_dir: Path,
    *,
    jd_text: Optional[str] = None,
    candidate_texts: Optional[list[tuple[str, str]]] = None,
) -> IngestResult:
    """Ingest all inputs and return a validated :class:`IngestResult`.

    Parameters
    ----------
    jd_path:
        Path to the JD file. Required when *jd_text* is not provided.
    candidates_dir:
        Directory containing resume files. Required when *candidate_texts*
        is not provided.
    jd_text:
        Raw JD text, used instead of loading from *jd_path* (e.g. for
        Streamlit textarea uploads).
    candidate_texts:
        List of (filename, raw_text) tuples, used instead of scanning
        *candidates_dir* (e.g. for in-memory file uploads).

    Returns
    -------
    IngestResult
        Contains the validated JD, all successfully parsed candidates, and
        a list of error strings for any files that failed.
    """
    errors: list[str] = []

    # ── JD ──────────────────────────────────────────────────────────────────
    if jd_text is not None:
        jd = JDInput(
            source_path=None,
            file_format=FileFormat.TXT,
            raw_text=jd_text,
        )
        logger.info("JD loaded from in-memory text (%d words)", jd.word_count)
    else:
        jd = load_jd(jd_path)
        logger.info("JD loaded from %s (%d words)", jd_path.name, jd.word_count)

    # ── Candidates ──────────────────────────────────────────────────────────
    candidates: list[CandidateInput] = []

    if candidate_texts is not None:
        # In-memory mode (e.g., Streamlit file uploader bytes)
        for filename, raw_text in candidate_texts:
            fake_path = Path(filename)
            candidate_id = _resolve_candidate_id(raw_text, fake_path)
            portfolio_links = extract_portfolio_links(raw_text)
            try:
                candidate = CandidateInput(
                    candidate_id=candidate_id,
                    source_path=None,
                    file_format=_detect_format(fake_path),
                    raw_text=raw_text,
                    portfolio_links=portfolio_links,
                )
                candidates.append(candidate)
                logger.debug("Ingested in-memory candidate: %s", candidate_id)
            except Exception as exc:
                msg = f"{filename}: {exc}"
                errors.append(msg)
                logger.warning("Failed to ingest candidate %s: %s", filename, exc)
    else:
        # Filesystem mode
        candidate_files = sorted(
            p for p in candidates_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not candidate_files:
            raise ValueError(
                f"No supported resume files found in {candidates_dir}. "
                f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            )

        for path in candidate_files:
            try:
                candidate = load_candidate(path)
                candidates.append(candidate)
                logger.debug(
                    "Ingested %s → %s (%d words)",
                    path.name,
                    candidate.candidate_id,
                    candidate.word_count,
                )
            except Exception as exc:
                msg = f"{path.name}: {exc}"
                errors.append(msg)
                logger.warning("Failed to ingest candidate %s: %s", path.name, exc)

    if not candidates:
        raise ValueError(
            "No candidates could be ingested. Check the errors list for details.\n"
            + "\n".join(errors)
        )

    logger.info(
        "Ingestion complete: 1 JD, %d candidates, %d errors",
        len(candidates),
        len(errors),
    )

    return IngestResult(jd=jd, candidates=candidates, errors=errors)
