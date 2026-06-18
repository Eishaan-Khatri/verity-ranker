"""
Layer 12 — Recruiter Audit Report.

Generates a complete, evidence-backed recruiter report from all upstream
layers. Separate from the required ranked output JSON — this is the
human-readable document a recruiter actually reads.

Contents per candidate:
  - Rank, score, score label, rank confidence
  - Dimension score breakdown (6 dimensions)
  - Verified vs unverified claim summary
  - Top strengths (evidence-cited, from Layer 9)
  - Risks and gaps
  - Why ranked above the next candidate (Layer 11 pairwise justifications)
  - Recommended interview questions (LLM-tailored or template-based)

Report-level:
  - Quick ranking summary table
  - Rank stability warnings for low-confidence placements
  - Run notes (fallback mode, API errors, etc.)

Saved as:
  outputs/final/<run_id>_report.json  (machine-readable)
  outputs/final/<run_id>_report.md    (Markdown — readable in any editor)

Public API
----------
from ai_hiring_ranker.reporting import (
    generate_report,   # main entry point → RecruiterReport
    save_report,       # RecruiterReport → (json_path, md_path)
    RecruiterReport,
    CandidateCard,
    InterviewQuestion,
)
"""

from .reporter import generate_report, save_report
from .schemas import CandidateCard, InterviewQuestion, RecruiterReport

__all__ = [
    "generate_report",
    "save_report",
    "RecruiterReport",
    "CandidateCard",
    "InterviewQuestion",
]
