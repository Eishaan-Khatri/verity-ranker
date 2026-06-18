"""
Output schemas for Layer 12 — Recruiter Audit Report.

The recruiter report is a human-readable document separate from the
required ranked output JSON. It explains the ranking to the recruiter
in plain language, backed by cited evidence for every claim.

Contents per candidate card:
  - Rank, name, final score, score label
  - Top strengths (evidence-cited)
  - Risks and gaps
  - Verified vs unverified claim summary
  - Why ranked above the next candidate (from Layer 11 pairwise justifications)
  - Recommended interview verification questions
  - Full dimension score breakdown

The full report also includes:
  - A run summary (job, total candidates, top picks, unstable ranks)
  - Interview question bank per candidate
  - Rank stability warnings for low-confidence placements

Serialised to:
  outputs/final/<run_id>_report.json  (machine-readable)
  outputs/final/<run_id>_report.md    (human-readable Markdown)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Interview question
# ---------------------------------------------------------------------------


class InterviewQuestion(BaseModel):
    """One recommended interview question for verifying a specific claim."""

    question:    str  = Field(..., description="The interview question text.")
    skill:       str  = Field(default="", description="The skill this question probes.")
    rationale:   str  = Field(
        default="",
        description="Why this question is recommended (e.g. claim is unverified).",
    )
    priority:    str  = Field(
        default="medium",
        description="high / medium / low — how critical this verification is.",
    )


# ---------------------------------------------------------------------------
# Per-candidate card
# ---------------------------------------------------------------------------


class CandidateCard(BaseModel):
    """
    Complete recruiter-facing summary for one candidate.

    This is what a recruiter reads to understand why a candidate was
    ranked where they were and what to ask in the interview.
    """

    # Identity & rank
    rank:               int    = Field(ge=1)
    candidate_id:       str    = Field(...)
    candidate_name:     str    = Field(default="")
    final_score:        float  = Field(ge=0.0, le=100.0)
    score_label:        str    = Field(default="")  # exceptional/strong/moderate/weak/poor

    # Dimension scores (0–1)
    skill_fit:          float  = Field(default=0.0, ge=0.0, le=1.0)
    experience_depth:   float  = Field(default=0.0, ge=0.0, le=1.0)
    seniority_match:    float  = Field(default=0.0, ge=0.0, le=1.0)
    domain_match:       float  = Field(default=0.0, ge=0.0, le=1.0)
    career_growth:      float  = Field(default=0.0, ge=0.0, le=1.0)
    proof_strength:     float  = Field(default=0.0, ge=0.0, le=1.0)

    # Evidence summary
    verified_claims:    int    = Field(default=0)
    unverified_claims:  int    = Field(default=0)
    verified_skills:    list[str] = Field(default_factory=list)
    unverified_skills:  list[str] = Field(default_factory=list)

    # Narrative
    strengths:          list[str] = Field(default_factory=list)
    risks:              list[str] = Field(default_factory=list)
    summary:            str       = Field(default="")

    # Why above next candidate
    why_above_next:     str       = Field(
        default="",
        description=(
            "Evidence-based explanation of why this candidate is ranked "
            "above the next one. From Layer 11 pairwise justifications."
        ),
    )
    score_gap_to_next:  Optional[float] = Field(
        default=None,
        description="Score gap to the immediately lower-ranked candidate.",
    )
    rank_confidence:    str = Field(
        default="medium",
        description="high / medium / low / unstable — from Layer 11.",
    )

    # Interview questions
    interview_questions: list[InterviewQuestion] = Field(default_factory=list)

    @property
    def proof_ratio(self) -> float:
        total = self.verified_claims + self.unverified_claims
        return round(self.verified_claims / total, 2) if total > 0 else 0.0

    def to_markdown(self) -> str:
        """Render this card as a Markdown section."""
        lines: list[str] = [
            f"## #{self.rank} — {self.candidate_name or self.candidate_id}",
            f"**Score:** {self.final_score:.1f}/100  [{self.score_label.upper()}]  "
            f"| Rank confidence: {self.rank_confidence}",
            "",
            "### Dimension Scores",
            f"| Dimension | Score |",
            f"|-----------|-------|",
            f"| Skill Fit          | {self.skill_fit:.2f} |",
            f"| Experience Depth   | {self.experience_depth:.2f} |",
            f"| Seniority Match    | {self.seniority_match:.2f} |",
            f"| Domain Match       | {self.domain_match:.2f} |",
            f"| Career Growth      | {self.career_growth:.2f} |",
            f"| Proof Strength     | {self.proof_strength:.2f} |",
            "",
            "### Evidence Summary",
            f"- **Verified claims:** {self.verified_claims}  "
            f"| **Unverified:** {self.unverified_claims}  "
            f"| **Proof ratio:** {self.proof_ratio:.0%}",
        ]

        if self.verified_skills:
            lines.append(
                f"- **Verified skills:** {', '.join(self.verified_skills[:8])}"
            )
        if self.unverified_skills:
            lines.append(
                f"- **Unverified skills:** {', '.join(self.unverified_skills[:5])}"
            )

        lines += ["", "### Strengths"]
        for s in self.strengths:
            lines.append(f"- {s}")

        lines += ["", "### Risks & Gaps"]
        if self.risks:
            for r in self.risks:
                lines.append(f"- ⚠ {r}")
        else:
            lines.append("- No critical risks identified.")

        if self.summary:
            lines += ["", "### Summary", self.summary]

        if self.why_above_next:
            gap_str = (
                f" (gap: {self.score_gap_to_next:.1f} pts)"
                if self.score_gap_to_next is not None else ""
            )
            lines += [
                "", f"### Why Ranked Above #{self.rank + 1}{gap_str}",
                self.why_above_next,
            ]

        if self.interview_questions:
            lines += ["", "### Recommended Interview Questions"]
            for iq in self.interview_questions:
                priority_icon = "🔴" if iq.priority == "high" else "🟡" if iq.priority == "medium" else "🟢"
                lines.append(f"{priority_icon} **[{iq.skill}]** {iq.question}")
                if iq.rationale:
                    lines.append(f"   *{iq.rationale}*")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full recruiter report
# ---------------------------------------------------------------------------


class RecruiterReport(BaseModel):
    """
    Complete recruiter audit report for one pipeline run.

    Contains one CandidateCard per shortlisted candidate, a run summary,
    and global flags (unstable ranks, low-evidence candidates).
    """

    run_id:          str                  = Field(default="")
    job_title:       str                  = Field(default="")
    generated_at:    datetime             = Field(default_factory=datetime.utcnow)
    total_evaluated: int                  = Field(default=0)
    cards:           list[CandidateCard]  = Field(
        default_factory=list,
        description="One card per candidate, sorted by rank ascending.",
    )
    unstable_rank_warnings: list[str]     = Field(
        default_factory=list,
        description="Warnings for candidates with LOW or UNSTABLE rank confidence.",
    )
    run_notes:       list[str]            = Field(
        default_factory=list,
        description="Global notes about this run (e.g. fallback mode used, API errors).",
    )

    @property
    def top_card(self) -> Optional[CandidateCard]:
        return self.cards[0] if self.cards else None

    def get_card(self, candidate_id: str) -> Optional[CandidateCard]:
        return next((c for c in self.cards if c.candidate_id == candidate_id), None)

    def to_markdown(self) -> str:
        """Render the full report as a Markdown document."""
        lines: list[str] = [
            f"# Recruiter Audit Report",
            f"**Job:** {self.job_title}  "
            f"| **Run:** {self.run_id}  "
            f"| **Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Candidates evaluated:** {self.total_evaluated}  "
            f"| **Shortlisted:** {len(self.cards)}",
            "",
        ]

        if self.unstable_rank_warnings:
            lines.append("## ⚠ Rank Stability Warnings")
            for w in self.unstable_rank_warnings:
                lines.append(f"- {w}")
            lines.append("")

        if self.run_notes:
            lines.append("## Run Notes")
            for n in self.run_notes:
                lines.append(f"- {n}")
            lines.append("")

        lines.append("## Quick Ranking Summary")
        lines.append(
            f"| Rank | Candidate | Score | Label | Proof% |"
        )
        lines.append(f"|------|-----------|-------|-------|--------|")
        for card in self.cards:
            lines.append(
                f"| #{card.rank} | {card.candidate_name or card.candidate_id} "
                f"| {card.final_score:.1f} | {card.score_label} "
                f"| {card.proof_ratio:.0%} |"
            )

        lines.append("")
        lines.append("---")
        lines.append("")

        for card in self.cards:
            lines.append(card.to_markdown())
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def to_export_dict(self) -> dict:
        """Serialise to JSON-serialisable dict."""
        return {
            "run_id":          self.run_id,
            "job_title":       self.job_title,
            "generated_at":    self.generated_at.isoformat(),
            "total_evaluated": self.total_evaluated,
            "unstable_rank_warnings": self.unstable_rank_warnings,
            "run_notes":       self.run_notes,
            "candidates":      [
                {
                    "rank":              c.rank,
                    "candidate_id":      c.candidate_id,
                    "candidate_name":    c.candidate_name,
                    "final_score":       round(c.final_score, 2),
                    "score_label":       c.score_label,
                    "rank_confidence":   c.rank_confidence,
                    "skill_fit":         round(c.skill_fit, 4),
                    "experience_depth":  round(c.experience_depth, 4),
                    "seniority_match":   round(c.seniority_match, 4),
                    "domain_match":      round(c.domain_match, 4),
                    "career_growth":     round(c.career_growth, 4),
                    "proof_strength":    round(c.proof_strength, 4),
                    "verified_claims":   c.verified_claims,
                    "unverified_claims": c.unverified_claims,
                    "verified_skills":   c.verified_skills,
                    "strengths":         c.strengths,
                    "risks":             c.risks,
                    "summary":           c.summary,
                    "why_above_next":    c.why_above_next,
                    "score_gap_to_next": c.score_gap_to_next,
                    "interview_questions": [
                        {
                            "skill":     iq.skill,
                            "question":  iq.question,
                            "priority":  iq.priority,
                            "rationale": iq.rationale,
                        }
                        for iq in c.interview_questions
                    ],
                }
                for c in self.cards
            ],
        }
