"""
Output schemas for Layer 14 — Final Output Generator.

Layer 14 is the terminal layer of the pipeline. It collects outputs
from every upstream layer and produces two things:

  1. Required ranked output  — matches schemas/ranked_output.schema.json exactly.
                               Written to outputs/final/<run_id>_ranked.json.

  2. Pipeline run manifest   — a single JSON file that records every
                               output file path, run metadata, timing,
                               layer statuses, and any warnings.
                               Written to outputs/runs/<run_id>_manifest.json.

This layer does NOT score or rank — it only assembles, validates,
and serialises what all previous layers already computed.

PipelineResult is the in-memory container that holds every layer's
output for the duration of one run. It is the single object passed
between all pipeline stages and returned to the caller.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Layer status
# ---------------------------------------------------------------------------


class LayerStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETE  = "complete"
    SKIPPED   = "skipped"   # layer was intentionally skipped (e.g. no API key)
    FAILED    = "failed"    # layer errored but pipeline continued


class LayerRecord(BaseModel):
    """Execution record for one pipeline layer."""

    layer_num:    int         = Field(...)
    layer_name:   str         = Field(...)
    status:       LayerStatus = Field(default=LayerStatus.PENDING)
    duration_s:   float       = Field(default=0.0, description="Wall-clock seconds.")
    notes:        str         = Field(default="")
    error:        str         = Field(default="", description="Error message if status=FAILED.")


# ---------------------------------------------------------------------------
# Output file manifest
# ---------------------------------------------------------------------------


class OutputManifest(BaseModel):
    """
    Paths of every file written by the pipeline for one run.
    All paths are stored as strings for JSON serialisability.
    """

    run_id:           str            = Field(...)
    job_title:        str            = Field(default="")
    output_dir:       str            = Field(default="")
    ranked_json:      Optional[str]  = Field(default=None, description="Required ranked output.")
    ledger_json:      Optional[str]  = Field(default=None, description="Evidence ledger.")
    report_json:      Optional[str]  = Field(default=None, description="Recruiter report (JSON).")
    report_md:        Optional[str]  = Field(default=None, description="Recruiter report (Markdown).")
    audit_json:       Optional[str]  = Field(default=None, description="Layer 13 audit report.")
    manifest_json:    Optional[str]  = Field(default=None, description="This manifest file.")

    def to_dict(self) -> dict:
        return {
            "run_id":        self.run_id,
            "job_title":     self.job_title,
            "output_dir":    self.output_dir,
            "ranked_json":   self.ranked_json,
            "ledger_json":   self.ledger_json,
            "report_json":   self.report_json,
            "report_md":     self.report_md,
            "audit_json":    self.audit_json,
            "manifest_json": self.manifest_json,
        }


# ---------------------------------------------------------------------------
# Pipeline run result — in-memory container for all layer outputs
# ---------------------------------------------------------------------------


class PipelineResult(BaseModel):
    """
    Container for the complete output of one pipeline run.

    Holds every layer's typed output in memory so callers (CLI, Streamlit,
    tests) can access any layer's result without re-running the pipeline.

    Also carries execution metadata: run_id, timing, layer statuses,
    warnings, and the output file manifest.
    """

    # Run identity
    run_id:          str       = Field(...)
    job_title:       str       = Field(default="")
    started_at:      datetime  = Field(default_factory=datetime.utcnow)
    completed_at:    Optional[datetime] = Field(default=None)
    force_fallback:  bool      = Field(
        default=False,
        description="True if the run used rule-based fallbacks throughout (no LLM).",
    )

    # Layer execution records (one per layer, appended as layers complete)
    layer_records:   list[LayerRecord] = Field(default_factory=list)

    # Final output files
    manifest:        OutputManifest    = Field(
        default_factory=lambda: OutputManifest(run_id=""),
    )

    # Warnings collected across all layers
    warnings:        list[str]         = Field(default_factory=list)

    # Quick-access flags
    has_audit_warnings:  bool = Field(default=False)
    has_unstable_ranks:  bool = Field(default=False)
    has_fairness_flags:  bool = Field(default=False)

    # The required ranked output as a list of dicts
    # (matches schemas/ranked_output.schema.json)
    ranked_output:   list[dict] = Field(
        default_factory=list,
        description="Final ranked list. Matches ranked_output.schema.json.",
    )

    # ---------------------------------------------------------------------------
    # Convenience accessors
    # ---------------------------------------------------------------------------

    @property
    def duration_s(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    @property
    def candidate_count(self) -> int:
        return len(self.ranked_output)

    def get_layer(self, layer_num: int) -> Optional[LayerRecord]:
        return next((r for r in self.layer_records if r.layer_num == layer_num), None)

    def top_candidates(self, n: int = 5) -> list[dict]:
        """Return the top-n ranked candidates from the final output."""
        return self.ranked_output[:n]

    def summary(self) -> str:
        total_s = self.duration_s
        layers_ok = sum(1 for r in self.layer_records if r.status == LayerStatus.COMPLETE)
        layers_fail = sum(1 for r in self.layer_records if r.status == LayerStatus.FAILED)
        lines = [
            f"Pipeline run: {self.run_id}",
            f"Job:          {self.job_title}",
            f"Candidates:   {self.candidate_count}",
            f"Duration:     {total_s:.1f}s",
            f"Layers:       {layers_ok} complete, {layers_fail} failed",
        ]
        if self.warnings:
            lines.append(f"Warnings:     {len(self.warnings)}")
        if self.manifest.ranked_json:
            lines.append(f"Output:       {self.manifest.ranked_json}")
        if self.manifest.report_md:
            lines.append(f"Report:       {self.manifest.report_md}")
        return "\n".join(lines)

    def to_manifest_dict(self) -> dict:
        """Serialise the full run manifest for writing to disk."""
        return {
            "run_id":           self.run_id,
            "job_title":        self.job_title,
            "started_at":       self.started_at.isoformat(),
            "completed_at":     self.completed_at.isoformat() if self.completed_at else None,
            "duration_s":       round(self.duration_s, 2),
            "force_fallback":   self.force_fallback,
            "candidate_count":  self.candidate_count,
            "has_audit_warnings":  self.has_audit_warnings,
            "has_unstable_ranks":  self.has_unstable_ranks,
            "has_fairness_flags":  self.has_fairness_flags,
            "warnings":         self.warnings,
            "output_files":     self.manifest.to_dict(),
            "layer_records": [
                {
                    "layer":    r.layer_num,
                    "name":     r.layer_name,
                    "status":   r.status.value,
                    "duration": round(r.duration_s, 2),
                    "notes":    r.notes,
                    "error":    r.error,
                }
                for r in self.layer_records
            ],
        }
