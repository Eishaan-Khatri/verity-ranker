"""
Layer 14 — Final Output Generator & Pipeline Orchestrator.

This is the terminal layer of the verity-ranker V2 pipeline.
It wires all 13 upstream layers into a single `run_pipeline()` call
and produces all required output files.

What it does
------------
1. Runs every layer in order (Layers 1–13)
2. Assembles the final ranked output in the correct schema format
3. Writes all output files to outputs/final/ and outputs/runs/
4. Returns a PipelineResult with all layer outputs + metadata

Output files produced
---------------------
outputs/
  final/
    <run_id>_ranked.json     — required ranked output (ranked_output.schema.json)
    <run_id>_report.json     — recruiter report (machine-readable)
    <run_id>_report.md       — recruiter report (human Markdown)
    <run_id>_audit.json      — fairness + stability audit
    <run_id>_manifest.json   — run manifest with all metadata + timing
  runs/
    <run_id>_ledger.json     — evidence ledger (claim audit trail)

Usage
-----
# From filesystem paths (CLI):
from ai_hiring_ranker.evaluation import run_pipeline

result = run_pipeline(
    jd_path="data/sample/jd.txt",
    candidates_dir="data/sample/candidates/",
    output_dir="outputs/",
    force_fallback=True,   # no API key needed
)
print(result.summary())

# From in-memory text (Streamlit):
from ai_hiring_ranker.evaluation import run_pipeline_from_texts

result = run_pipeline_from_texts(
    jd_text="We are looking for a Senior ML Engineer...",
    candidate_texts=[
        ("C001", "Resume text for candidate 1..."),
        ("C002", "Resume text for candidate 2..."),
    ],
    force_fallback=True,
)

Public API
----------
from ai_hiring_ranker.evaluation import (
    run_pipeline,              # filesystem mode
    run_pipeline_from_texts,   # in-memory mode
    PipelineResult,
    LayerRecord,
    LayerStatus,
    OutputManifest,
)
"""

from .pipeline import run_pipeline, run_pipeline_from_texts
from .schemas import LayerRecord, LayerStatus, OutputManifest, PipelineResult

__all__ = [
    "run_pipeline",
    "run_pipeline_from_texts",
    "PipelineResult",
    "LayerRecord",
    "LayerStatus",
    "OutputManifest",
]
