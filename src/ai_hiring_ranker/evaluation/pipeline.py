"""
Final Output Generator & Pipeline Orchestrator — Layer 14.

This is the terminal layer of the verity-ranker V2 pipeline.
It wires every upstream layer into a single `run_pipeline()` call
and produces all required output files.

Pipeline execution order
------------------------
Layer 1   ingest()                     → JDInput + [CandidateInput]
Layer 2   analyse_jd()                 → HiringProfile
Layer 3   generate_hyde_profiles()     → HyDEResult
Layer 4   extract_all_candidates()     → [CandidateProfile]
Layer 5   verify_all_candidates()      → [VerificationReport]
Layer 6   build_run_ledger()           → RunLedger  +  save_run_ledger()
Layer 7   expand_skills()              (used inside Layers 8 & 9, no top-level call)
Layer 8   retrieve()                   → ShortlistResult
Layer 9   evaluate_all()               → BatchEvaluationResult
Layer 10  score_candidates()           → RankedOutput  +  save_ranked_output()
Layer 11  rerank()                     → RerankResult
Layer 12  generate_report()            → RecruiterReport  +  save_report()
Layer 13  run_audit()                  → AuditReport  +  save_audit()
Layer 14  assemble_final_output()      → writes final ranked JSON + manifest

Output files
------------
outputs/
  final/
    <run_id>_ranked.json       — required ranked output (matches schema)
    <run_id>_report.json       — recruiter report (machine-readable)
    <run_id>_report.md         — recruiter report (human Markdown)
    <run_id>_audit.json        — fairness + stability audit
    <run_id>_manifest.json     — full run manifest with all metadata
  runs/
    <run_id>_ledger.json       — evidence ledger

Public API
----------
run_pipeline(jd_path, candidates_dir, output_dir, run_id, force_fallback, k)
    → PipelineResult

run_pipeline_from_texts(jd_text, candidate_texts, ...)
    → PipelineResult
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..agents.orchestrator import evaluate_all
from ..audits.auditor import run_audit, save_audit
from ..candidate_extraction.extractor import extract_all_candidates
from ..claim_verification.agent import verify_all_candidates
from ..evidence.ledger import build_run_ledger, save_run_ledger
from ..hyde.generator import generate_hyde_profiles
from ..ingestion.loader import IngestResult, ingest
from ..jd_intelligence.agent import analyse_jd
from ..reporting.reporter import generate_report, save_report
from ..retrieval.retriever import retrieve
from ..reranking.reranker import rerank
from ..scoring.scorer import save_ranked_output, score_candidates
from .schemas import LayerRecord, LayerStatus, OutputManifest, PipelineResult

logger = logging.getLogger(__name__)

# Default output structure
_FINAL_DIR = "final"
_RUNS_DIR  = "runs"


# ---------------------------------------------------------------------------
# Layer timing helper
# ---------------------------------------------------------------------------

class _Timer:
    def __init__(self) -> None:
        self._start = time.monotonic()

    def elapsed(self) -> float:
        return round(time.monotonic() - self._start, 3)


def _record(
    result: PipelineResult,
    layer_num: int,
    layer_name: str,
    status: LayerStatus,
    duration_s: float,
    notes: str = "",
    error: str = "",
) -> None:
    result.layer_records.append(LayerRecord(
        layer_num=layer_num,
        layer_name=layer_name,
        status=status,
        duration_s=duration_s,
        notes=notes,
        error=error,
    ))


# ---------------------------------------------------------------------------
# Final output assembler (Layer 14 proper)
# ---------------------------------------------------------------------------


def assemble_final_output(
    rerank_result,
    ranked_output,
    output_dir: Path,
    run_id: str,
) -> list[dict]:
    """
    Produce the required ranked output JSON from the reranked result.

    The final ranked list uses the reranked order from Layer 11,
    but the scores and dimension breakdowns come from Layer 10.
    This guarantees:
      - The rank field reflects the listwise-corrected order
      - All dimension scores are traceable to agent verdicts
      - The output matches schemas/ranked_output.schema.json exactly
    """
    reranked_ids = [e.candidate_id for e in rerank_result.entries]
    score_map    = {s.candidate_id: s for s in ranked_output.scores}

    final_rows: list[dict] = []
    for rank, cid in enumerate(reranked_ids, start=1):
        score = score_map.get(cid)
        if score is None:
            continue
        row = score.to_export_dict()
        row["rank"] = rank
        # Annotate unstable ranks
        entry = rerank_result.get_entry(cid)
        if entry and entry.rank_confidence.value in ("low", "unstable"):
            row["rank_confidence"] = entry.rank_confidence.value
        final_rows.append(row)

    return final_rows


def _save_final_ranked(
    final_rows: list[dict],
    output_dir: Path,
    run_id: str,
) -> Path:
    path = output_dir / _FINAL_DIR / f"{run_id}_ranked.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(final_rows, fh, indent=2, ensure_ascii=False)
    logger.info("Final ranked output → %s", path)
    return path


def _save_manifest(result: PipelineResult, output_dir: Path) -> Path:
    path = output_dir / _RUNS_DIR / f"{result.run_id}_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_manifest_dict(), fh, indent=2, ensure_ascii=False)
    logger.info("Manifest → %s", path)
    return path


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    jd_path:        Optional[Path | str] = None,
    candidates_dir: Optional[Path | str] = None,
    *,
    output_dir:     Path | str = "outputs",
    run_id:         Optional[str] = None,
    force_fallback: bool = False,
    k:              int  = 25,
    stability_runs: int  = 5,
    # In-memory mode (for Streamlit / testing)
    jd_text:        Optional[str] = None,
    candidate_texts: Optional[list[tuple[str, str]]] = None,
) -> PipelineResult:
    """
    Run the complete verity-ranker V2 pipeline end-to-end.

    Filesystem mode (CLI):
        run_pipeline(jd_path="data/jd.txt", candidates_dir="data/candidates/")

    In-memory mode (Streamlit / testing):
        run_pipeline(jd_text="...", candidate_texts=[("C001", "resume text"), ...])

    Args:
        jd_path:          Path to the JD file (filesystem mode).
        candidates_dir:   Directory containing candidate files (filesystem mode).
        output_dir:       Root output directory. Default: "outputs/".
        run_id:           Explicit run ID; auto-generated UUID4 if None.
        force_fallback:   Use rule-based evaluation throughout (no LLM calls).
        k:                Retrieval shortlist size. Default 25.
        stability_runs:   Perturbation runs for the stability test. Default 5.
        jd_text:          Raw JD text (in-memory mode).
        candidate_texts:  List of (candidate_id, resume_text) tuples (in-memory mode).

    Returns:
        PipelineResult with all layer outputs and output file paths.
    """
    resolved_run_id = run_id or str(uuid.uuid4())[:8]
    out_dir         = Path(output_dir)
    final_dir       = out_dir / _FINAL_DIR
    runs_dir        = out_dir / _RUNS_DIR
    final_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    result = PipelineResult(
        run_id=resolved_run_id,
        force_fallback=force_fallback,
        manifest=OutputManifest(
            run_id=resolved_run_id,
            output_dir=str(out_dir.resolve()),
        ),
    )

    logger.info(
        "=== verity-ranker V2 pipeline start: run_id=%s force_fallback=%s ===",
        resolved_run_id, force_fallback,
    )

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 1 — Ingestion
    # ─────────────────────────────────────────────────────────────────────
    t = _Timer()
    try:
        if jd_text is not None and candidate_texts is not None:
            ingest_result: IngestResult = ingest(
                jd_path=None,
                candidates_dir=None,
                jd_text=jd_text,
                candidate_texts=candidate_texts,
            )
        else:
            ingest_result = ingest(
                jd_path=str(jd_path),
                candidates_dir=str(candidates_dir),
            )
        if ingest_result.errors:
            result.warnings.extend(ingest_result.errors)
        _record(result, 1, "Input Layer", LayerStatus.COMPLETE, t.elapsed(),
                notes=f"{len(ingest_result.candidates)} candidates ingested")
    except Exception as exc:
        _record(result, 1, "Input Layer", LayerStatus.FAILED, t.elapsed(), error=str(exc))
        result.completed_at = datetime.utcnow()
        logger.error("Layer 1 failed: %s", exc)
        return result

    jd_input        = ingest_result.jd
    candidate_inputs = ingest_result.candidates
    result.job_title = jd_input.raw_text[:60].replace("\n", " ").strip()

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 2 — JD Intelligence
    # ─────────────────────────────────────────────────────────────────────
    t = _Timer()
    try:
        hiring_profile = analyse_jd(jd_input, force_fallback=force_fallback)
        result.job_title = hiring_profile.job_title
        result.manifest.job_title = hiring_profile.job_title
        _record(result, 2, "JD Intelligence", LayerStatus.COMPLETE, t.elapsed(),
                notes=f"{len(hiring_profile.all_skill_names)} skills extracted")
    except Exception as exc:
        _record(result, 2, "JD Intelligence", LayerStatus.FAILED, t.elapsed(), error=str(exc))
        result.warnings.append(f"Layer 2 failed: {exc}")
        result.completed_at = datetime.utcnow()
        return result

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 3 — HyDE Generation
    # ─────────────────────────────────────────────────────────────────────
    t = _Timer()
    try:
        hyde_result = generate_hyde_profiles(hiring_profile, force_fallback=force_fallback)
        _record(result, 3, "HyDE Generation", LayerStatus.COMPLETE, t.elapsed(),
                notes="3 ideal profiles generated")
    except Exception as exc:
        _record(result, 3, "HyDE Generation", LayerStatus.FAILED, t.elapsed(), error=str(exc))
        result.warnings.append(f"Layer 3 failed: {exc}")
        result.completed_at = datetime.utcnow()
        return result

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 4 — Candidate Profile Extraction
    # ─────────────────────────────────────────────────────────────────────
    t = _Timer()
    try:
        profiles = extract_all_candidates(
            candidate_inputs, force_fallback=force_fallback
        )
        _record(result, 4, "Profile Extraction", LayerStatus.COMPLETE, t.elapsed(),
                notes=f"{len(profiles)} profiles extracted")
    except Exception as exc:
        _record(result, 4, "Profile Extraction", LayerStatus.FAILED, t.elapsed(), error=str(exc))
        result.warnings.append(f"Layer 4 failed: {exc}")
        result.completed_at = datetime.utcnow()
        return result

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 5 — Claim Verification
    # ─────────────────────────────────────────────────────────────────────
    t = _Timer()
    try:
        verification_reports = verify_all_candidates(
            profiles, candidate_inputs, force_fallback=force_fallback
        )
        _record(result, 5, "Claim Verification", LayerStatus.COMPLETE, t.elapsed(),
                notes=f"{len(verification_reports)} reports")
    except Exception as exc:
        _record(result, 5, "Claim Verification", LayerStatus.FAILED, t.elapsed(), error=str(exc))
        result.warnings.append(f"Layer 5 failed: {exc}")
        verification_reports = []

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 6 — Evidence Ledger
    # ─────────────────────────────────────────────────────────────────────
    t = _Timer()
    try:
        run_ledger = build_run_ledger(
            verification_reports,
            job_title=hiring_profile.job_title,
            run_id=resolved_run_id,
        )
        ledger_path = save_run_ledger(run_ledger, runs_dir)
        result.manifest.ledger_json = str(ledger_path)
        ledger_map = {c.candidate_id: c for c in run_ledger.candidates}
        _record(result, 6, "Evidence Ledger", LayerStatus.COMPLETE, t.elapsed(),
                notes=f"{run_ledger.total_claims} claims recorded")
    except Exception as exc:
        _record(result, 6, "Evidence Ledger", LayerStatus.FAILED, t.elapsed(), error=str(exc))
        result.warnings.append(f"Layer 6 failed: {exc}")
        ledger_map = {}

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 7 — Skill Graph (used implicitly inside Layers 8 & 9)
    # ─────────────────────────────────────────────────────────────────────
    _record(result, 7, "Skill Graph", LayerStatus.SKIPPED, 0.0,
            notes="Graph loaded on-demand inside Layers 8 and 9")

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 8 — Hybrid Retrieval
    # ─────────────────────────────────────────────────────────────────────
    t = _Timer()
    try:
        shortlist = retrieve(
            profiles, hiring_profile, hyde_result,
            k=k, force_fallback=force_fallback,
        )
        shortlist_ids = shortlist.shortlisted_ids
        _record(result, 8, "Hybrid Retrieval", LayerStatus.COMPLETE, t.elapsed(),
                notes=f"Shortlisted {len(shortlist_ids)}/{len(profiles)} candidates")
    except Exception as exc:
        _record(result, 8, "Hybrid Retrieval", LayerStatus.FAILED, t.elapsed(), error=str(exc))
        result.warnings.append(f"Layer 8 failed — evaluating all candidates: {exc}")
        shortlist_ids = [p.candidate_id for p in profiles]

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 9 — Multi-Agent Evaluation
    # ─────────────────────────────────────────────────────────────────────
    t = _Timer()
    try:
        batch_eval = evaluate_all(
            profiles, hiring_profile,
            ledger_map=ledger_map,
            shortlist_ids=shortlist_ids,
            force_fallback=force_fallback,
        )
        _record(result, 9, "Multi-Agent Evaluation", LayerStatus.COMPLETE, t.elapsed(),
                notes=f"{len(batch_eval.results)} candidates evaluated")
    except Exception as exc:
        _record(result, 9, "Multi-Agent Evaluation", LayerStatus.FAILED, t.elapsed(), error=str(exc))
        result.warnings.append(f"Layer 9 failed: {exc}")
        result.completed_at = datetime.utcnow()
        return result

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 10 — Rubric Scoring
    # ─────────────────────────────────────────────────────────────────────
    t = _Timer()
    try:
        ranked_output = score_candidates(
            batch_eval, hiring_profile,
            ledger_map=ledger_map,
            run_id=resolved_run_id,
        )
        # Save Layer 10 ranked output (intermediate — Layer 14 will overwrite with final)
        save_ranked_output(ranked_output, final_dir)
        _record(result, 10, "Rubric Scoring", LayerStatus.COMPLETE, t.elapsed(),
                notes=f"Top score: {ranked_output.ranked[0].final_score:.1f}" if ranked_output.ranked else "")
    except Exception as exc:
        _record(result, 10, "Rubric Scoring", LayerStatus.FAILED, t.elapsed(), error=str(exc))
        result.warnings.append(f"Layer 10 failed: {exc}")
        result.completed_at = datetime.utcnow()
        return result

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 11 — Listwise Re-Ranking
    # ─────────────────────────────────────────────────────────────────────
    t = _Timer()
    try:
        rerank_result = rerank(
            ranked_output, hiring_profile,
            eval_results=batch_eval,
            force_fallback=force_fallback,
        )
        result.has_unstable_ranks = rerank_result.unstable_count > 0
        if result.has_unstable_ranks:
            result.warnings.append(
                f"{rerank_result.unstable_count} candidate(s) have unstable rank confidence."
            )
        _record(result, 11, "Listwise Re-Ranking", LayerStatus.COMPLETE, t.elapsed(),
                notes=f"method={rerank_result.rerank_method} unstable={rerank_result.unstable_count}")
    except Exception as exc:
        _record(result, 11, "Listwise Re-Ranking", LayerStatus.FAILED, t.elapsed(), error=str(exc))
        result.warnings.append(f"Layer 11 failed — using Layer 10 order: {exc}")
        # Build a minimal RerankResult from Layer 10 order
        from ..reranking.schemas import RankConfidence, RerankEntry, RerankResult
        entries = [
            RerankEntry(
                candidate_score=s,
                original_rank=i + 1,
                reranked_rank=i + 1,
                rank_confidence=RankConfidence.MEDIUM,
            )
            for i, s in enumerate(ranked_output.ranked)
        ]
        rerank_result = RerankResult(
            job_title=hiring_profile.job_title,
            entries=entries,
            rerank_method="fallback",
        )

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 12 — Recruiter Report
    # ─────────────────────────────────────────────────────────────────────
    t = _Timer()
    try:
        recruiter_report = generate_report(
            rerank_result, hiring_profile,
            eval_results=batch_eval,
            ledger_map=ledger_map,
            run_id=resolved_run_id,
            force_fallback=force_fallback,
        )
        report_json, report_md = save_report(recruiter_report, final_dir)
        result.manifest.report_json = str(report_json)
        result.manifest.report_md   = str(report_md)
        _record(result, 12, "Recruiter Report", LayerStatus.COMPLETE, t.elapsed(),
                notes=f"{len(recruiter_report.cards)} candidate cards")
    except Exception as exc:
        _record(result, 12, "Recruiter Report", LayerStatus.FAILED, t.elapsed(), error=str(exc))
        result.warnings.append(f"Layer 12 failed: {exc}")

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 13 — Fairness + Stability Audit
    # ─────────────────────────────────────────────────────────────────────
    t = _Timer()
    try:
        audit_report = run_audit(
            profiles, rerank_result, hiring_profile,
            run_id=resolved_run_id,
            n_runs=stability_runs,
        )
        audit_path = save_audit(audit_report, final_dir)
        result.manifest.audit_json = str(audit_path)
        result.has_audit_warnings  = audit_report.has_any_warnings
        result.has_fairness_flags  = audit_report.fairness.has_warnings
        if audit_report.has_any_warnings:
            result.warnings.append(
                f"Audit warnings: fairness={audit_report.fairness.overall_risk_level.value}, "
                f"unstable={audit_report.stability.unstable_count}"
            )
        _record(result, 13, "Fairness + Stability Audit", LayerStatus.COMPLETE, t.elapsed(),
                notes=(
                    f"fairness={audit_report.fairness.overall_risk_level.value} "
                    f"stable={audit_report.stability.stable_count}/"
                    f"{audit_report.stability.stable_count + audit_report.stability.unstable_count}"
                ))
    except Exception as exc:
        _record(result, 13, "Fairness + Stability Audit", LayerStatus.FAILED, t.elapsed(), error=str(exc))
        result.warnings.append(f"Layer 13 failed: {exc}")

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 14 — Final Output Assembly
    # ─────────────────────────────────────────────────────────────────────
    t = _Timer()
    try:
        final_rows = assemble_final_output(rerank_result, ranked_output, out_dir, resolved_run_id)
        ranked_path = _save_final_ranked(final_rows, out_dir, resolved_run_id)
        result.ranked_output        = final_rows
        result.manifest.ranked_json = str(ranked_path)
        _record(result, 14, "Final Output Generator", LayerStatus.COMPLETE, t.elapsed(),
                notes=f"{len(final_rows)} candidates in final output")
    except Exception as exc:
        _record(result, 14, "Final Output Generator", LayerStatus.FAILED, t.elapsed(), error=str(exc))
        result.warnings.append(f"Layer 14 failed: {exc}")

    # ─────────────────────────────────────────────────────────────────────
    # Write manifest
    # ─────────────────────────────────────────────────────────────────────
    result.completed_at = datetime.utcnow()
    try:
        manifest_path = _save_manifest(result, out_dir)
        result.manifest.manifest_json = str(manifest_path)
    except Exception as exc:
        result.warnings.append(f"Manifest write failed: {exc}")

    logger.info(
        "=== Pipeline complete: run_id=%s  duration=%.1fs  candidates=%d ===",
        resolved_run_id, result.duration_s, result.candidate_count,
    )
    logger.info(result.summary())

    return result


# ---------------------------------------------------------------------------
# In-memory convenience wrapper (for Streamlit)
# ---------------------------------------------------------------------------


def run_pipeline_from_texts(
    jd_text:          str,
    candidate_texts:  list[tuple[str, str]],
    *,
    output_dir:     Path | str = "outputs",
    run_id:         Optional[str] = None,
    force_fallback: bool = False,
    k:              int  = 25,
    stability_runs: int  = 3,
) -> PipelineResult:
    """
    Run the pipeline from in-memory text inputs (no filesystem reads).

    Args:
        jd_text:          Raw JD text string.
        candidate_texts:  List of (candidate_id, resume_text) tuples.
        output_dir:       Output directory for all generated files.
        run_id:           Optional explicit run ID.
        force_fallback:   Use rule-based evaluation throughout.
        k:                Retrieval shortlist size.
        stability_runs:   Perturbation runs for stability test.

    Returns:
        PipelineResult — same as run_pipeline().
    """
    return run_pipeline(
        jd_text=jd_text,
        candidate_texts=candidate_texts,
        output_dir=output_dir,
        run_id=run_id,
        force_fallback=force_fallback,
        k=k,
        stability_runs=stability_runs,
    )
