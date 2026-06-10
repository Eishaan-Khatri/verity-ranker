from dataclasses import dataclass
from pathlib import Path

from .candidates import load_candidates
from .io import load_weights, write_audit_report, write_claim_verification_report, write_evidence_ledger, write_ranked_output
from .jd import parse_jd
from .scoring import rank_candidates
from .validate import validate_ranked_output
from .verification import build_claim_verifications, build_evidence_ledger, build_verification_index


@dataclass(frozen=True)
class PipelineResult:
    output_path: Path
    report_path: Path
    evidence_path: Path | None
    claim_report_path: Path | None
    candidate_count: int


def run_pipeline(
    jd_path: Path,
    candidates_dir: Path,
    output_path: Path,
    report_path: Path,
    weights_path: Path,
    evidence_path: Path | None = None,
    claim_report_path: Path | None = None,
) -> PipelineResult:
    role = parse_jd(jd_path.read_text(encoding="utf-8"))
    candidates = load_candidates(candidates_dir)
    weights = load_weights(weights_path)
    verifications = build_claim_verifications(candidates, role)
    verification_index = build_verification_index(verifications)
    ledger = build_evidence_ledger(verifications)
    scores = rank_candidates(candidates, role, weights, verification_index)
    write_ranked_output(output_path, scores)
    validate_ranked_output(output_path)
    write_audit_report(report_path, role, scores)
    if evidence_path is not None:
        write_evidence_ledger(evidence_path, ledger)
    if claim_report_path is not None:
        write_claim_verification_report(claim_report_path, verifications)
    return PipelineResult(
        output_path=output_path,
        report_path=report_path,
        evidence_path=evidence_path,
        claim_report_path=claim_report_path,
        candidate_count=len(scores),
    )
