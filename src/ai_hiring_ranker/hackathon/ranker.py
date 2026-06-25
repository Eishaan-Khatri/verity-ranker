"""Fast offline top-100 selection and CSV export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .reasoning import build_reasoning
from .schemas import SubmissionRow


def load_feature_cache(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Null-safe float coercion. Treats missing key AND explicit None the same way."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_value(row: dict[str, Any]) -> float:
    val = row.get("final_score")
    if val is None:
        val = row.get("base_score")
    return _safe_float(val, 0.0)


def _listwise_sort_key(row: dict[str, Any]) -> tuple:
    """
    Rule-based listwise tie-break (Layer 11 offline).

    Uses full-precision score plus cached dimensions to compare candidates.
    """
    dims = row.get("dimensions") or {}
    return (
        -_score_value(row),
        -_safe_float(dims.get("proof_strength")),
        -_safe_float(dims.get("skill_fit")),
        -_safe_float(dims.get("seniority_match")),
        -_safe_float(dims.get("career_growth")),
        -_safe_float(dims.get("experience_depth")),
        -_safe_float(row.get("github_activity_score")),
        str(row.get("candidate_id", "")),
    )


def _submission_sort_key(row: dict[str, Any]) -> tuple:
    """Final ordering for CSV: rounded score desc, candidate_id asc on ties."""
    return (-round(_score_value(row), 2), str(row.get("candidate_id", "")))


def rank_candidates(
    features: Iterable[dict[str, Any]],
    *,
    job_title: str,
    top_k: int = 100,
    listwise_pool: int = 300,
) -> list[SubmissionRow]:
    """
    Sort candidates and return exactly ``top_k`` submission rows.

    1. Listwise re-rank a shortlist pool (Layer 11, offline rules).
    2. Re-order by displayed score + candidate_id tie-break for validator compliance.
    3. Assign ranks 1..top_k with fact-grounded reasoning.
    """
    all_rows = list(features)
    pool_size = max(top_k, listwise_pool)
    pool = sorted(all_rows, key=_listwise_sort_key)[:pool_size]
    ranked = sorted(pool[:top_k], key=_submission_sort_key)

    output: list[SubmissionRow] = []
    for idx, row in enumerate(ranked, start=1):
        score = round(_score_value(row), 2)
        output.append(
            SubmissionRow(
                candidate_id=str(row["candidate_id"]),
                rank=idx,
                score=score,
                reasoning=build_reasoning(row, job_title, idx),
            )
        )

    return output


def write_submission_csv(rows: list[SubmissionRow], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row.candidate_id,
                    "rank": row.rank,
                    "score": f"{row.score:.2f}",
                    "reasoning": row.reasoning,
                }
            )
    return path