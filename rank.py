#!/usr/bin/env python3
"""
Offline hackathon ranker (sandbox script).

NO network calls. NO GPU. Loads precomputed cache + JD, scores all candidates,
outputs exactly 100 rows:

  candidate_id,rank,score,reasoning

Usage:
  python rank.py --jd data/jd.txt --candidates data/candidates.jsonl --cache-dir cache --output submission/ranked_output.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ai_hiring_ranker.hackathon.ranker import (  # noqa: E402
    load_feature_cache,
    rank_candidates,
    write_submission_csv,
)
from ai_hiring_ranker.jd_intelligence.schemas import HiringProfile  # noqa: E402
from scripts.validate_submission import validate_submission  # noqa: E402

logger = logging.getLogger(__name__)


def _load_hiring_profile(jd_path: Path, cache_dir: Path) -> HiringProfile:
    jd_cache = cache_dir / "jd_profile.json"
    if jd_cache.exists():
        return HiringProfile.model_validate_json(jd_cache.read_text(encoding="utf-8"))

    raise FileNotFoundError(
        f"Missing {jd_cache}. Run precompute.py first:\n"
        f"  python precompute.py --jd {jd_path} --candidates <candidates.jsonl> "
        f"--cache-dir {cache_dir}"
    )


def run_rank(
    jd_path: Path,
    candidates_path: Path,
    cache_dir: Path,
    output_path: Path,
    *,
    top_k: int = 100,
    skip_validate: bool = False,
) -> Path:
    t0 = time.perf_counter()
    hiring_profile = _load_hiring_profile(jd_path, cache_dir)

    feature_cache = cache_dir / "candidate_features.jsonl"
    if not feature_cache.exists():
        raise FileNotFoundError(
            f"Missing {feature_cache}. Run precompute.py first on {candidates_path}."
        )

    logger.info("Loading precomputed cache: %s", feature_cache)
    features = load_feature_cache(feature_cache)
    if not features:
        raise ValueError(f"Feature cache is empty: {feature_cache}")

    rows = rank_candidates(features, job_title=hiring_profile.job_title, top_k=top_k)
    out = write_submission_csv(rows, output_path)

    if not skip_validate:
        errors = validate_submission(out)
        if errors:
            raise ValueError("Submission validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    elapsed = time.perf_counter() - t0
    logger.info(
        "Ranked %d candidates → top %d in %.2fs → %s",
        len(features),
        len(rows),
        elapsed,
        out,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline hackathon ranker (sandbox).")
    parser.add_argument("--jd", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--cache-dir", default=Path("cache"), type=Path)
    parser.add_argument("--output", default=Path("submission/ranked_output.csv"), type=Path)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    run_rank(
        jd_path=args.jd,
        candidates_path=args.candidates,
        cache_dir=args.cache_dir,
        output_path=args.output,
        top_k=args.top_k,
        skip_validate=args.skip_validate,
    )


if __name__ == "__main__":
    main()
