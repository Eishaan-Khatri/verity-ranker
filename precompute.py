#!/usr/bin/env python3
"""
Precompute hackathon features (internet allowed, no time limit).

Runs the heavy V2 intelligence once and writes:
  <cache-dir>/jd_profile.json
  <cache-dir>/hyde_profiles.json
  <cache-dir>/candidate_features.jsonl
  <cache-dir>/manifest.json

No GitHub API calls — uses github_activity_score from the dataset only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ai_hiring_ranker.hackathon.precompute_runner import run_precompute  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute hackathon candidate features.")
    parser.add_argument("--jd", required=True, type=Path, help="Path to job description text file")
    parser.add_argument("--candidates", required=True, type=Path, help="Path to candidates.jsonl")
    parser.add_argument("--cache-dir", default=Path("cache"), type=Path, help="Output cache directory")
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Skip LLM for JD analysis (fully offline)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N candidates")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    run_precompute(
        jd_path=args.jd,
        candidates_path=args.candidates,
        cache_dir=args.cache_dir,
        force_fallback=args.force_fallback,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
