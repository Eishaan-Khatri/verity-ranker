"""Orchestrate hackathon precompute across V2 layers (offline-safe per candidate)."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from ..hyde.generator import generate_hyde_profiles
from ..ingestion.schemas import JDInput
from ..jd_intelligence.agent import analyse_jd
from .dataset import iter_candidates
from .features import build_features
from .schemas import PrecomputeManifest

logger = logging.getLogger(__name__)


def _has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY"))


def run_precompute(
    jd_path: Path,
    candidates_path: Path,
    cache_dir: Path,
    *,
    force_fallback: bool = False,
    limit: int | None = None,
) -> PrecomputeManifest:
    """
    Run heavy intelligence once and write disk cache for ``rank.py``.

    Layers executed here:
      2  JD Intelligence
      3  HyDE ideal profiles (retrieval-only artefact)
      4  Profile extraction + 6–10 per candidate (agents, ledger, rubric)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    jd_cache = cache_dir / "jd_profile.json"
    hyde_cache = cache_dir / "hyde_profiles.json"
    feature_cache = cache_dir / "candidate_features.jsonl"

    use_fallback = force_fallback or not _has_api_key()
    jd_text = jd_path.read_text(encoding="utf-8")
    jd_input = JDInput(raw_text=jd_text, source_path=str(jd_path))

    t0 = time.perf_counter()

    logger.info("Layer 2: JD intelligence (force_fallback=%s)", use_fallback)
    hiring_profile = analyse_jd(jd_input, force_fallback=use_fallback)
    jd_cache.write_text(hiring_profile.model_dump_json(indent=2), encoding="utf-8")

    logger.info("Layer 3: HyDE ideal profiles")
    hyde_result = generate_hyde_profiles(hiring_profile, force_fallback=True)
    hyde_cache.write_text(hyde_result.model_dump_json(indent=2), encoding="utf-8")

    count = 0
    with feature_cache.open("w", encoding="utf-8") as out:
        for line_no, record in iter_candidates(candidates_path):
            if limit is not None and count >= limit:
                break
            features = build_features(
                record,
                hiring_profile,
                line_no=line_no,
                force_fallback=True,
            )
            out.write(json.dumps(features, ensure_ascii=False) + "\n")
            count += 1
            if count % 5000 == 0:
                logger.info("Precomputed %d candidates...", count)

    manifest = PrecomputeManifest(
        jd_path=str(jd_path.resolve()),
        candidates_path=str(candidates_path.resolve()),
        candidate_count=count,
        force_fallback=use_fallback,
        used_llm=not use_fallback,
        job_title=hiring_profile.job_title,
        cache_file=feature_cache.name,
        jd_cache_file=jd_cache.name,
        hyde_cache_file=hyde_cache.name,
        layers_precomputed=[
            "JD Intelligence",
            "HyDE Ideal Profiles",
            "Profile Extraction",
            "Evidence Ledger (github_activity_score)",
            "Multi-Agent Evaluation (rule-based)",
            "Rubric Scoring",
            "Honeypot / Stuffer / Engagement Guards",
        ],
        elapsed_seconds=round(time.perf_counter() - t0, 2),
    )
    (cache_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Precompute complete: %d candidates in %.1fs → %s",
        count,
        manifest.elapsed_seconds,
        cache_dir,
    )
    return manifest
