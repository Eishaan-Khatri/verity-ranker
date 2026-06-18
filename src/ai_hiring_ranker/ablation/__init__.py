"""
Layer 15 — Evaluation & Ablation Report.

Produces two reports from a completed PipelineResult (Layer 14):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. EvaluationReport — pipeline quality metrics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Measures without requiring ground-truth labels:

  Score distribution
    mean, std, spread (max−min), tier counts (exceptional/strong/moderate/weak/poor)

  Dimension discriminative power
    σ per dimension — low σ means the dimension can't separate candidates

  Proof coverage
    fraction of candidates with proof_strength ≥ 0.40
    average verified claims per candidate

  Retrieval recall
    did Layer 8's shortlist capture all strong candidates?

  Rank stability (from Layer 13)
    what fraction of ranks were stable across perturbation runs?

  Fairness risk (from Layer 13)
    was any proxy bias detected?

  Narrative
    auto-generated strengths, concerns, and recommendations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. AblationReport — signal importance study
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Removes each scoring dimension one at a time, re-ranks, measures:

  Kendall's τ  — rank correlation with full-signal ranking
  rank_changes — number of candidates whose rank changed
  avg_rank_shift — mean absolute shift
  importance    — critical / important / moderate / low

  τ ≈ 1.0 → signal has little effect (possibly redundant)
  τ ≈ 0.0 → signal is critical for ranking (removing it breaks order)

No new API calls. Uses only data from the completed PipelineResult.

Output files
------------
outputs/final/<run_id>_eval_ablation.json
outputs/final/<run_id>_eval_ablation.md

Public API
----------
from ai_hiring_ranker.ablation import (
    run_eval_ablation,    # main entry point → EvalAblationBundle
    run_evaluation,       # → EvaluationReport
    run_ablation,         # → AblationReport
    save_eval_ablation,   # → (json_path, md_path)
    EvalAblationBundle,
    EvaluationReport,
    AblationReport,
    AblationRun,
    DimensionSummary,
)
"""

from .evaluator import (
    run_ablation,
    run_eval_ablation,
    run_evaluation,
    save_eval_ablation,
)
from .schemas import (
    AblationReport,
    AblationRun,
    DimensionSummary,
    EvalAblationBundle,
    EvaluationReport,
)

__all__ = [
    # Functions
    "run_eval_ablation",
    "run_evaluation",
    "run_ablation",
    "save_eval_ablation",
    # Schema types
    "EvalAblationBundle",
    "EvaluationReport",
    "AblationReport",
    "AblationRun",
    "DimensionSummary",
]
