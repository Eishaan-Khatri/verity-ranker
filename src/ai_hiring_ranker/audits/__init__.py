"""
Layer 13 — Fairness + Proxy Audit & Rank Stability Test.

Two independent, non-blocking audits run on the final ranking.
Neither audit changes the ranking — they only attach flags and
warnings to the output for recruiter and compliance review.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Audit 1 — Fairness + Proxy Audit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Checks proxy fields that must NOT drive ranking:
  university_name  — prestige bias (top-k impact ratio)
  graduation_year  — age proxy (Kendall's τ correlation)
  career_gap       — gap penalty bias
  location         — geographic bias (best-effort)
  name             — name/gender signal (bucketed heuristic)

Severity levels: INFO → WARNING → HIGH

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Audit 2 — Rank Stability Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Re-runs rubric scoring N times with controlled perturbations:
  weight_jitter  — ±5% nudge on each dimension weight
  score_jitter   — ±3% Gaussian noise on each dimension score

Per candidate:
  is_stable = True  if rank varied by ≤ 1 position
  is_stable = False → flagged as lower-confidence placement

Output written to: outputs/final/<run_id>_audit.json

Public API
----------
from ai_hiring_ranker.audits import (
    run_audit,             # combined entry point → AuditReport
    run_fairness_audit,    # → FairnessAuditResult
    run_stability_audit,   # → StabilityAuditResult
    save_audit,            # AuditReport → Path
    AuditReport,
    FairnessAuditResult,
    StabilityAuditResult,
    ProxyFlag,
    CandidateStability,
    FlagSeverity,
)
"""

from .auditor import run_audit, run_fairness_audit, run_stability_audit, save_audit
from .schemas import (
    AuditReport,
    CandidateStability,
    FairnessAuditResult,
    FlagSeverity,
    ProxyFlag,
    StabilityAuditResult,
    TopKImpactRatio,
)

__all__ = [
    # Functions
    "run_audit",
    "run_fairness_audit",
    "run_stability_audit",
    "save_audit",
    # Schema types
    "AuditReport",
    "FairnessAuditResult",
    "StabilityAuditResult",
    "ProxyFlag",
    "CandidateStability",
    "TopKImpactRatio",
    "FlagSeverity",
]
