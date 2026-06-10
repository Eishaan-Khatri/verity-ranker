# V2 Methodology

V2 adds a proof-strength layer to the V1 ranking baseline.

## What Changed From V1

- Claims are classified as `verified`, `weakly_supported`, `unsupported`, or `unverifiable`.
- Resume body evidence scores higher than skills-line-only claims.
- Negated claims are treated as unsupported.
- Required and preferred skill fit are proof-adjusted.
- The pipeline generates `evidence_ledger.json`.
- The pipeline generates `claim_verification_report.json`.

## Current Verification Rules

- Strong action evidence in candidate body text: `verified`.
- Body text mention without strong action evidence: `weakly_supported`.
- Skills-line-only claim: `weakly_supported` with lower proof strength.
- Negated claim: `unsupported`.
- No candidate evidence for a JD skill: `unverifiable`.

## Important Limitation

This local V2 branch does not call the GitHub API or inspect external repositories yet. It prepares the evidence and scoring structure needed for that later step.

