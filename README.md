# Verity Ranker V2

V2 is a local proof-of-work baseline for candidate ranking.

It is intentionally limited. It does not claim full agentic evaluation, external GitHub API verification, graph reasoning, or fairness guarantees. Its job is to prove that the project can:

- read a job description
- read candidate profiles/resumes
- extract basic role and candidate signals
- classify resume claims by proof strength
- rank candidates with a deterministic scoring rubric
- produce a valid ranked output file
- generate evidence, claim verification, and audit reports

## Why V2 Exists

Most ranking systems trust resume claims too easily. V2 adds a deterministic proof-strength layer so unsupported or weakly supported claims do not score the same as claims backed by evidence.

## Run Sample

From this folder:

```bash
python run_v2.py run --jd data/sample/jd.txt --candidates data/sample/candidates --output outputs/final/ranked_output.csv --report outputs/final/audit_report.json
python run_v2.py validate --file outputs/final/ranked_output.csv
```

## Web Demo

V2 includes a Streamlit app because GitHub Pages cannot run the Python ranker server-side. Streamlit Community Cloud can deploy this repository from GitHub and run the existing Python pipeline.

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app can:

- run the included sample JD and candidate files
- accept uploaded `.txt` or `.md` candidate profiles
- display ranked output
- download `ranked_output.csv`
- download `audit_report.json`
- download `evidence_ledger.json`
- download `claim_verification_report.json`

## Current Scope

Included:

- basic JD parsing
- basic candidate parsing
- skill extraction through a controlled skill vocabulary
- claim verification labels
- evidence ledger generation
- proof-adjusted skill scoring
- deterministic weighted scoring
- ranked CSV output
- JSON audit report
- output validation

Not included:

- external GitHub API proof-of-work verification
- skill/role knowledge graph
- HyDE retrieval
- multi-agent committee
- listwise reranking
- fairness/proxy audit
- rank stability audit

These are intentionally left for later versions.
