# Quickstart

## 1. Run The Sample Pipeline

```bash
python run_v1.py run --jd data/sample/jd.txt --candidates data/sample/candidates --output outputs/final/ranked_output.csv --report outputs/final/audit_report.json
```

## 2. Validate The Output

```bash
python run_v1.py validate --file outputs/final/ranked_output.csv
```

## 3. Run Tests

```bash
python -m unittest discover -v
```

## 4. Run The Web Demo Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 5. Deploy Free Streamlit App

Use Streamlit Community Cloud:

1. Push this repo to GitHub.
2. Create a new Streamlit app.
3. Select repository `Eishaan-Khatri/verity-ranker`.
4. Set main file path to `app.py`.
5. Deploy.

GitHub Pages is not used because it only serves static HTML/CSS/JavaScript and cannot run this Python pipeline server-side.

## 6. Inspect Outputs

- `outputs/final/ranked_output.csv`
- `outputs/final/audit_report.json`

Generated outputs are intentionally ignored by Git. Regenerate them locally with the sample command above.

## 7. Replace Sample Data

Put the official challenge JD and candidate files under `data/raw/`, then point the command at those paths.

Do not assume the sample output schema is the official schema. Once the official format is known, update `src/verity_ranker/validate.py`.
