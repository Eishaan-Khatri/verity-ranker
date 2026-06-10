from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from verity_ranker.pipeline import run_pipeline


SAMPLE_JD = ROOT / "data" / "sample" / "jd.txt"
SAMPLE_CANDIDATES = ROOT / "data" / "sample" / "candidates"
WEIGHTS = ROOT / "configs" / "scoring_weights.json"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_with_sample_data() -> tuple[Path, Path, list[dict[str, str]], dict]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="verity_v1_"))
    output_path = tmp_dir / "ranked_output.csv"
    report_path = tmp_dir / "audit_report.json"
    run_pipeline(
        jd_path=SAMPLE_JD,
        candidates_dir=SAMPLE_CANDIDATES,
        output_path=output_path,
        report_path=report_path,
        weights_path=WEIGHTS,
    )
    rows = read_csv_rows(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return output_path, report_path, rows, report


def run_with_uploaded_data(jd_text: str, candidate_files) -> tuple[Path, Path, list[dict[str, str]], dict]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="verity_upload_"))
    jd_path = tmp_dir / "jd.txt"
    candidates_dir = tmp_dir / "candidates"
    output_path = tmp_dir / "ranked_output.csv"
    report_path = tmp_dir / "audit_report.json"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    jd_path.write_text(jd_text, encoding="utf-8")
    for index, uploaded in enumerate(candidate_files, start=1):
        suffix = Path(uploaded.name).suffix or ".txt"
        safe_name = f"candidate_{index:03d}{suffix}"
        (candidates_dir / safe_name).write_bytes(uploaded.getvalue())

    run_pipeline(
        jd_path=jd_path,
        candidates_dir=candidates_dir,
        output_path=output_path,
        report_path=report_path,
        weights_path=WEIGHTS,
    )
    rows = read_csv_rows(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return output_path, report_path, rows, report


def render_report(report: dict) -> None:
    role = report.get("role_profile", {})
    candidates = report.get("ranked_candidates", [])

    st.subheader("Role Profile")
    st.json(role, expanded=False)

    st.subheader("Candidate Audit")
    for candidate in candidates:
        with st.expander(f"#{candidate['rank']} {candidate['candidate_name']} ({candidate['candidate_id']})"):
            st.metric("Score", candidate["score"])
            st.write("Matched required skills:", ", ".join(candidate["matched_required_skills"]) or "None")
            st.write("Matched preferred skills:", ", ".join(candidate["matched_preferred_skills"]) or "None")
            st.write("Missing required skills:", ", ".join(candidate["missing_required_skills"]) or "None")
            st.json(candidate["evidence_summary"], expanded=False)

    st.subheader("V1 Limitations")
    for item in report.get("limitations", []):
        st.write(f"- {item}")


def main() -> None:
    st.set_page_config(page_title="Verity Ranker V1", page_icon="V", layout="wide")

    st.title("Verity Ranker V1")
    st.caption("Baseline candidate ranking demo. Runs the local V1 Python pipeline and validates ranked output.")

    st.info(
        "V1 is intentionally limited: no GitHub proof verification, no multi-agent scoring, "
        "no graph retrieval, no fairness audit, and no rank-stability audit."
    )

    mode = st.radio("Input mode", ["Use sample data", "Upload JD and candidate files"], horizontal=True)

    output_path = None
    report_path = None
    rows = None
    report = None

    if mode == "Use sample data":
        st.write("Runs the included sample JD and four synthetic candidates.")
        if st.button("Run sample ranking", type="primary"):
            output_path, report_path, rows, report = run_with_sample_data()

    if mode == "Upload JD and candidate files":
        jd_text = st.text_area("Job description", height=220)
        candidate_files = st.file_uploader(
            "Candidate files",
            type=["txt", "md"],
            accept_multiple_files=True,
            help="V1 accepts plain text or markdown candidate profiles.",
        )
        if st.button("Run uploaded ranking", type="primary"):
            if not jd_text.strip():
                st.error("Job description is required.")
            elif not candidate_files:
                st.error("Upload at least one candidate file.")
            else:
                output_path, report_path, rows, report = run_with_uploaded_data(jd_text, candidate_files)

    if rows is not None and report is not None and output_path is not None and report_path is not None:
        st.success(f"Ranked {len(rows)} candidates.")

        st.subheader("Ranked Output")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            st.download_button(
                "Download ranked_output.csv",
                data=output_path.read_bytes(),
                file_name="ranked_output.csv",
                mime="text/csv",
            )
        with right:
            st.download_button(
                "Download audit_report.json",
                data=report_path.read_bytes(),
                file_name="audit_report.json",
                mime="application/json",
            )

        render_report(report)


if __name__ == "__main__":
    main()
