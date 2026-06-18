"""
Verity Ranker V2 — Recruiter Dashboard
Streamlit frontend for the full 15-layer AI hiring pipeline.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Verity Ranker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
html, body, [data-testid="stApp"] {
    background: #0f0f1a;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Cards */
.vr-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d2d4e;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: border-color 0.2s;
}
.vr-card:hover { border-color: #6366f1; }

/* Rank badge */
.rank-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 40px; height: 40px; border-radius: 50%;
    font-weight: 800; font-size: 1rem; color: #fff;
}
.rank-1 { background: linear-gradient(135deg,#f59e0b,#f97316); }
.rank-2 { background: linear-gradient(135deg,#94a3b8,#64748b); }
.rank-3 { background: linear-gradient(135deg,#92400e,#b45309); }
.rank-n { background: linear-gradient(135deg,#4f46e5,#7c3aed); }

/* Score pill */
.score-pill {
    display: inline-block; padding: .25rem .75rem;
    border-radius: 999px; font-weight: 700; font-size:.85rem;
}
.pill-exceptional { background:#14532d; color:#4ade80; }
.pill-strong      { background:#1e3a5f; color:#60a5fa; }
.pill-moderate    { background:#3b2200; color:#fb923c; }
.pill-weak        { background:#450a0a; color:#f87171; }
.pill-poor        { background:#1c1c1c; color:#94a3b8; }

/* Confidence badge */
.conf-high     { color:#4ade80; font-weight:700; }
.conf-medium   { color:#facc15; font-weight:700; }
.conf-low      { color:#fb923c; font-weight:700; }
.conf-unstable { color:#f87171; font-weight:700; }

/* Skill chips */
.chip-verified   { background:#14532d; color:#4ade80; border-radius:8px;
                   padding:.15rem .6rem; margin:.15rem; display:inline-block;
                   font-size:.78rem; font-weight:600; }
.chip-unverified { background:#450a0a; color:#f87171; border-radius:8px;
                   padding:.15rem .6rem; margin:.15rem; display:inline-block;
                   font-size:.78rem; font-weight:600; }
.chip-skill      { background:#1e1b4b; color:#a5b4fc; border-radius:8px;
                   padding:.15rem .6rem; margin:.15rem; display:inline-block;
                   font-size:.78rem; font-weight:600; }

/* Metric box */
.metric-box {
    background:#1a1a2e; border:1px solid #2d2d4e; border-radius:12px;
    padding:.8rem 1rem; text-align:center;
}
.metric-box .val { font-size:1.6rem; font-weight:800; color:#6366f1; }
.metric-box .lbl { font-size:.75rem; color:#94a3b8; margin-top:.15rem; }

/* Question cards */
.q-high   { border-left: 3px solid #f87171; padding-left:.75rem; margin:.4rem 0; }
.q-medium { border-left: 3px solid #facc15; padding-left:.75rem; margin:.4rem 0; }
.q-low    { border-left: 3px solid #4ade80; padding-left:.75rem; margin:.4rem 0; }

/* Section header */
.section-hdr {
    font-size:.7rem; font-weight:700; letter-spacing:.12em;
    text-transform:uppercase; color:#6366f1; margin:.6rem 0 .3rem;
}

/* Progress bar override */
.stProgress > div > div { background: linear-gradient(90deg,#6366f1,#8b5cf6) !important; }

/* Tab styling */
button[data-baseweb="tab"] { font-weight:600 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color:#6366f1 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background:#0f0f1a !important; border-right:1px solid #2d2d4e; }

/* Divider */
hr { border-color:#2d2d4e !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper renderers ─────────────────────────────────────────────────────────

SCORE_LABEL_MAP = {
    "exceptional": "pill-exceptional",
    "strong":      "pill-strong",
    "moderate":    "pill-moderate",
    "weak":        "pill-weak",
    "poor":        "pill-poor",
}
CONF_CLASS_MAP = {
    "high":     "conf-high",
    "medium":   "conf-medium",
    "low":      "conf-low",
    "unstable": "conf-unstable",
}
RANK_CLASS = {1: "rank-1", 2: "rank-2", 3: "rank-3"}


def score_pill(label: str, score: float) -> str:
    cls = SCORE_LABEL_MAP.get(label.lower(), "pill-poor")
    return f'<span class="score-pill {cls}">{score:.1f} / 100</span>'


def rank_badge(rank: int) -> str:
    cls = RANK_CLASS.get(rank, "rank-n")
    return f'<span class="rank-badge {cls}">{rank}</span>'


def conf_badge(conf: str) -> str:
    cls = CONF_CLASS_MAP.get(conf.lower(), "conf-medium")
    icon = {"high": "●", "medium": "●", "low": "⚠", "unstable": "⚠"}.get(conf.lower(), "●")
    return f'<span class="{cls}">{icon} {conf.upper()}</span>'


def dim_bar(label: str, value: float, color: str = "#6366f1") -> None:
    pct = int(value * 100)
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:.6rem;margin:.2rem 0'>"
        f"<span style='width:140px;font-size:.8rem;color:#94a3b8;flex-shrink:0'>{label}</span>"
        f"<div style='flex:1;background:#2d2d4e;border-radius:999px;height:8px;overflow:hidden'>"
        f"<div style='width:{pct}%;background:{color};height:100%;border-radius:999px'></div></div>"
        f"<span style='width:38px;text-align:right;font-size:.8rem;font-weight:700;color:#e2e8f0'>"
        f"{value:.2f}</span></div>",
        unsafe_allow_html=True,
    )


def skill_chips(skills: list[str], chip_class: str) -> str:
    if not skills:
        return "<span style='color:#64748b;font-size:.8rem'>None</span>"
    return " ".join(f'<span class="{chip_class}">{s}</span>' for s in skills)


def section_header(text: str) -> None:
    st.markdown(f'<div class="section-hdr">{text}</div>', unsafe_allow_html=True)


# ── Candidate card renderer ──────────────────────────────────────────────────

def render_candidate_card(card: dict, expanded: bool = False) -> None:
    """Render one full candidate card in a Streamlit expander."""
    rank   = card.get("rank", 0)
    name   = card.get("candidate_name") or card.get("candidate_id", "Unknown")
    score  = card.get("final_score", 0.0)
    label  = card.get("score_label", "moderate")
    conf   = card.get("rank_confidence", "medium")
    cid    = card.get("candidate_id", "")

    # Header row
    col_badge, col_info, col_score = st.columns([0.06, 0.6, 0.34])
    with col_badge:
        st.markdown(rank_badge(rank), unsafe_allow_html=True)
    with col_info:
        st.markdown(
            f"<div style='line-height:1.3'>"
            f"<span style='font-size:1.05rem;font-weight:700;color:#e2e8f0'>{name}</span>"
            f"<span style='font-size:.78rem;color:#64748b;margin-left:.6rem'>{cid}</span><br>"
            f"{conf_badge(conf)}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_score:
        st.markdown(
            f"<div style='text-align:right'>{score_pill(label, score)}</div>",
            unsafe_allow_html=True,
        )

    # Expander content
    with st.expander("View full report", expanded=expanded):
        tab_overview, tab_dims, tab_evidence, tab_why, tab_questions = st.tabs([
            "📊 Overview", "📐 Dimensions", "🔍 Evidence", "⚖️ Why This Rank", "💬 Interview Q"
        ])

        # ── Overview tab ───────────────────────────────────────────────
        with tab_overview:
            col_a, col_b, col_c, col_d = st.columns(4)
            verified   = card.get("verified_claims", 0)
            unverified = card.get("unverified_claims", 0)
            total_c    = verified + unverified
            proof_pct  = int(verified / total_c * 100) if total_c > 0 else 0

            col_a.markdown(
                f'<div class="metric-box"><div class="val">{score:.0f}</div>'
                f'<div class="lbl">Final Score</div></div>', unsafe_allow_html=True)
            col_b.markdown(
                f'<div class="metric-box"><div class="val">{verified}</div>'
                f'<div class="lbl">Verified Claims</div></div>', unsafe_allow_html=True)
            col_c.markdown(
                f'<div class="metric-box"><div class="val">{proof_pct}%</div>'
                f'<div class="lbl">Proof Ratio</div></div>', unsafe_allow_html=True)
            col_d.markdown(
                f'<div class="metric-box"><div class="val">{label.capitalize()}</div>'
                f'<div class="lbl">Tier</div></div>', unsafe_allow_html=True)

            if card.get("summary"):
                st.markdown(
                    f"<div style='background:#16213e;border-radius:10px;padding:.9rem 1rem;"
                    f"margin:.8rem 0;font-size:.9rem;color:#cbd5e1;line-height:1.6'>"
                    f"{card['summary']}</div>",
                    unsafe_allow_html=True,
                )

            col_s, col_r = st.columns(2)
            with col_s:
                section_header("Strengths")
                for s in card.get("strengths", []):
                    st.markdown(
                        f"<div style='display:flex;gap:.5rem;align-items:flex-start;"
                        f"margin:.25rem 0'><span style='color:#4ade80;flex-shrink:0'>✓</span>"
                        f"<span style='font-size:.85rem;color:#cbd5e1'>{s}</span></div>",
                        unsafe_allow_html=True,
                    )
            with col_r:
                section_header("Risks & Gaps")
                risks = card.get("risks", [])
                if risks:
                    for r in risks:
                        st.markdown(
                            f"<div style='display:flex;gap:.5rem;align-items:flex-start;"
                            f"margin:.25rem 0'><span style='color:#f87171;flex-shrink:0'>⚠</span>"
                            f"<span style='font-size:.85rem;color:#cbd5e1'>{r}</span></div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        "<span style='color:#4ade80;font-size:.85rem'>No critical risks.</span>",
                        unsafe_allow_html=True,
                    )

        # ── Dimensions tab ─────────────────────────────────────────────
        with tab_dims:
            section_header("Rubric Score Breakdown")
            dims = [
                ("Skill Fit",        card.get("skill_fit", 0),        "#6366f1"),
                ("Experience Depth", card.get("experience_depth", 0), "#8b5cf6"),
                ("Seniority Match",  card.get("seniority_match", 0),  "#06b6d4"),
                ("Domain Match",     card.get("domain_match", 0),     "#10b981"),
                ("Career Growth",    card.get("career_growth", 0),    "#f59e0b"),
                ("Proof Strength",   card.get("proof_strength", 0),   "#ef4444"),
            ]
            for lbl, val, color in dims:
                dim_bar(lbl, val, color)

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Weights Applied")
            w_cols = st.columns(6)
            weight_map = {"Skill Fit": "0.30", "Experience Depth": "0.20",
                          "Seniority Match": "0.15", "Domain Match": "0.15",
                          "Career Growth": "0.10", "Proof Strength": "0.10"}
            for i, (lbl, _, _) in enumerate(dims):
                w_cols[i].markdown(
                    f"<div style='text-align:center'>"
                    f"<div style='font-size:1rem;font-weight:700;color:#6366f1'>"
                    f"{weight_map[lbl]}</div>"
                    f"<div style='font-size:.65rem;color:#64748b'>{lbl}</div></div>",
                    unsafe_allow_html=True,
                )

        # ── Evidence tab ───────────────────────────────────────────────
        with tab_evidence:
            col_v, col_u = st.columns(2)
            with col_v:
                section_header(f"✓ Verified Skills  ({verified})")
                st.markdown(
                    skill_chips(card.get("verified_skills", [])[:10], "chip-verified"),
                    unsafe_allow_html=True,
                )
            with col_u:
                section_header(f"✗ Unverified Skills  ({unverified})")
                st.markdown(
                    skill_chips(card.get("unverified_skills", [])[:8], "chip-unverified"),
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='background:#16213e;border-radius:10px;padding:.9rem 1.1rem;"
                f"font-size:.82rem;color:#94a3b8'>"
                f"<span style='color:#6366f1;font-weight:700'>Proof Strength: "
                f"{card.get('proof_strength', 0):.2f}</span>  ·  "
                f"{verified} verified  ·  {unverified} unverified  ·  "
                f"{proof_pct}% proof ratio</div>",
                unsafe_allow_html=True,
            )

        # ── Why this rank tab ──────────────────────────────────────────
        with tab_why:
            why = card.get("why_above_next", "")
            gap = card.get("score_gap_to_next")

            if gap is not None:
                gap_color = "#4ade80" if gap > 5 else "#facc15" if gap > 2 else "#f87171"
                st.markdown(
                    f"<div style='background:#16213e;border-radius:10px;"
                    f"padding:.7rem 1rem;margin-bottom:.8rem;display:flex;"
                    f"align-items:center;gap:.8rem'>"
                    f"<span style='font-size:1.5rem;font-weight:800;color:{gap_color}'>"
                    f"+{gap:.1f}</span>"
                    f"<span style='font-size:.82rem;color:#94a3b8'>points above next candidate</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if why:
                st.markdown(
                    f"<div style='background:#1a1a2e;border-left:3px solid #6366f1;"
                    f"padding:.9rem 1.1rem;border-radius:0 10px 10px 0;"
                    f"font-size:.88rem;color:#cbd5e1;line-height:1.7'>{why}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No pairwise justification available for this candidate.")

        # ── Interview questions tab ────────────────────────────────────
        with tab_questions:
            questions = card.get("interview_questions", [])
            if not questions:
                st.info("No interview questions generated.")
            else:
                for q in questions:
                    priority = q.get("priority", "medium")
                    q_cls    = f"q-{priority}"
                    icon     = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
                    skill    = q.get("skill", "")
                    rationale = q.get("rationale", "")
                    st.markdown(
                        f"<div class='{q_cls}'>"
                        f"<div style='font-size:.82rem;margin-bottom:.2rem'>"
                        f"{icon} "
                        f"<span class='chip-skill'>{skill}</span>"
                        f"</div>"
                        f"<div style='font-size:.9rem;color:#e2e8f0;font-weight:600;"
                        f"margin:.25rem 0'>{q.get('question', '')}</div>"
                        + (f"<div style='font-size:.78rem;color:#64748b;font-style:italic'>"
                           f"{rationale}</div>" if rationale else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )

    st.markdown("<hr style='margin:.6rem 0'>", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, list, bool]:
    """Render sidebar and return (jd_text, candidate_files, force_fallback)."""
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:.5rem 0 1rem'>"
            "<span style='font-size:1.8rem'>⚡</span><br>"
            "<span style='font-size:1.1rem;font-weight:800;color:#e2e8f0'>Verity Ranker</span><br>"
            "<span style='font-size:.72rem;color:#6366f1;letter-spacing:.1em'>V2 · EVIDENCE-BACKED</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        mode = st.radio(
            "Input mode",
            ["📂 Sample data", "📤 Upload files"],
            label_visibility="collapsed",
        )

        jd_text        = ""
        candidate_files = []
        force_fallback  = True

        if mode == "📂 Sample data":
            st.markdown(
                "<div style='font-size:.8rem;color:#64748b;margin:.4rem 0 .8rem'>"
                "Runs the included JD + 4 synthetic candidates.</div>",
                unsafe_allow_html=True,
            )
            use_sample = True
        else:
            use_sample = False
            jd_text = st.text_area(
                "Job Description",
                placeholder="Paste the full job description here...",
                height=200,
            )
            candidate_files = st.file_uploader(
                "Candidate Resumes",
                type=["txt", "md", "pdf"],
                accept_multiple_files=True,
                help="Upload one file per candidate. PDF, TXT, or MD.",
            )

        st.divider()
        st.markdown('<div class="section-hdr">Pipeline Options</div>', unsafe_allow_html=True)
        force_fallback = not st.toggle(
            "Use LLM (requires API key)",
            value=False,
            help="When off, the pipeline runs fully offline with rule-based fallbacks.",
        )
        shortlist_k = st.slider("Retrieval shortlist size", 5, 50, 25, 5)
        stab_runs   = st.slider("Stability test runs", 3, 10, 5, 1)

        st.divider()
        run_btn = st.button("▶  Run Pipeline", type="primary", use_container_width=True)

    return mode, jd_text, candidate_files, force_fallback, shortlist_k, stab_runs, run_btn, \
           (mode == "📂 Sample data")


# ── Pipeline runner ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_v2_pipeline(
    jd_text: str,
    candidate_texts: list[tuple[str, str]],
    force_fallback: bool,
    k: int,
    stability_runs: int,
) -> dict:
    """Run the V2 pipeline and return a JSON-serialisable result dict."""
    from ai_hiring_ranker.evaluation import run_pipeline_from_texts
    from ai_hiring_ranker.audits.auditor import run_audit
    from ai_hiring_ranker.ablation import run_eval_ablation

    result = run_pipeline_from_texts(
        jd_text=jd_text,
        candidate_texts=candidate_texts,
        output_dir=str(ROOT / "outputs"),
        force_fallback=force_fallback,
        k=k,
        stability_runs=stability_runs,
    )
    return result.to_manifest_dict() | {"ranked_output": result.ranked_output}


def load_report_json(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


# ── Dashboard tabs ───────────────────────────────────────────────────────────

def render_overview_tab(manifest: dict, ranked: list[dict]) -> None:
    """Top-level run summary metrics."""
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    n = len(ranked)
    top_score = ranked[0]["final_score"] if ranked else 0
    verified_avg = (
        sum(r.get("verified_claims", 0) for r in ranked) / n if n else 0
    )
    stable_pct = (1 - manifest.get("has_unstable_ranks", False)) * 100

    for col, val, lbl in [
        (c1, str(n), "Candidates"),
        (c2, f"{top_score:.0f}", "Top Score"),
        (c3, f"{verified_avg:.1f}", "Avg Verified Claims"),
        (c4, manifest.get("fairness_risk", "info").upper(), "Fairness"),
        (c5, f"{manifest.get('duration_s', 0):.1f}s", "Run Time"),
    ]:
        col.markdown(
            f'<div class="metric-box"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    if manifest.get("warnings"):
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander(f"⚠ {len(manifest['warnings'])} pipeline warning(s)", expanded=False):
            for w in manifest["warnings"]:
                st.markdown(f"<div style='font-size:.82rem;color:#fb923c'>• {w}</div>",
                            unsafe_allow_html=True)

    # Quick ranking table
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Quick Ranking Summary")

    header_cols = st.columns([0.05, 0.22, 0.13, 0.13, 0.13, 0.13, 0.13, 0.08])
    labels = ["#", "Candidate", "Score", "Skill Fit", "Experience", "Seniority",
              "Proof Str.", "Tier"]
    for col, lbl in zip(header_cols, labels):
        col.markdown(
            f"<div style='font-size:.72rem;font-weight:700;color:#6366f1;"
            f"letter-spacing:.08em;text-transform:uppercase'>{lbl}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='background:#2d2d4e;height:1px;margin:.3rem 0'></div>",
                unsafe_allow_html=True)

    for row in ranked[:15]:
        r_cols = st.columns([0.05, 0.22, 0.13, 0.13, 0.13, 0.13, 0.13, 0.08])
        rank   = row.get("rank", 0)
        name   = row.get("candidate_name") or row.get("candidate_id", "")
        score  = row.get("final_score", 0.0)
        label  = row.get("score_label", "moderate")
        pill_cls = SCORE_LABEL_MAP.get(label, "pill-poor")

        r_cols[0].markdown(rank_badge(rank), unsafe_allow_html=True)
        r_cols[1].markdown(
            f"<span style='font-size:.85rem;font-weight:600;color:#e2e8f0'>{name}</span>",
            unsafe_allow_html=True,
        )
        r_cols[2].markdown(
            f"<span style='font-size:.85rem;font-weight:700;color:#e2e8f0'>{score:.1f}</span>",
            unsafe_allow_html=True,
        )
        for ci, dim in enumerate(["skill_fit", "experience_depth",
                                   "seniority_match", "proof_strength"]):
            val = row.get(dim, 0.0)
            bar_w = int(val * 100)
            r_cols[3 + ci].markdown(
                f"<div style='display:flex;align-items:center;gap:.3rem'>"
                f"<div style='flex:1;background:#2d2d4e;border-radius:999px;"
                f"height:6px;overflow:hidden'>"
                f"<div style='width:{bar_w}%;background:#6366f1;height:100%'></div></div>"
                f"<span style='font-size:.78rem;color:#94a3b8'>{val:.2f}</span></div>",
                unsafe_allow_html=True,
            )
        r_cols[7].markdown(
            f'<span class="score-pill {pill_cls}" style="font-size:.72rem">{label}</span>',
            unsafe_allow_html=True,
        )
