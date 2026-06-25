"""
PRISM — Proof-driven Ranking & Intelligent Selection Model
Streamlit frontend, rebuilt against the verified real output schema of
ai_hiring_ranker.evaluation.run_pipeline_from_texts().

Design concept: a prism splits one beam of light into a spectrum. PRISM
splits one candidate score into six rubric dimensions. That single idea —
the "spectral signature" — is the recurring visual device used everywhere
a score needs to be read.

Data flow (verified):
  run_pipeline_from_texts(...) returns `result` where:
    - result.to_manifest_dict()  -> run metadata + layer_records + warnings
    - result.manifest.to_dict()  -> output file paths (report/audit/ledger/...)
  report.json  -> {"candidates": [ {rank, candidate_id, candidate_name,
                    final_score, score_label, rank_confidence, skill_fit,
                    experience_depth, seniority_match, domain_match,
                    career_growth, proof_strength, verified_claims,
                    unverified_claims, verified_skills, strengths, risks,
                    summary, why_above_next, score_gap_to_next,
                    interview_questions}, ... ]}
  audit.json   -> {fairness_risk_level, fairness_flags, top_k_ratios,
                    stability_ratio, unstable_candidates,
                    candidate_stability: [{candidate_id, base_rank,
                    rank_range, score_std, is_stable}, ...]}
  ledger.json  -> {candidates: [{candidate_id, claims: [{claim_id, skill,
                    claim_text, source, verification_status, confidence,
                    evidence_url, recency_years, notes}, ...]}, ...]}

Note: exact field shapes for `fairness_flags` and `top_k_ratios` were not
observed in testing (empty in our trial runs) — they're rendered through a
defensive fallback that shows every key/value pair rather than guessing
field names and silently dropping data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
import sys
sys.path.insert(0, str(SRC))

st.set_page_config(
    page_title="PRISM",
    page_icon="🔺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens — single source of truth ───────────────────────────────────
# Every dimension owns exactly one hex, used identically in CSS vars, the
# logo, and the spectral-signature SVG. No second palette anywhere.
DIMENSIONS = [
    ("skill_fit", "Skill Fit", "0.30", "#FF4D6D"),
    ("experience_depth", "Experience Depth", "0.20", "#FF8A3D"),
    ("seniority_match", "Seniority Match", "0.15", "#FFD93D"),
    ("domain_match", "Domain Match", "0.15", "#38D39F"),
    ("career_growth", "Career Growth", "0.10", "#4DA8FF"),
    ("proof_strength", "Proof Strength", "0.10", "#8B5CF6"),
]
TIER_LEVELS = {"exceptional": 5, "strong": 4, "moderate": 3, "weak": 2, "poor": 1}
TIER_COLOR = {
    "exceptional": "var(--spec-domain)", "strong": "var(--spec-domain)",
    "moderate": "var(--spec-seniority)", "weak": "var(--spec-skill)", "poor": "var(--spec-skill)",
}
CONF_COLOR = {
    "high": "var(--spec-domain)", "medium": "var(--spec-seniority)",
    "low": "var(--spec-skill)", "unstable": "var(--spec-skill)",
}
STATUS_COLOR = {
    "verified": "var(--spec-domain)", "weak": "var(--spec-seniority)",
    "inferred": "var(--spec-growth)", "unsupported": "var(--spec-skill)",
    "contradicted": "var(--spec-skill)", "pending": "var(--ink-muted)",
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');

:root{
  --bg:#050816;
  --bg2:#0B1023;
  --surface:rgba(255,255,255,.06);
  --surface-hover:rgba(255,255,255,.10);
  --paper:rgba(255,255,255,.04);
  --line:rgba(255,255,255,.08);
  --ink:#F8FAFF;
  --ink-muted:#AAB2D5;

  --spec-skill:#FF4D6D;
  --spec-experience:#FF8A3D;
  --spec-seniority:#FFD93D;
  --spec-domain:#38D39F;
  --spec-growth:#4DA8FF;
  --spec-proof:#8B5CF6;
}

html, body, [data-testid="stApp"]{
    background:
      radial-gradient(circle at top left, rgba(139,92,246,.25), transparent 30%),
      radial-gradient(circle at top right, rgba(77,168,255,.18), transparent 30%),
      radial-gradient(circle at bottom left, rgba(255,77,109,.18), transparent 35%),
      linear-gradient(135deg,var(--bg),var(--bg2));
    color:var(--ink);
    font-family:'Inter',sans-serif;
}

#MainMenu, footer{ visibility:hidden; }

[data-testid="stSidebar"]{
    background:rgba(10,14,30,.95);
    backdrop-filter:blur(24px);
    border-right:1px solid var(--line);
}
[data-testid="stSidebar"] *{ color:var(--ink) !important; }

h1,h2,h3,h4,h5,h6{ font-family:'Space Grotesk',sans-serif !important; color:var(--ink); }
p,span,label,div{ color:var(--ink); }

.prism-card,
[data-testid="stVerticalBlockBorderWrapper"] > div,
[data-testid="stExpander"]{
    background:var(--surface) !important;
    backdrop-filter:blur(20px);
    -webkit-backdrop-filter:blur(20px);
    border:1px solid var(--line) !important;
    border-radius:20px !important;
    box-shadow:0 8px 40px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.08);
    transition:.3s ease;
}
.prism-card:hover{ transform:translateY(-4px); box-shadow:0 16px 45px rgba(139,92,246,.25); }
.prism-card.leader{ border:1px solid rgba(139,92,246,.55) !important; box-shadow:0 0 0 1px rgba(139,92,246,.25), 0 8px 40px rgba(139,92,246,.35); }

.stButton button, .stDownloadButton button{
    background:linear-gradient(135deg,#8B5CF6,#4DA8FF);
    color:white !important;
    border:none !important;
    border-radius:14px !important;
    font-family:'IBM Plex Mono',monospace;
    font-weight:600;
    padding:.7rem 1rem;
    box-shadow:0 0 25px rgba(139,92,246,.4);
    transition:.25s ease;
}
.stButton button:hover, .stDownloadButton button:hover{
    transform:translateY(-2px) scale(1.02);
    box-shadow:0 0 40px rgba(139,92,246,.7);
}

.stTextArea textarea, .stTextInput input{
    background:rgba(255,255,255,.05) !important;
    color:white !important;
    border:1px solid var(--line) !important;
    border-radius:14px !important;
}

button[data-baseweb="tab"]{ color:var(--ink-muted) !important; font-family:'IBM Plex Mono',monospace; text-transform:uppercase; }
button[data-baseweb="tab"][aria-selected="true"]{ color:#8B5CF6 !important; }
div[data-baseweb="tab-highlight"]{ background:#8B5CF6 !important; }

::-webkit-scrollbar{ width:8px; }
::-webkit-scrollbar-thumb{ background:#8B5CF6; border-radius:50px; }

.hero{
    padding:2.6rem 2.4rem;
    border-radius:24px;
    background:linear-gradient(135deg, rgba(139,92,246,.25), rgba(77,168,255,.12), rgba(255,77,109,.10));
    backdrop-filter:blur(25px);
    border:1px solid rgba(255,255,255,.08);
    margin-bottom:1.5rem;
}
.hero-eyebrow{
    font-family:'IBM Plex Mono',monospace;
    font-size:.72rem;
    letter-spacing:.18em;
    text-transform:uppercase;
    color:var(--spec-proof);
    font-weight:600;
    margin-bottom:.3rem;
}
.hero-title{
    font-size:clamp(3.2rem, 8vw, 6.8rem);
    line-height:.92;
    font-weight:800;
    letter-spacing:-.03em;
    margin:0;
    background:linear-gradient(90deg,#FF4D6D,#FF8A3D,#FFD93D,#38D39F,#4DA8FF,#8B5CF6);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    filter:drop-shadow(0 0 36px rgba(139,92,246,.4)) drop-shadow(0 0 70px rgba(255,77,109,.18));
}
.hero-subtitle{ color:var(--ink-muted); margin-top:.5rem; font-size:1.08rem; font-weight:500; letter-spacing:.01em; }

.id-tag{
    font-family:'IBM Plex Mono',monospace;
    font-size:.68rem;
    color:var(--ink-muted);
    background:var(--paper);
    border:1px solid var(--line);
    border-radius:6px;
    padding:.1rem .4rem;
}
.section-hdr{
    font-family:'IBM Plex Mono',monospace;
    font-size:.7rem;
    letter-spacing:.08em;
    text-transform:uppercase;
    color:var(--ink-muted);
    margin:.5rem 0 .4rem;
}
.chip{
    display:inline-block;
    font-family:'IBM Plex Mono',monospace;
    font-size:.72rem;
    padding:.18rem .55rem;
    border-radius:100px;
    margin:.12rem;
}
.chip.verified{ background:rgba(56,211,159,.15); color:var(--spec-domain); border:1px solid rgba(56,211,159,.35); }
.chip.unverified{ background:rgba(255,77,109,.12); color:var(--spec-skill); border:1px solid rgba(255,77,109,.30); }
.chip.skill{ background:rgba(139,92,246,.15); color:var(--spec-proof); border:1px solid rgba(139,92,246,.35); }

.readout-strip{ display:flex; gap:1.4rem; flex-wrap:wrap; margin:.6rem 0 1rem; }
.readout-cell .v{ font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:800; color:var(--ink); }
.readout-cell .l{ font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.06em; color:var(--ink-muted); text-transform:uppercase; }

.ledger-row{ padding:.6rem .9rem; margin:.4rem 0; background:var(--paper); border-radius:0 10px 10px 0; }
</style>
""", unsafe_allow_html=True)

# ── Signature graphics ────────────────────────────────────────────────────────

def prism_logo_svg(size: int = 40) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">'
        f'<line x1="2" y1="32" x2="22" y2="32" stroke="#F8FAFF" stroke-width="2.5"/>'
        f'<polygon points="22,46 22,18 46,32" fill="none" stroke="#F8FAFF" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<line x1="46" y1="32" x2="62" y2="14" stroke="#FF4D6D" stroke-width="2"/>'
        f'<line x1="46" y1="32" x2="62" y2="21" stroke="#FF8A3D" stroke-width="2"/>'
        f'<line x1="46" y1="32" x2="62" y2="27" stroke="#FFD93D" stroke-width="2"/>'
        f'<line x1="46" y1="32" x2="62" y2="33" stroke="#38D39F" stroke-width="2"/>'
        f'<line x1="46" y1="32" x2="62" y2="39" stroke="#4DA8FF" stroke-width="2"/>'
        f'<line x1="46" y1="32" x2="62" y2="46" stroke="#8B5CF6" stroke-width="2"/>'
        f'</svg>'
    )


def spectrum_rule(thick: int = 6) -> None:
    st.markdown(
        f"""
        <div style="height:{thick}px;border-radius:100px;
            background:linear-gradient(90deg,#FF4D6D 0%,#FF8A3D 20%,#FFD93D 40%,#38D39F 60%,#4DA8FF 80%,#8B5CF6 100%);
            box-shadow:0 0 8px rgba(255,77,109,.8),0 0 16px rgba(77,168,255,.7),0 0 24px rgba(139,92,246,.8);
            margin:1rem 0 1.5rem;position:relative;overflow:hidden;">
            <div style="position:absolute;top:0;left:-100%;width:100%;height:100%;
                background:linear-gradient(90deg,transparent,rgba(255,255,255,.5),transparent);
                animation:shimmer 3s infinite;"></div>
        </div>
        <style>@keyframes shimmer{{0%{{left:-100%;}}100%{{left:100%;}}}}</style>
        """,
        unsafe_allow_html=True,
    )


def spectral_signature(card: dict, height: int = 28, bar_w: int = 10, gap: int = 3) -> str:
    """One bar per rubric dimension, colored by its fixed wavelength, height by score."""
    x = 0
    bars = []
    for key, _, _, hexcolor in DIMENSIONS:
        v = max(0.0, min(1.0, float(card.get(key, 0) or 0)))
        h = max(2, round(v * height))
        y = height - h
        bars.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" rx="2" fill="{hexcolor}"/>')
        x += bar_w + gap
    total_w = x - gap
    return (
        f'<svg width="{total_w}" height="{height}" viewBox="0 0 {total_w} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle">' + "".join(bars) + "</svg>"
    )


# ── Small render helpers ──────────────────────────────────────────────────────

def section_header(text: str) -> None:
    st.markdown(f"<div class='section-hdr'>{text}</div>", unsafe_allow_html=True)


def id_tag(cid: str) -> str:
    return f"<span class='id-tag'>{cid}</span>"


def rank_mono(rank: Any, leader: bool = False) -> str:
    color = "var(--spec-proof)" if leader else "var(--ink-muted)"
    weight = 700 if leader else 600
    try:
        txt = f"№{int(rank):02d}"
    except (TypeError, ValueError):
        txt = "№??"
    return (f"<span style='font-family:IBM Plex Mono,monospace;font-size:1rem;"
            f"color:{color};font-weight:{weight}'>{txt}</span>")


def tier_meter(label: Optional[str]) -> str:
    raw = (label or "moderate").strip()
    key = raw.lower()
    n = TIER_LEVELS.get(key, 3)
    color = TIER_COLOR.get(key, "var(--ink-muted)")
    bars = "".join(
        f"<span style='display:inline-block;width:5px;height:{5 + i * 3}px;"
        f"background:{color if i < n else 'var(--line)'};margin-right:2px;border-radius:1px;"
        f"vertical-align:bottom'></span>"
        for i in range(5)
    )
    return (f"<span style='display:inline-flex;align-items:flex-end;margin-right:.5rem'>{bars}</span>"
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:.72rem;letter-spacing:.07em;"
            f"color:{color};text-transform:uppercase;font-weight:700'>{raw}</span>")


def conf_tag(conf: Optional[str]) -> str:
    raw = (conf or "medium").strip()
    color = CONF_COLOR.get(raw.lower(), "var(--ink-muted)")
    return (f"<span style='font-family:IBM Plex Mono,monospace;font-size:.7rem;letter-spacing:.05em;"
            f"color:{color};font-weight:700'>{raw.upper()}</span>")


def dim_row(label: str, value: Any, color: str) -> None:
    pct = int(max(0.0, min(1.0, float(value or 0))) * 100)
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:.7rem;margin:.3rem 0'>"
        f"<span style='width:150px;font-family:IBM Plex Mono,monospace;font-size:.72rem;"
        f"letter-spacing:.03em;color:var(--ink-muted);flex-shrink:0;text-transform:uppercase'>{label}</span>"
        f"<div style='flex:1;background:var(--line);border-radius:3px;height:8px;overflow:hidden'>"
        f"<div style='width:{pct}%;background:{color};height:100%'></div></div>"
        f"<span style='width:42px;text-align:right;font-family:IBM Plex Mono,monospace;"
        f"font-size:.78rem;font-weight:700;color:var(--ink)'>{float(value or 0):.2f}</span></div>",
        unsafe_allow_html=True,
    )


def render_weights() -> None:
    section_header("Weights applied")
    row = " &nbsp;&nbsp; ".join(
        f"<span style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:var(--ink-muted)'>"
        f"{lbl} <b style='color:var(--ink)'>{w}</b></span>"
        for _, lbl, w, _ in DIMENSIONS
    )
    st.markdown(row, unsafe_allow_html=True)


def skill_chips(items: Optional[list], variant: str, empty_note: str = "none") -> str:
    if not items:
        return f"<span style='color:var(--ink-muted);font-size:.8rem;font-family:IBM Plex Mono,monospace;font-style:italic'>{empty_note}</span>"
    return " ".join(f'<span class="chip {variant}">{s}</span>' for s in items)


def readout_strip(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f"<div class='readout-cell'><div class='v'>{val}</div><div class='l'>{lbl}</div></div>"
        for lbl, val in items
    )
    st.markdown(f"<div class='readout-strip'>{cells}</div>", unsafe_allow_html=True)


def render_ledger_rows(claims: list[dict]) -> None:
    if not claims:
        st.markdown(
            "<span style='color:var(--ink-muted);font-family:IBM Plex Mono,monospace;font-size:.85rem'>"
            "No evidence ledger entries for this candidate.</span>",
            unsafe_allow_html=True,
        )
        return
    for claim in claims:
        status_raw = (claim.get("verification_status") or "pending").strip()
        color = STATUS_COLOR.get(status_raw.lower(), "var(--ink-muted)")
        conf = claim.get("confidence", 0.0) or 0.0
        source = claim.get("source", "resume")
        snippet = claim.get("notes") or claim.get("claim_text", "")
        url = claim.get("evidence_url")
        recency = claim.get("recency_years")
        st.markdown(
            f"<div class='ledger-row' style='border-left:3px solid {color}'>"
            f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
            f"<span style='font-family:Space Grotesk,sans-serif;font-weight:600;color:var(--ink)'>"
            f"{claim.get('skill', '')}</span>"
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:.72rem;letter-spacing:.06em;"
            f"color:{color};font-weight:700;text-transform:uppercase'>{status_raw}</span></div>"
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:var(--ink-muted);margin:.3rem 0'>"
            f"confidence {conf:.2f} · source {source}"
            + (f" · {recency:.1f}y ago" if recency is not None else "")
            + "</div>"
            + (f"<div style='font-family:Source Serif 4,serif;font-size:.86rem;color:var(--ink);"
               f"margin-top:.3rem;font-style:italic'>{snippet}</div>" if snippet else "")
            + (f"<div style='margin-top:.3rem'><a href='{url}' style='color:var(--spec-proof);"
               f"font-family:IBM Plex Mono,monospace;font-size:.72rem'>{url}</a></div>" if url else "")
            + "</div>",
            unsafe_allow_html=True,
        )


def render_interview_questions(questions: list[dict]) -> None:
    if not questions:
        st.markdown(
            "<span style='color:var(--ink-muted);font-size:.85rem'>No interview questions generated.</span>",
            unsafe_allow_html=True,
        )
        return
    for q in questions:
        priority = (q.get("priority") or "medium").lower()
        color = ("var(--spec-skill)" if priority == "high" else
                 "var(--spec-seniority)" if priority == "medium" else "var(--spec-domain)")
        rationale = q.get("rationale", "")
        st.markdown(
            f"<div style='border-left:3px solid {color};padding:.5rem .9rem;margin:.4rem 0'>"
            f"<span class='chip skill'>{q.get('skill', '')}</span><br>"
            f"<span style='font-family:Source Serif 4,serif;font-size:.92rem;color:var(--ink);"
            f"font-weight:600'>{q.get('question', '')}</span>"
            + (f"<div style='font-size:.78rem;color:var(--ink-muted);font-style:italic;margin-top:.2rem'>"
               f"{rationale}</div>" if rationale else "")
            + "</div>",
            unsafe_allow_html=True,
        )


def render_flexible_items(items: list, empty_msg: str, title_keys: list[str], body_keys: list[str]) -> None:
    """
    Defensive renderer for schemas we haven't verified (fairness_flags,
    top_k_ratios). Tries a list of plausible key names for a title/body
    line; if none match, dumps every key:value pair so nothing is silently
    dropped just because a field name was guessed wrong.
    """
    if not items:
        st.markdown(f"<span style='color:var(--spec-domain);font-family:IBM Plex Mono,monospace;font-size:.85rem'>{empty_msg}</span>", unsafe_allow_html=True)
        return
    for item in items:
        if not isinstance(item, dict):
            st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:.82rem;color:var(--ink)'>{item}</div>", unsafe_allow_html=True)
            continue
        title = next((item[k] for k in title_keys if item.get(k)), None)
        body = next((item[k] for k in body_keys if item.get(k)), None)
        if title or body:
            st.markdown(
                f"<div style='border-left:3px solid var(--spec-seniority);padding:.5rem .9rem;margin:.3rem 0;"
                f"background:var(--paper);border-radius:0 8px 8px 0'>"
                + (f"<span style='font-family:IBM Plex Mono,monospace;font-weight:700;font-size:.78rem;"
                   f"color:var(--ink)'>{title}</span><br>" if title else "")
                + (f"<span style='font-family:Source Serif 4,serif;font-size:.85rem;color:var(--ink-muted)'>{body}</span>" if body else "")
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            # unknown shape — show everything rather than dropping it
            rows = "".join(f"<div><b style='color:var(--ink-muted)'>{k}</b>: {v}</div>" for k, v in item.items())
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:.78rem;color:var(--ink);"
                f"background:var(--paper);border-radius:8px;padding:.5rem .8rem;margin:.3rem 0'>{rows}</div>",
                unsafe_allow_html=True,
            )


def derive_unverified_skills(ledger_entry: Optional[dict]) -> list[str]:
    """report.json has no `unverified_skills` field — derive it from the
    evidence ledger's per-claim verification_status instead of guessing."""
    if not ledger_entry:
        return []
    seen, out = set(), []
    for c in ledger_entry.get("claims", []) or []:
        status = (c.get("verification_status") or "").lower()
        skill = c.get("skill")
        if skill and status != "verified":
            key = skill.strip().lower()
            if key not in seen:
                seen.add(key)
                out.append(skill)
    return out


# ── Roster card ────────────────────────────────────────────────────────────────

def render_roster_card(card: dict, ledger_entry: Optional[dict], expanded: bool = False) -> None:
    rank = card.get("rank", 0)
    name = card.get("candidate_name") or card.get("candidate_id", "Unknown")
    score = card.get("final_score", 0.0)
    label = card.get("score_label", "moderate")
    conf = card.get("rank_confidence", "medium")
    cid = card.get("candidate_id", "")
    leader = str(rank) == "1"

    with st.container(border=True):
        cols = st.columns([0.08, 0.46, 0.24, 0.22])
        with cols[0]:
            st.markdown(rank_mono(rank, leader=leader), unsafe_allow_html=True)
        with cols[1]:
            st.markdown(
                f"<span style='font-family:Space Grotesk,sans-serif;font-size:1.02rem;"
                f"font-weight:700;color:var(--ink)'>{name}</span> {id_tag(cid)}<br>{conf_tag(conf)}",
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(spectral_signature(card), unsafe_allow_html=True)
        with cols[3]:
            st.markdown(
                f"<div style='text-align:right'>{tier_meter(label)}<br>"
                f"<span style='font-family:IBM Plex Mono,monospace;font-size:1.15rem;"
                f"font-weight:700;color:var(--ink)'>{score:.1f}</span></div>",
                unsafe_allow_html=True,
            )

        with st.expander("Open full report", expanded=expanded):
            tab_summary, tab_scoring, tab_evidence, tab_rank, tab_interview = st.tabs(
                ["Summary", "Scoring", "Evidence", "Rank rationale", "Interview prep"]
            )
            verified = card.get("verified_claims", 0)
            unverified = card.get("unverified_claims", 0)
            total_c = verified + unverified
            proof_pct = int(verified / total_c * 100) if total_c else 0

            with tab_summary:
                readout_strip([
                    ("Score", f"{score:.0f}"),
                    ("Verified", str(verified)),
                    ("Proof ratio", f"{proof_pct}%"),
                    ("Tier", str(label).capitalize()),
                ])
                if card.get("summary"):
                    st.markdown(
                        f"<div style='font-family:Source Serif 4,serif;font-size:.92rem;color:var(--ink);"
                        f"line-height:1.7;background:var(--paper);border-radius:8px;"
                        f"padding:.9rem 1.1rem;margin:.6rem 0'>{card['summary']}</div>",
                        unsafe_allow_html=True,
                    )
                c1, c2 = st.columns(2)
                with c1:
                    section_header("Strengths")
                    for s in card.get("strengths", []):
                        st.markdown(f"<div style='font-size:.86rem;color:var(--ink);margin:.2rem 0'>+ {s}</div>", unsafe_allow_html=True)
                with c2:
                    section_header("Risks & gaps")
                    risks = card.get("risks", [])
                    if risks:
                        for r in risks:
                            st.markdown(f"<div style='font-size:.86rem;color:var(--ink);margin:.2rem 0'>− {r}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:var(--spec-domain);font-size:.85rem'>No critical risks.</span>", unsafe_allow_html=True)

            with tab_scoring:
                for key, lbl, _, color in DIMENSIONS:
                    dim_row(lbl, card.get(key, 0), color)
                st.markdown("<br>", unsafe_allow_html=True)
                render_weights()

            with tab_evidence:
                e1, e2 = st.columns(2)
                with e1:
                    section_header(f"Verified ({verified})")
                    st.markdown(skill_chips(card.get("verified_skills", [])[:10], "verified"), unsafe_allow_html=True)
                with e2:
                    unverified_skills = derive_unverified_skills(ledger_entry)
                    section_header(f"Unverified ({unverified})")
                    st.markdown(skill_chips(unverified_skills[:8], "unverified"), unsafe_allow_html=True)

            with tab_rank:
                gap = card.get("score_gap_to_next")
                why = card.get("why_above_next", "")
                if gap is not None:
                    st.markdown(
                        f"<span style='font-family:Space Grotesk,sans-serif;font-size:1.3rem;"
                        f"font-weight:800;color:var(--spec-proof)'>+{gap:.1f}</span> "
                        f"<span style='font-family:IBM Plex Mono,monospace;font-size:.78rem;"
                        f"color:var(--ink-muted)'>points above next candidate</span>",
                        unsafe_allow_html=True,
                    )
                if why:
                    st.markdown(
                        f"<div style='border-left:3px solid var(--spec-proof);padding:.7rem 1rem;"
                        f"margin-top:.6rem;font-family:Source Serif 4,serif;font-size:.88rem;"
                        f"color:var(--ink);background:var(--paper)'>{why}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<span style='color:var(--ink-muted);font-size:.85rem'>No pairwise justification available.</span>", unsafe_allow_html=True)

            with tab_interview:
                render_interview_questions(card.get("interview_questions", []))


# ── Sidebar / input ──────────────────────────────────────────────────────────

DEMO_JD = (
    "We are hiring a Backend Engineer with 3+ years of experience in Python, "
    "AWS, and building production APIs. Preferred: Docker, Kubernetes, "
    "PostgreSQL, FastAPI. The ideal candidate has shipped production systems "
    "at scale and can work independently."
)
DEMO_CANDIDATES = [
    ("DEMO_001", "Backend Engineer, 5 years experience. Built and scaled FastAPI "
     "services on AWS (EC2, RDS, Lambda) serving 2M+ requests/day. Proficient in "
     "Python, Docker, Kubernetes, PostgreSQL. Open source contributor."),
    ("DEMO_002", "Frontend Developer, 2 years experience with React and "
     "TypeScript. Some exposure to Node.js APIs. Limited backend/cloud experience."),
    ("DEMO_003", "DevOps engineer, 6 years, manages infrastructure, not "
     "application development. Backend systems & APIs. AWS AWS AWS Python "
     "Python Docker Docker Kubernetes Kubernetes."),
    ("DEMO_004", "Senior Backend Engineer, 8 years. Led a team building "
     "Python/FastAPI microservices on AWS, with PostgreSQL and Kubernetes in "
     "production. Speaks at conferences about distributed systems."),
]


def parse_pasted_candidates(text: str) -> list[tuple[str, str]]:
    """Blocks separated by a line of '---'. Optional '# Name' first line per block."""
    blocks = [b.strip() for b in text.split("---") if b.strip()]
    out = []
    for i, b in enumerate(blocks, start=1):
        lines = b.splitlines()
        cid = f"C{i:03d}"
        body = b
        if lines and lines[0].strip().startswith("#"):
            label = lines[0].strip().lstrip("#").strip()
            if label:
                cid = label
            body = "\n".join(lines[1:]).strip()
        out.append((cid, body))
    return out


def render_sidebar() -> tuple:
    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center;padding:.4rem 0 1rem'>"
            f"{prism_logo_svg(46)}"
            f"<div style='font-family:Space Grotesk,sans-serif;font-size:1.3rem;font-weight:800;"
            f"color:var(--ink);margin-top:.4rem'>PRISM</div>"
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.6rem;letter-spacing:.08em;"
            f"color:var(--spec-proof);text-transform:uppercase;margin-top:.15rem;line-height:1.4'>"
            f"Proof-driven Ranking &amp;<br>Intelligent Selection Model</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        spectrum_rule(3)

        mode = st.radio("Input", ["Built-in demo", "Paste your own"], label_visibility="collapsed")

        jd_text = DEMO_JD
        candidate_texts = DEMO_CANDIDATES

        if mode == "Built-in demo":
            st.markdown(
                "<div style='font-family:Source Serif 4,serif;font-size:.82rem;color:var(--ink-muted);"
                "margin:.4rem 0 .8rem'>4 synthetic candidates, including one keyword-stuffer — "
                "useful for showing the guard layers in action.</div>",
                unsafe_allow_html=True,
            )
        else:
            jd_text = st.text_area("Job description", placeholder="Paste the full job description here…", height=150)
            cand_raw = st.text_area(
                "Candidates",
                placeholder="# Optional Name\nResume text for candidate 1...\n---\n# Another Name\nResume text for candidate 2...",
                height=220,
                help="Separate each candidate with a line containing only ---. Optional '# Name' as the first line of a block.",
            )
            files = st.file_uploader("Or upload .txt / .md resumes", type=["txt", "md"], accept_multiple_files=True)
            if files:
                candidate_texts = [(Path(f.name).stem, f.getvalue().decode("utf-8", errors="ignore")) for f in files]
            elif cand_raw.strip():
                candidate_texts = parse_pasted_candidates(cand_raw)
            else:
                candidate_texts = []

        st.divider()
        section_header("Pipeline options")
        force_fallback = not st.toggle(
            "Use LLM (requires API key)", value=False,
            help="When off, the pipeline runs fully offline with rule-based fallbacks.",
        )
        shortlist_k = st.slider("Retrieval shortlist size", 5, 50, 10, 5)
        stab_runs = st.slider("Stability test runs", 3, 10, 3, 1)

        st.divider()
        run_btn = st.button("Run pipeline", type="primary", use_container_width=True)

    return jd_text, candidate_texts, force_fallback, shortlist_k, stab_runs, run_btn


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _load_json(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def run_pipeline(jd_text: str, candidate_texts: list[tuple[str, str]], force_fallback: bool, k: int, stability_runs: int) -> dict:
    from ai_hiring_ranker.evaluation import run_pipeline_from_texts

    result = run_pipeline_from_texts(
        jd_text=jd_text,
        candidate_texts=candidate_texts,
        output_dir=str(ROOT / "outputs"),
        force_fallback=force_fallback,
        k=k,
        stability_runs=stability_runs,
    )
    manifest = result.to_manifest_dict()
    files = result.manifest.to_dict()
    return {
        "manifest": manifest,
        "files": files,
        "report": _load_json(files.get("report_json")),
        "audit": _load_json(files.get("audit_json")),
        "ledger": _load_json(files.get("ledger_json")),
    }


def _ledger_map(ledger: Optional[dict]) -> dict[str, dict]:
    if not ledger:
        return {}
    return {c["candidate_id"]: c for c in ledger.get("candidates", [])}


def _stability_map(audit: Optional[dict]) -> dict[str, dict]:
    if not audit:
        return {}
    return {row["candidate_id"]: row for row in audit.get("candidate_stability", [])}


# ── Tab: Overview ────────────────────────────────────────────────────────────

def render_ranking_row(row: dict) -> None:
    rank = row.get("rank", 0)
    name = row.get("candidate_name") or row.get("candidate_id", "")
    score = row.get("final_score", 0.0)
    label = row.get("score_label", "moderate")
    cid = row.get("candidate_id", "")
    leader = str(rank) == "1"

    cols = st.columns([0.07, 0.27, 0.14, 0.22, 0.3])
    with cols[0]:
        st.markdown(rank_mono(rank, leader=leader), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(
            f"<span style='font-family:Space Grotesk,sans-serif;font-weight:600;color:var(--ink)'>{name}</span>"
            f"<br>{id_tag(cid)}",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f"<span style='font-family:IBM Plex Mono,monospace;font-weight:700;font-size:1.05rem;"
            f"color:var(--ink)'>{score:.1f}</span>",
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(spectral_signature(row), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(tier_meter(label), unsafe_allow_html=True)
    st.markdown("<div style='border-bottom:1px solid var(--line);margin:.3rem 0 .6rem'></div>", unsafe_allow_html=True)


def render_overview_tab(manifest: dict, candidates: list[dict]) -> None:
    n = len(candidates)
    top_score = candidates[0]["final_score"] if candidates else 0
    verified_avg = sum(c.get("verified_claims", 0) for c in candidates) / n if n else 0
    fairness = (manifest.get("has_fairness_flags") and "FLAGGED") or "CLEAR"

    st.markdown("<br>", unsafe_allow_html=True)
    readout_strip([
        ("Candidates", str(n)),
        ("Top score", f"{top_score:.0f}"),
        ("Avg verified claims", f"{verified_avg:.1f}"),
        ("Fairness", fairness),
        ("Run time", f"{manifest.get('duration_s', 0):.1f}s"),
    ])

    warnings = manifest.get("warnings") or []
    if warnings:
        with st.expander(f"{len(warnings)} pipeline warning(s)"):
            for w in warnings:
                st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:.8rem;color:var(--spec-seniority)'>· {w}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Ranking")
    spectrum_rule(2)
    for c in candidates[:15]:
        render_ranking_row(c)


# ── Tab: Roster ───────────────────────────────────────────────────────────────

def render_roster_tab(candidates: list[dict], ledgers: dict[str, dict]) -> None:
    st.markdown(
        "<div style='font-family:Source Serif 4,serif;font-size:.86rem;color:var(--ink-muted);"
        "margin-bottom:.8rem'>Expand any candidate for the full report, or use the "
        "<b>Candidate</b> tab for a single-candidate deep dive.</div>",
        unsafe_allow_html=True,
    )
    for c in candidates:
        render_roster_card(c, ledgers.get(c.get("candidate_id", "")), expanded=False)


# ── Tab: Candidate ────────────────────────────────────────────────────────────

def render_candidate_deep_dive(card: dict, ledger_entry: Optional[dict], stability: Optional[dict], audit: Optional[dict]) -> None:
    cid = card.get("candidate_id", "")
    name = card.get("candidate_name") or cid
    rank = card.get("rank", "?")
    score = card.get("final_score", 0.0)
    label = card.get("score_label", "moderate")
    conf = card.get("rank_confidence", "medium")
    leader = str(rank) == "1"

    st.markdown(
        f"<div class='prism-card{' leader' if leader else ''}' style='display:flex;align-items:center;gap:1.1rem;padding:1.1rem 1.3rem'>"
        f"{rank_mono(rank, leader=leader)}"
        f"<div style='flex:1'>"
        f"<div style='font-family:Space Grotesk,sans-serif;font-size:1.4rem;font-weight:800;color:var(--ink)'>{name}</div>"
        f"<div style='margin-top:.2rem'>{id_tag(cid)} &nbsp; {conf_tag(conf)}</div>"
        f"</div>"
        f"<div style='text-align:right'>{tier_meter(label)}<div style='font-family:IBM Plex Mono,monospace;"
        f"font-size:1.5rem;font-weight:800;color:var(--ink);margin-top:.2rem'>{score:.1f}</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    tab_summary, tab_ledger, tab_scoring, tab_rank, tab_interview, tab_audit = st.tabs(
        ["Summary", "Evidence ledger", "Scoring", "Rank rationale", "Interview prep", "Audit flags"]
    )

    with tab_summary:
        if card.get("summary"):
            st.markdown(
                f"<div style='font-family:Source Serif 4,serif;font-size:.92rem;color:var(--ink);"
                f"line-height:1.8;background:var(--paper);border-radius:8px;padding:1rem 1.2rem'>{card['summary']}</div>",
                unsafe_allow_html=True,
            )
        c1, c2 = st.columns(2)
        with c1:
            section_header("Strengths")
            for s in card.get("strengths", []):
                st.markdown(f"<div style='font-size:.88rem;color:var(--ink);margin:.25rem 0'>+ {s}</div>", unsafe_allow_html=True)
        with c2:
            section_header("Risks & gaps")
            risks = card.get("risks", [])
            if risks:
                for r in risks:
                    st.markdown(f"<div style='font-size:.88rem;color:var(--ink);margin:.25rem 0'>− {r}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:var(--spec-domain);font-size:.86rem'>No critical risks flagged.</span>", unsafe_allow_html=True)

        section_header("Verified vs unverified skills")
        unverified_skills = derive_unverified_skills(ledger_entry)
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(skill_chips(card.get("verified_skills", []), "verified"), unsafe_allow_html=True)
        with s2:
            st.markdown(skill_chips(unverified_skills, "unverified"), unsafe_allow_html=True)

    with tab_ledger:
        section_header("Claim verification")
        claims = ledger_entry.get("claims", []) if ledger_entry else []
        verified = sum(1 for c in claims if (c.get("verification_status") or "").lower() == "verified")
        readout_strip([
            ("Total claims", str(len(claims))),
            ("Verified", str(verified)),
            ("Proof strength", f"{card.get('proof_strength', 0):.2f}"),
        ])
        render_ledger_rows(claims)

    with tab_scoring:
        section_header("Rubric dimensions")
        for key, lbl, _, color in DIMENSIONS:
            dim_row(lbl, card.get(key, 0), color)
        st.markdown("<br>", unsafe_allow_html=True)
        render_weights()
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Score evidence")
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.85rem;color:var(--ink);line-height:1.9'>"
            f"verified claims — {card.get('verified_claims', 0)}<br>"
            f"unverified claims — {card.get('unverified_claims', 0)}<br>"
            f"final score — {score:.1f}/100 ({label})</div>",
            unsafe_allow_html=True,
        )

    with tab_rank:
        gap = card.get("score_gap_to_next")
        if gap is not None:
            st.markdown(
                f"<span style='font-family:IBM Plex Mono,monospace;font-size:.78rem;color:var(--ink-muted)'>SCORE GAP TO NEXT</span><br>"
                f"<span style='font-family:Space Grotesk,sans-serif;font-size:1.6rem;font-weight:800;color:var(--spec-proof)'>+{gap:.1f} pts</span>",
                unsafe_allow_html=True,
            )
        why = card.get("why_above_next", "")
        if why:
            st.markdown(
                f"<div style='border-left:3px solid var(--spec-proof);padding:1rem 1.2rem;margin-top:.8rem;"
                f"background:var(--paper);font-family:Source Serif 4,serif;font-size:.9rem;color:var(--ink)'>{why}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<span style='color:var(--ink-muted);font-size:.85rem'>No pairwise rank justification available.</span>", unsafe_allow_html=True)

        if stability:
            stable = stability.get("is_stable", True)
            color = "var(--spec-domain)" if stable else "var(--spec-skill)"
            base_rank = stability.get("base_rank", "?")
            rank_range = stability.get("rank_range", 0)
            st.markdown(
                f"<div style='margin-top:.9rem;font-family:IBM Plex Mono,monospace;font-size:.82rem;color:var(--ink)'>"
                f"<span style='color:{color};font-weight:700'>{'STABLE' if stable else 'UNSTABLE'}</span> &nbsp; "
                f"base rank {base_rank} (±{rank_range}), σ={stability.get('score_std', 0):.3f}</div>",
                unsafe_allow_html=True,
            )

    with tab_interview:
        render_interview_questions(card.get("interview_questions", []))

    with tab_audit:
        flags = []
        if audit:
            for flag in audit.get("fairness_flags", []) or []:
                affected = flag.get("affected_ids") or flag.get("candidate_ids") or []
                if not affected or cid in affected:
                    flags.append(flag)
            if cid in (audit.get("unstable_candidates") or []):
                flags.append({"description": "Rank is unstable across perturbation runs.", "severity": "warning"})
        if flags:
            render_flexible_items(flags, "", title_keys=["proxy_field", "field", "title"], body_keys=["description", "message", "detail"])
        else:
            st.markdown("<span style='color:var(--spec-domain);font-size:.85rem'>No audit flags for this candidate.</span>", unsafe_allow_html=True)


def render_candidate_tab(candidates: list[dict], ledgers: dict[str, dict], stability: dict[str, dict], audit: Optional[dict]) -> None:
    if not candidates:
        st.markdown("<span style='color:var(--ink-muted)'>Run the pipeline first to generate candidate reports.</span>", unsafe_allow_html=True)
        return

    options = [
        f"№{int(c.get('rank', 0)):02d} — {c.get('candidate_name') or c['candidate_id']} ({c.get('final_score', 0):.0f} pts)"
        for c in candidates
    ]
    selected = st.selectbox("Select candidate", options, label_visibility="collapsed")
    idx = options.index(selected)
    card = candidates[idx]
    cid = card.get("candidate_id", "")

    render_candidate_deep_dive(card, ledgers.get(cid), stability.get(cid), audit)


# ── Tab: Fairness & stability ─────────────────────────────────────────────────

def render_fairness_tab(audit: Optional[dict]) -> None:
    if not audit:
        st.markdown("<span style='font-family:IBM Plex Mono,monospace;color:var(--ink-muted)'>No audit report available for this run.</span>", unsafe_allow_html=True)
        return

    st.markdown("<br>", unsafe_allow_html=True)
    readout_strip([
        ("Fairness risk", str(audit.get("fairness_risk_level", "info")).upper()),
        ("Stability ratio", f"{audit.get('stability_ratio', 0):.0%}"),
        ("Unstable", str(len(audit.get("unstable_candidates", []) or []))),
    ])

    section_header("Fairness flags")
    render_flexible_items(
        audit.get("fairness_flags", []) or [], "No fairness proxy flags detected.",
        title_keys=["proxy_field", "field", "attribute", "title"],
        body_keys=["description", "message", "detail", "recommendation"],
    )

    section_header("Top-k impact ratios")
    render_flexible_items(
        audit.get("top_k_ratios", []) or [], "No top-k impact ratios reported.",
        title_keys=["attribute", "field"],
        body_keys=["summary", "description"],
    )

    section_header("Rank stability by candidate")
    for row in audit.get("candidate_stability", []) or []:
        stable = row.get("is_stable", True)
        color = "var(--spec-domain)" if stable else "var(--spec-skill)"
        mark = "STABLE" if stable else "UNSTABLE"
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.8rem;color:var(--ink);margin:.2rem 0'>"
            f"<span style='color:{color};font-weight:700'>{mark}</span> &nbsp; {row.get('candidate_id')} — "
            f"base rank {row.get('base_rank')} (±{row.get('rank_range', 0)}), σ={row.get('score_std', 0):.3f}</div>",
            unsafe_allow_html=True,
        )


# ── Tab: Pipeline ─────────────────────────────────────────────────────────────

def render_pipeline_tab(manifest: dict) -> None:
    section_header("Layer execution log")
    for layer in manifest.get("layer_records", []) or []:
        status = (layer.get("status") or "pending").lower()
        color = ("var(--spec-domain)" if status == "complete" else
                 "var(--spec-seniority)" if status == "skipped" else
                 "var(--spec-skill)" if status == "error" else "var(--ink-muted)")
        try:
            layer_no = int(layer.get("layer", 0))
        except (TypeError, ValueError):
            layer_no = 0
        st.markdown(
            f"<div style='display:flex;gap:.8rem;align-items:baseline;padding:.5rem 0;border-bottom:1px solid var(--line)'>"
            f"<span style='font-family:IBM Plex Mono,monospace;font-weight:700;color:var(--ink-muted);width:34px'>{layer_no:02d}</span>"
            f"<span style='font-family:Space Grotesk,sans-serif;font-weight:600;color:var(--ink);flex:1'>{layer.get('name')}</span>"
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:{color};font-weight:700;letter-spacing:.05em'>{status.upper()}</span>"
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:var(--ink-muted);width:55px;text-align:right'>{layer.get('duration', 0):.2f}s</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if layer.get("notes"):
            st.markdown(f"<div style='font-family:Source Serif 4,serif;font-size:.82rem;color:var(--ink-muted);margin:.15rem 0 .4rem 44px'>{layer.get('notes')}</div>", unsafe_allow_html=True)
        if layer.get("error"):
            st.markdown(f"<div style='color:var(--spec-skill);font-family:IBM Plex Mono,monospace;font-size:.78rem;margin-left:44px'>{layer['error']}</div>", unsafe_allow_html=True)


# ── Tab: Export ────────────────────────────────────────────────────────────────

def render_export_tab(files: dict) -> None:
    labels = {
        "ranked_json": "Ranked output",
        "report_json": "Recruiter report (JSON)",
        "report_md": "Recruiter report (Markdown)",
        "audit_json": "Audit report",
        "ledger_json": "Evidence ledger",
        "manifest_json": "Run manifest",
    }
    section_header("Export run artefacts")
    cols = st.columns(2)
    i = 0
    for key, label in labels.items():
        path = files.get(key)
        if not path or not Path(path).exists():
            continue
        data = Path(path).read_bytes()
        with cols[i % 2]:
            st.markdown(
                f"<div class='prism-card' style='padding:.8rem 1rem;margin-bottom:.6rem'>"
                f"<div style='font-family:Space Grotesk,sans-serif;font-weight:600;color:var(--ink)'>{label}</div>"
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:.7rem;color:var(--ink-muted);margin:.2rem 0 0'>{Path(path).name}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.download_button(
                label="Download", data=data, file_name=Path(path).name,
                mime="application/json" if path.endswith(".json") else "text/markdown",
                use_container_width=True, key=f"dl_{key}",
            )
        i += 1


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div style="display:flex;align-items:center;gap:1.5rem;">
                <div>{prism_logo_svg(65)}</div>
                <div>
                    <div class="hero-eyebrow">Hiring Intelligence · Verity Ranker</div>
                    <div class="hero-title">PRISM</div>
                    <div class="hero-subtitle">Proof-driven Ranking &amp; Intelligent Selection Model</div>
                    <div style="margin-top:.6rem;font-family:'IBM Plex Mono', monospace;font-size:.85rem;color:var(--ink-muted);letter-spacing:.05em;">
                        ✦ Evidence-backed Candidate Ranking <br>
                        ✦ Multi-layer Intelligent Hiring Pipeline
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    spectrum_rule(6)

    jd_text, candidate_texts, force_fallback, shortlist_k, stab_runs, run_btn = render_sidebar()

    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None

    if run_btn:
        if not jd_text.strip():
            st.error("Paste a job description first.")
            st.stop()
        if not candidate_texts:
            st.error("Add at least one candidate (paste, upload, or switch to Built-in demo).")
            st.stop()
        try:
            with st.spinner("Running the pipeline…"):
                st.session_state.pipeline_result = run_pipeline(jd_text, candidate_texts, force_fallback, shortlist_k, stab_runs)
            st.success("Pipeline complete.")
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

    result = st.session_state.pipeline_result
    if result is None:
        st.markdown(
            "<div class='prism-card' style='text-align:center;padding:2.6rem 1.5rem'>"
            "<div style='font-family:Space Grotesk,sans-serif;font-size:1.15rem;font-weight:700;color:var(--ink)'>Nothing to refract yet</div>"
            "<div style='font-family:Source Serif 4,serif;font-size:.88rem;color:var(--ink-muted);margin-top:.5rem'>"
            "Choose the built-in demo or paste your own data, then click <b>Run pipeline</b> in the sidebar.</div></div>",
            unsafe_allow_html=True,
        )
        return

    manifest = result["manifest"]
    report = result.get("report") or {}
    audit = result.get("audit")
    ledger = result.get("ledger")
    candidates = sorted(report.get("candidates", []), key=lambda c: c.get("rank", 9999))
    ledgers = _ledger_map(ledger)
    stability = _stability_map(audit)

    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:.78rem;color:var(--ink-muted)'>"
        f"run {manifest.get('run_id', '')} · {manifest.get('job_title', 'Job')[:60]} · "
        f"{manifest.get('candidate_count', len(candidates))} candidates · "
        f"{manifest.get('duration_s', 0):.1f}s</div>",
        unsafe_allow_html=True,
    )

    tab_overview, tab_roster, tab_candidate, tab_fairness, tab_pipeline, tab_export = st.tabs(
        ["Overview", "Roster", "Candidate", "Fairness & stability", "Pipeline", "Export"]
    )
    with tab_overview:
        render_overview_tab(manifest, candidates)
    with tab_roster:
        render_roster_tab(candidates, ledgers)
    with tab_candidate:
        render_candidate_tab(candidates, ledgers, stability, audit)
    with tab_fairness:
        render_fairness_tab(audit)
    with tab_pipeline:
        render_pipeline_tab(manifest)
    with tab_export:
        render_export_tab(result["files"])


if __name__ == "__main__":
    main()