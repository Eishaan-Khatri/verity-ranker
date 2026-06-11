"""
Tests for Layer 1 — Input Layer (ai_hiring_ranker.ingestion).

Covers:
- JDInput and CandidateInput Pydantic validation
- Plain-text file parsing
- Link extraction (GitHub, Kaggle, LinkedIn, generic)
- Full ingest() round-trip against the sample data
- Error handling for unsupported formats and missing files
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make src importable without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_hiring_ranker.ingestion.schemas import (
    CandidateInput,
    FileFormat,
    JDInput,
    PortfolioLinks,
)
from ai_hiring_ranker.ingestion.link_extractor import extract_portfolio_links
from ai_hiring_ranker.ingestion.parsers import extract_text
from ai_hiring_ranker.ingestion.loader import ingest, load_jd, load_candidate


SAMPLE_JD = ROOT / "data" / "sample" / "jd.txt"
SAMPLE_CANDIDATES = ROOT / "data" / "sample" / "candidates"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestJDInputSchema:
    def test_valid_jd(self):
        jd = JDInput(raw_text="We need a Python developer with ML experience and FastAPI skills.")
        assert jd.word_count > 0
        assert jd.char_count > 0
        assert jd.file_format == FileFormat.TXT

    def test_too_short_raises(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            JDInput(raw_text="Too short.")

    def test_whitespace_stripped(self):
        jd = JDInput(raw_text="  " + "We need a Python developer with machine learning experience for building APIs.  ")
        assert not jd.raw_text.startswith(" ")
        assert not jd.raw_text.endswith(" ")


class TestCandidateInputSchema:
    def test_valid_candidate(self):
        c = CandidateInput(
            candidate_id="C001",
            raw_text="Experienced ML engineer with Python, FastAPI, Docker, and model evaluation skills.",
        )
        assert c.candidate_id == "C001"
        assert c.word_count > 0

    def test_empty_id_raises(self):
        with pytest.raises(Exception):
            CandidateInput(
                candidate_id="",
                raw_text="Valid resume text with enough content here.",
            )

    def test_too_short_raises(self):
        with pytest.raises(Exception):
            CandidateInput(candidate_id="X", raw_text="Short.")


class TestPortfolioLinks:
    def test_has_any_false_when_empty(self):
        pl = PortfolioLinks()
        assert not pl.has_any

    def test_has_any_true_with_github(self):
        pl = PortfolioLinks(github="https://github.com/user")
        assert pl.has_any


# ---------------------------------------------------------------------------
# Link extraction
# ---------------------------------------------------------------------------


class TestLinkExtractor:
    def test_github_extracted(self):
        text = "Check my work at github.com/johndoe"
        links = extract_portfolio_links(text)
        assert links.github is not None
        assert "github.com/johndoe" in links.github

    def test_kaggle_extracted(self):
        text = "Kaggle profile: https://www.kaggle.com/janedoe"
        links = extract_portfolio_links(text)
        assert links.kaggle is not None
        assert "kaggle.com/janedoe" in links.kaggle

    def test_linkedin_extracted(self):
        text = "Connect with me: linkedin.com/in/alice-smith"
        links = extract_portfolio_links(text)
        assert links.linkedin is not None
        assert "alice-smith" in links.linkedin

    def test_generic_portfolio_url(self):
        text = "Portfolio: https://alicesmith.dev/projects"
        links = extract_portfolio_links(text)
        assert any("alicesmith.dev" in u for u in links.portfolio)

    def test_no_links(self):
        text = "I am a senior ML engineer with 5 years of Python experience."
        links = extract_portfolio_links(text)
        assert not links.has_any

    def test_github_not_in_portfolio(self):
        text = "github.com/testuser and https://mysite.com"
        links = extract_portfolio_links(text)
        assert links.github is not None
        assert not any("github.com" in u for u in links.portfolio)

    def test_normalise_adds_https(self):
        text = "github.com/normtest"
        links = extract_portfolio_links(text)
        assert links.github.startswith("https://")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_txt_file(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("Hello world from a text file.", encoding="utf-8")
        assert extract_text(p) == "Hello world from a text file."

    def test_md_file(self, tmp_path):
        p = tmp_path / "test.md"
        p.write_text("# Title\n\nSome content.", encoding="utf-8")
        text = extract_text(p)
        assert "Title" in text

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_text(tmp_path / "nonexistent.txt")

    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "test.xlsx"
        p.write_bytes(b"fake content")
        with pytest.raises(ValueError, match="Unsupported file format"):
            extract_text(p)


# ---------------------------------------------------------------------------
# Full ingest() round-trip
# ---------------------------------------------------------------------------


class TestIngest:
    def test_sample_data_ingest(self):
        result = ingest(jd_path=SAMPLE_JD, candidates_dir=SAMPLE_CANDIDATES)
        assert result.jd.word_count > 10
        assert result.candidate_count >= 1
        assert not result.has_errors

    def test_jd_has_correct_format(self):
        result = ingest(jd_path=SAMPLE_JD, candidates_dir=SAMPLE_CANDIDATES)
        assert result.jd.file_format == FileFormat.TXT

    def test_candidate_ids_unique(self):
        result = ingest(jd_path=SAMPLE_JD, candidates_dir=SAMPLE_CANDIDATES)
        ids = [c.candidate_id for c in result.candidates]
        assert len(ids) == len(set(ids)), "Duplicate candidate IDs detected"

    def test_summary_runs(self):
        result = ingest(jd_path=SAMPLE_JD, candidates_dir=SAMPLE_CANDIDATES)
        summary = result.summary()
        assert "JD ingested" in summary
        assert "Candidates ingested" in summary

    def test_in_memory_jd(self):
        jd_text = "We need a Python developer with machine learning and model evaluation skills for production API work."
        result = ingest(
            jd_path=SAMPLE_JD,  # ignored because jd_text is provided
            candidates_dir=SAMPLE_CANDIDATES,
            jd_text=jd_text,
        )
        assert result.jd.raw_text == jd_text

    def test_in_memory_candidates(self):
        candidate_texts = [
            ("resume_a.txt", "Name: Alice\nSkills: Python, ML, FastAPI, Docker, model evaluation, production engineering"),
            ("resume_b.txt", "Candidate ID: B002\nName: Bob\nExperienced data engineer with SQL, Python, and Spark skills."),
        ]
        result = ingest(
            jd_path=SAMPLE_JD,
            candidates_dir=SAMPLE_CANDIDATES,
            candidate_texts=candidate_texts,
        )
        assert result.candidate_count == 2
        assert any(c.candidate_id == "B002" for c in result.candidates)

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No supported resume files"):
            ingest(jd_path=SAMPLE_JD, candidates_dir=tmp_path)

    def test_bad_candidate_collected_not_raised(self, tmp_path):
        """A candidate file with invalid content errors gracefully; others still load."""
        # Write one valid candidate
        (tmp_path / "good.txt").write_text(
            "Candidate ID: GOOD\nPython engineer with ML and FastAPI and production deployment skills.", encoding="utf-8"
        )
        # Write a .txt file with content too short to pass CandidateInput validation
        (tmp_path / "bad.txt").write_text("Hi.", encoding="utf-8")
        result = ingest(jd_path=SAMPLE_JD, candidates_dir=tmp_path)
        # good.txt loads fine
        assert result.candidate_count == 1
        # bad.txt validation error is captured
        assert any("bad.txt" in err for err in result.errors)

    def test_load_jd_file(self):
        jd = load_jd(SAMPLE_JD)
        assert jd.word_count > 0
        assert "Python" in jd.raw_text or "machine" in jd.raw_text.lower()

    def test_load_candidate_file(self):
        path = SAMPLE_CANDIDATES / "C001.txt"
        candidate = load_candidate(path)
        assert candidate.candidate_id == "C001"
        assert candidate.word_count > 0
