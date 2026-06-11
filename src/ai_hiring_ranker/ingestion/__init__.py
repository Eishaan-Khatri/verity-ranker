"""Input Layer — load, parse, and validate all inputs before the pipeline begins."""

from .loader import ingest, IngestResult
from .parsers import parse_text_file, parse_pdf, parse_docx, extract_text
from .schemas import JDInput, CandidateInput, PortfolioLinks

__all__ = [
    "ingest",
    "IngestResult",
    "parse_text_file",
    "parse_pdf",
    "parse_docx",
    "extract_text",
    "JDInput",
    "CandidateInput",
    "PortfolioLinks",
]
