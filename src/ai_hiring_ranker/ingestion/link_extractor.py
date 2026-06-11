"""
Link extractor — scans raw resume text for GitHub, Kaggle, LinkedIn,
and general portfolio / project URLs.

Design decisions:
- All matching is done with compiled regex, no HTTP requests.
- Returns a PortfolioLinks instance ready for the CandidateInput schema.
- Duplicate URLs are deduplicated; order is preserved.
"""

from __future__ import annotations

import re
from typing import Optional

from .schemas import PortfolioLinks


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Match github.com profile or repo URLs (http/https optional)
_GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)?",
    re.IGNORECASE,
)

# Match kaggle.com profile URLs
_KAGGLE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?kaggle\.com/[A-Za-z0-9_.\-]+",
    re.IGNORECASE,
)

# Match linkedin.com/in/ profile URLs
_LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_.\-]+/?",
    re.IGNORECASE,
)

# Generic URL pattern to catch personal sites, Hugging Face, GitLab, etc.
# Deliberately conservative to avoid matching version numbers like "3.10".
_GENERIC_URL_RE = re.compile(
    r"https?://[A-Za-z0-9_.\-]+\.[a-z]{2,}(?:/[^\s,;\"'<>()]*)?",
    re.IGNORECASE,
)

# Domains already captured by the specific patterns above
_KNOWN_DOMAINS = {"github.com", "kaggle.com", "linkedin.com", "www.github.com", "www.kaggle.com", "www.linkedin.com"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_url(url: str) -> str:
    """Ensure URL has a scheme so it is unambiguous downstream."""
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url


def _first_match(pattern: re.Pattern, text: str) -> Optional[str]:
    match = pattern.search(text)
    return _normalise_url(match.group(0)) if match else None


def _all_matches(pattern: re.Pattern, text: str) -> list[str]:
    return [_normalise_url(m) for m in pattern.findall(text)]


def _domain_of(url: str) -> str:
    """Extract bare domain (lowercase, no www) from a URL string."""
    url = url.lower().removeprefix("https://").removeprefix("http://")
    return url.split("/")[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_portfolio_links(text: str) -> PortfolioLinks:
    """Scan *text* and return a :class:`PortfolioLinks` instance.

    Strategy:
    1. Try specific patterns for GitHub, Kaggle, LinkedIn first.
    2. Sweep remaining generic HTTPS URLs for anything not already captured.
    3. Deduplicate while preserving first-seen order.
    """
    github = _first_match(_GITHUB_RE, text)
    kaggle = _first_match(_KAGGLE_RE, text)
    linkedin = _first_match(_LINKEDIN_RE, text)

    # Gather all generic URLs and exclude ones already captured above
    all_generic = _all_matches(_GENERIC_URL_RE, text)
    seen: set[str] = set()
    portfolio: list[str] = []
    for url in all_generic:
        domain = _domain_of(url)
        if domain in _KNOWN_DOMAINS:
            continue
        if url in seen:
            continue
        seen.add(url)
        portfolio.append(url)

    return PortfolioLinks(
        github=github,
        kaggle=kaggle,
        linkedin=linkedin,
        portfolio=portfolio,
    )
