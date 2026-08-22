"""Optional Google Fact Check Tools (ClaimReview) lookup.

Queries Google's free Fact Check Tools API for professional fact-check
verdicts (Alt News, BOOM, PolitiFact, Snopes, Vishvas News, …). The API key
is free; when it is not configured the whole feature is skipped silently and
the ordinary news-evidence pipeline still works.

API reference: https://developers.google.com/fact-check/tools/api
"""

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests
import streamlit as st


FACT_CHECK_SEARCH_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
MAX_CLAIM_CHARACTERS = 200
MAX_RESULTS = 5
REQUEST_TIMEOUT = (5, 15)

# Ratings that mark a claim as refuted by the fact-checker.
_CONTRADICT_RATINGS = (
    "false", "fake", "misleading", "pants on fire", "mostly false",
    "fabricated", "doctored", "manipulated", "altered", "jhooth", "jhuthi",
    "farzi", "nakli", "galat",
)
# Ratings that mark a claim as confirmed.
_SUPPORT_RATINGS = ("true", "real", "sahi", "correct", "accurate")


class FactCheckError(RuntimeError):
    """Raised when the fact-check lookup cannot be completed (never fatal)."""


@dataclass(frozen=True)
class FactCheckVerdict:
    """One professional fact-check result from the ClaimReview database."""

    claim_text: str
    rating: str
    title: str
    url: str
    publisher: str
    stance: str  # SUPPORTS | CONTRADICTS | NEUTRAL


def is_configured(api_key: str | None) -> bool:
    """Return whether a non-empty Fact Check API key was supplied."""

    return bool(api_key and api_key.strip())


def classify_rating(textual_rating: str) -> str:
    """Map a fact-checker's textual rating onto a stance label."""

    rating = (textual_rating or "").lower()
    # Mixed verdicts like PolitiFact's "Half True" are deliberately neutral —
    # a bare 'true' substring must not turn them into support.
    if "half" in rating and "true" in rating:
        return "NEUTRAL"
    if any(marker in rating for marker in _CONTRADICT_RATINGS):
        return "CONTRADICTS"
    if any(marker in rating for marker in _SUPPORT_RATINGS):
        return "SUPPORTS"
    return "NEUTRAL"


def parse_fact_check_response(data: dict[str, Any]) -> tuple[FactCheckVerdict, ...]:
    """Convert a claims:search JSON payload into display-safe verdicts."""

    verdicts: list[FactCheckVerdict] = []
    seen_urls: set[str] = set()

    for claim in data.get("claims", []):
        claim_text = str(claim.get("text", "")).strip()
        for review in claim.get("claimReview", []):
            url = str(review.get("url", "")).strip()
            parsed = urlparse(url)
            if not url or parsed.scheme not in {"http", "https"}:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)

            rating = " ".join(str(review.get("textualRating", "")).split())
            publisher_block = review.get("publisher", {}) or {}
            publisher = str(publisher_block.get("name", "")).strip() or parsed.netloc
            title = " ".join(str(review.get("title", "")).split())

            verdicts.append(
                FactCheckVerdict(
                    claim_text=claim_text[:300],
                    rating=rating or "Unrated",
                    title=title or "Fact-check article",
                    url=url,
                    publisher=publisher,
                    stance=classify_rating(rating),
                )
            )
            if len(verdicts) >= MAX_RESULTS:
                return tuple(verdicts)

    return tuple(verdicts)


@st.cache_data(ttl=3_600, show_spinner=False)
def _cached_search(claim: str, _api_key: str) -> tuple[FactCheckVerdict, ...]:
    """Call claims:search once per claim per hour."""

    try:
        response = requests.get(
            FACT_CHECK_SEARCH_URL,
            params={
                "query": claim[:MAX_CLAIM_CHARACTERS],
                "key": _api_key,
                "languageCode": "en",
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as error:
        raise FactCheckError("Fact-check lookup could not reach Google.") from error

    if response.status_code == 403:
        raise FactCheckError("Fact Check API key was rejected (403).")
    if response.status_code == 429:
        raise FactCheckError("Fact Check API quota exceeded for today.")
    if not response.ok:
        raise FactCheckError("Fact-check lookup failed.")

    try:
        return parse_fact_check_response(response.json())
    except ValueError as error:
        raise FactCheckError("Fact-check lookup returned unreadable data.") from error


def fetch_fact_checks(claim: str, api_key: str | None) -> tuple[FactCheckVerdict, ...]:
    """Return professional fact-check verdicts, or () when unavailable."""

    if not is_configured(api_key):
        return ()
    try:
        return _cached_search(" ".join(claim.split()), _api_key=api_key)
    except FactCheckError:
        # Supporting feature only — never break the main pipeline.
        return ()
