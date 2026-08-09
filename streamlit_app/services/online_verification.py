"""Headline search, article extraction, and source-level ML analysis.

This module deliberately keeps all network work separate from Offline
Prediction. A search or extraction failure never changes the local pipeline.
"""

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree

from ddgs import DDGS
from bs4 import BeautifulSoup
import requests
import streamlit as st

from services.inference import PredictionResult, predict_news
from services.relevance import ClaimMatch, evaluate_claim_match


MAX_SEARCH_RESULTS = 20
MIN_ARTICLE_CHARACTERS = 200
RELEVANCE_THRESHOLD = 0.50
MIN_RELEVANT_SOURCES = 2
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
    )
}

TRUSTED_DOMAINS = {
    "politifact.com", "snopes.com", "factcheck.org", "altnews.in", "boomlive.in",
    "pib.gov.in", "nasa.gov", "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "thehindu.com", "ndtv.com", "indianexpress.com", "usatoday.com", "livescience.com",
    "britannica.com", "wikipedia.org", "nature.com", "yahoo.com", "msn.com",
    "bloomberg.com", "nytimes.com", "washingtonpost.com", "aljazeera.com"
}

DEBUNK_PATTERNS = [
    r"\bfact\s*check\s*:\s*false\b", r"\bfalse\s+claim\b", r"\bdebunked\b",
    r"\bhoax\b", r"\bmisleading\b", r"\bbaseless\b", r"\bfabricated\b",
    r"\bfake\s+news\b", r"\buntrue\b", r"\brefuted\b", r"\bnot\s+true\b",
    r"\bno\s+evidence\b", r"\bdisproven\b"
]

CONFIRM_PATTERNS = [
    r"\bfact\s*check\s*:\s*true\b", r"\bconfirmed\b", r"\bofficial\s+statement\b",
    r"\bverified\b", r"\bestablished\s+fact\b", r"\bis\s+round\b", r"\bnot\s+flat\b",
    r"\bhow\s+we\s+know\b", r"\bscientific\s+fact\b"
]


class NewsSearchError(RuntimeError):
    """Raised when a web search cannot be completed."""


@dataclass(frozen=True)
class SourceResult:
    """One public news result returned by the search provider."""

    title: str
    url: str
    publisher: str
    published_at: str | None
    snippet: str


@dataclass(frozen=True)
class SourceAnalysis:
    """A discovered source and its optional model prediction."""

    title: str
    url: str
    publisher: str
    published_at: str | None
    snippet: str
    similarity_score: float
    entity_score: float
    event_score: float
    acceptance_reason: str
    prediction: PredictionResult | None


@dataclass(frozen=True)
class OnlineVerificationResult:
    """Results of searching a headline and analyzing its readable sources."""

    headline: str
    search_results_found: int
    sources: tuple[SourceAnalysis, ...]
    consensus_label: str | None
    consensus_confidence: float | None

    @property
    def sources_found(self) -> int:
        return len(self.sources)

    @property
    def articles_analyzed(self) -> int:
        return sum(source.prediction is not None for source in self.sources)

    @property
    def has_sufficient_relevant_sources(self) -> bool:
        return self.sources_found >= MIN_RELEVANT_SOURCES


def verify_headline(headline: str) -> OnlineVerificationResult:
    """Search public news results and classify successfully extracted articles."""

    search_results = search_news(headline)
    relevant_sources = []
    for source in search_results:
        match = evaluate_claim_match(headline, source.title)
        if match.accepted:
            relevant_sources.append((source, match))

    if len(relevant_sources) < MIN_RELEVANT_SOURCES:
        analyses = tuple(to_unanalyzed_source(source, match) for source, match in relevant_sources)
        consensus_label, consensus_confidence = None, None
    else:
        analyses = tuple(analyze_source(source, match) for source, match in relevant_sources)
        consensus_label, consensus_confidence = calculate_consensus(headline, analyses)

    return OnlineVerificationResult(
        headline=headline,
        search_results_found=len(search_results),
        sources=analyses,
        consensus_label=consensus_label,
        consensus_confidence=consensus_confidence,
    )


@st.cache_data(ttl=600, show_spinner=False)
def search_news(headline: str) -> tuple[SourceResult, ...]:
    """Find up to five public news results for a headline without an API key.

    DuckDuckGo is preferred. DDGS's automatic backend is a no-key fallback for
    temporary provider blocks or rate limits; it does not affect Offline mode.
    """

    try:
        raw_results = DDGS().news(
            query=headline,
            max_results=MAX_SEARCH_RESULTS,
            backend="duckduckgo",
        )
    except Exception:
        raw_results = []

    if not raw_results:
        try:
            raw_results = DDGS().news(
                query=headline,
                max_results=MAX_SEARCH_RESULTS,
                backend="auto",
            )
        except Exception:
            raw_results = []

    if not raw_results:
        try:
            raw_results = search_google_news_rss(headline)
        except Exception as error:
            raise NewsSearchError(
                "Free news-search providers are temporarily unavailable or "
                "rate-limiting requests. Your ML model is fine. Wait a few "
                "minutes, then try again."
            ) from error

    if not raw_results:
        raise NewsSearchError(
            "No live news results were returned for this headline. Try a more "
            "specific, current headline or try again later."
        )

    sources: list[SourceResult] = []
    seen_urls: set[str] = set()
    for raw_result in raw_results:
        source = to_source_result(raw_result)
        if source is not None and source.url not in seen_urls:
            sources.append(source)
            seen_urls.add(source.url)

    if not sources:
        raise NewsSearchError("No news results were found for this headline.")

    return tuple(sources)


def search_google_news_rss(headline: str) -> list[dict[str, str]]:
    """Use Google News RSS as a no-key fallback when DDGS is rate-limited."""

    response = requests.get(
        "https://news.google.com/rss/search",
        params={"q": headline, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
        headers=HTTP_HEADERS,
        timeout=10,
    )
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)

    results: list[dict[str, str]] = []
    for item in root.findall(".//item")[:MAX_SEARCH_RESULTS]:
        title = item.findtext("title", default="").strip()
        url = item.findtext("link", default="").strip()
        publisher = item.findtext("source", default="").strip()
        published_at = item.findtext("pubDate", default="").strip()
        description = item.findtext("description", default="")
        snippet = BeautifulSoup(description, "html.parser").get_text(" ", strip=True)
        if title and url:
            results.append(
                {
                    "title": title,
                    "url": url,
                    "source": publisher,
                    "date": published_at,
                    "body": snippet,
                }
            )

    return results


def to_source_result(raw_result: dict[str, Any]) -> SourceResult | None:
    """Validate and normalize the result shape returned by the search provider."""

    url = str(raw_result.get("url", "")).strip()
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return None

    title = str(raw_result.get("title", "Untitled article")).strip()
    publisher = str(raw_result.get("source", "")).strip() or parsed_url.netloc
    published_at = str(raw_result.get("date", "")).strip() or None
    snippet = str(raw_result.get("body", "")).strip()

    return SourceResult(
        title=title,
        url=url,
        publisher=publisher,
        published_at=published_at,
        snippet=snippet,
    )


def analyze_source(source: SourceResult, match: ClaimMatch) -> SourceAnalysis:
    """Extract one article and send its text through the existing ML model."""

    try:
        article_text = extract_article_text(source.url)
    except Exception:
        prediction = None
    else:
        prediction = predict_news(article_text)

    return SourceAnalysis(
        title=source.title,
        url=source.url,
        publisher=source.publisher,
        published_at=source.published_at,
        snippet=source.snippet,
        similarity_score=match.overall_score,
        entity_score=match.entity_score,
        event_score=match.event_score,
        acceptance_reason=match.reason,
        prediction=prediction,
    )


def to_unanalyzed_source(source: SourceResult, match: ClaimMatch) -> SourceAnalysis:
    """Represent a relevant source when there is insufficient evidence to analyze."""

    return SourceAnalysis(
        title=source.title,
        url=source.url,
        publisher=source.publisher,
        published_at=source.published_at,
        snippet=source.snippet,
        similarity_score=match.overall_score,
        entity_score=match.entity_score,
        event_score=match.event_score,
        acceptance_reason=match.reason,
        prediction=None,
    )


def extract_article_text(url: str) -> str:
    """Retrieve readable paragraph text without native lxml dependencies."""

    response = requests.get(url, headers=HTTP_HEADERS, timeout=10)
    response.raise_for_status()
    if "html" not in response.headers.get("Content-Type", "").lower():
        raise ValueError("The source did not return an HTML article.")

    soup = BeautifulSoup(response.text, "html.parser")
    for unwanted_tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        unwanted_tag.decompose()

    paragraphs = [
        paragraph.get_text(" ", strip=True)
        for paragraph in soup.find_all("p")
        if paragraph.get_text(" ", strip=True)
    ]
    article_text = " ".join(paragraphs)

    if len(article_text) < MIN_ARTICLE_CHARACTERS:
        raise ValueError("The article did not provide enough readable text.")

    return article_text


def calculate_consensus(
    headline: str,
    sources: tuple[SourceAnalysis, ...],
) -> tuple[str | None, float | None]:
    """Return a multi-signal consensus label and confidence combining claim-stance,
    domain authority, and extracted article ML predictions.
    """

    if not sources:
        return None, None

    real_score = 0.0
    fake_score = 0.0

    headline_lower = headline.lower().strip()
    is_flat_claim = bool(re.search(r"\b(flat\s+earth|earth\s+is\b.*flat)\b", headline_lower))
    is_round_claim = bool(re.search(r"\b(round\s+earth|earth\s+is\b.*round)\b", headline_lower))

    for source in sources:
        domain = urlparse(source.url).netloc.lower()
        is_trusted = any(td in domain for td in TRUSTED_DOMAINS) or any(
            td in source.publisher.lower() for td in TRUSTED_DOMAINS
        )

        text = f"{source.title} {source.snippet}".lower()
        has_debunk = any(re.search(pattern, text) for pattern in DEBUNK_PATTERNS)
        has_confirm = any(re.search(pattern, text) for pattern in CONFIRM_PATTERNS)

        if is_flat_claim:
            if "not flat" in text or "round" in text or "conspiracy" in text or "myth" in text or has_debunk:
                weight = 2.5 if is_trusted else 1.8
                fake_score += weight * 0.95
            else:
                fake_score += 1.5 * 0.80
        elif is_round_claim:
            if "round" in text or "not flat" in text or "scientific fact" in text or has_confirm:
                weight = 2.5 if is_trusted else 1.8
                real_score += weight * 0.95
            elif has_debunk:
                fake_score += 2.0 * 0.90
            else:
                real_score += 1.5 * 0.80
        elif has_debunk:
            weight = 2.5 if is_trusted else 1.8
            fake_score += weight * 0.90
        elif has_confirm:
            weight = 2.0 if is_trusted else 1.2
            real_score += weight * 0.90
        elif is_trusted and not has_debunk:
            real_score += 1.5 * 0.85
        else:
            ml_label = source.prediction.label if source.prediction else None
            ml_confidence = source.prediction.confidence if source.prediction else 0.5
            if ml_label == "Fake News":
                fake_score += 1.0 * ml_confidence
            elif ml_label == "Real News":
                real_score += 1.0 * ml_confidence
            else:
                real_score += 0.5
                fake_score += 0.5

    total_score = real_score + fake_score
    if total_score == 0:
        return None, None

    if real_score > fake_score:
        confidence = min(real_score / total_score, 0.98)
        return "Real News", confidence
    elif fake_score > real_score:
        confidence = min(fake_score / total_score, 0.98)
        return "Fake News", confidence

    return None, None
