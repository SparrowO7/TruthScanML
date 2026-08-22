"""Headline search, article extraction, and multi-signal evidence consensus.

This module deliberately keeps all network work separate from Offline
Prediction. A search or extraction failure never changes the local pipeline.

Free-only architecture (no paid APIs):
  Search chain:  DuckDuckGo (ddgs) → Bing News RSS → Google News RSS
  Evidence:      parallel article extraction + stance patterns + ML support
  Cross-checks:  Google Fact Check Tools (ClaimReview, optional free key)
                 Wikipedia lead check for death claims (conservative)
  Verdict:       weighted consensus; returns INCONCLUSIVE when evidence is
                 insufficient or conflicting — never forces a call.
"""

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import re
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree

from bs4 import BeautifulSoup
from ddgs import DDGS
import streamlit as st

from services.factcheck import FactCheckVerdict, fetch_fact_checks
from services.inference import PredictionResult, predict_news
from services.relevance import ClaimMatch, evaluate_claim_match
from services.wikipedia import WikipediaCheck, wikipedia_death_check
from utils.http_helpers import http_get_with_retry


MAX_SEARCH_RESULTS = 20
MIN_ARTICLE_CHARACTERS = 200
RELEVANCE_THRESHOLD = 0.50
MIN_RELEVANT_SOURCES = 2
MAX_PARALLEL_FETCHES = 6
EXTRACTION_CACHE_SECONDS = 3_600
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
    )
}

# Trusted fact-check and authoritative news domains.
# Indian sources prioritised for local news verification.
TRUSTED_DOMAINS = {
    # International fact-checkers
    "politifact.com", "snopes.com", "factcheck.org", "fullfact.org",
    # Indian fact-checkers
    "altnews.in", "boomlive.in", "pib.gov.in", "factcrescendo.com",
    "vishvasnews.com", "newschecker.in", "indiatoday.in", "factly.in",
    # International wire services / broadcasters
    "nasa.gov", "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "bloomberg.com", "nytimes.com", "washingtonpost.com", "aljazeera.com",
    "usatoday.com", "livescience.com", "britannica.com", "wikipedia.org",
    "nature.com", "who.int", "un.org",
    # Indian mainstream news
    "thehindu.com", "ndtv.com", "indianexpress.com",
    "timesofindia.com", "hindustantimes.com", "scroll.in",
    "thewire.in", "theprint.in", "moneycontrol.com", "zeenews.india.com",
    "aajtak.in", "abplive.com", "thequint.com",
}

# Professional fact-check domains (highest credibility weight).
FACT_CHECKER_DOMAINS = {
    "altnews.in", "boomlive.in", "factcrescendo.com", "vishvasnews.com",
    "newschecker.in", "politifact.com", "snopes.com", "factcheck.org",
    "fullfact.org", "factly.in",
}

# Domains known to publish satire, parody, or unreliable viral content.
# Articles from these are excluded from scoring.
SATIRE_DOMAINS = {
    "fauxy.com", "theonion.com", "babylonbee.com", "newsthump.com",
    "clickhole.com", "worldnewsdailyreport.com", "empirenews.net",
}

# --- Fake / debunking signal patterns ---
DEBUNK_PATTERNS = [
    # Explicit fact-check verdicts
    r"\bfact\s*check\s*:\s*false\b", r"\bfact\s*check\b", r"\bfact\s*checked\b",
    # Debunking language
    r"\bdebunked\b", r"\bhoax\b", r"\bhoax\s+alert\b", r"\bfake\s+alert\b",
    r"\bmisleading\b", r"\bbaseless\b", r"\bfabricated\b",
    r"\bfake\s+news\b", r"\buntrue\b", r"\brefuted\b", r"\bnot\s+true\b",
    r"\bno\s+evidence\b", r"\bdisproven\b", r"\bmisinformation\b",
    r"\bfalsely\s+claims\b", r"\bfalse\s+claim\b", r"\bviral\s+fake\b",
    r"\brunfounded\b", r"\bunverified\s+claim\b", r"\bspread\s+misinformation\b",
    # Death hoax specific
    r"\bdeath\s+hoax\b", r"\bdeath\s+rumou?r\b", r"\bdeath\s+news\s+fake\b",
    r"\bdied\s+hoax\b", r"\bnot\s+dead\b", r"\bstill\s+alive\b",
    r"\bafwah\b", r"\brumour\b", r"\brumor\b",
    # Hindi / Hinglish debunk words
    r"\bjhooth\b", r"\bjhuth[ai]\b", r"\bfarzi\b", r"\bnakli\b",
    r"\bgalat\s+(?:khabar|news|jaankari|baat)\b", r"\bviral\s+afwah\b",
]

# --- Real / confirmation signal patterns ---
CONFIRM_PATTERNS = [
    r"\bfact\s*check\s*:\s*true\b", r"\bconfirmed\b", r"\bofficial\s+statement\b",
    r"\bverified\b", r"\bestablished\s+fact\b", r"\bscientific\s+fact\b",
    r"\baccording\s+to\s+scientists\b", r"\bofficially\s+confirmed\b",
    r"\bgovernment\s+confirms\b", r"\bpress\s+conference\b",
    r"\bofficial\s+source\b", r"\bisro\s+confirms\b", r"\bnasa\s+confirms\b",
    r"\bis\s+round\b", r"\bnot\s+flat\b", r"\bhow\s+we\s+know\b",
]

# --- Alive / person-is-fine signals (used in death-claim contradiction) ---
ALIVE_SIGNALS = [
    r"\balive\b", r"\bstill\s+alive\b", r"\bis\s+alive\b", r"\bzinda\b",
    r"\bdenies\b", r"\bdenied\b", r"\brefutes\b",
    r"\bis\s+fine\b", r"\bis\s+well\b", r"\bis\s+safe\b",
    r"\bappears\s+in\b", r"\bseen\s+at\b", r"\bspotted\b",
    r"\bposts\s+on\b", r"\bshares\s+on\b", r"\btweets\b",
    r"\bnot\s+dead\b", r"\bdeath\s+hoax\b", r"\bdeath\s+rumou?r\b",
    r"\bno\s+truth\b",
]

# Verb/adjective forms that signal a death claim in the headline.
_DEATH_WORDS = re.compile(
    r"\b(died|dead|death|passed\s+away|no\s+more|is\s+no\s+more|nahi\s+rahe|"
    r"gujar\s+gaye|guzar\s+gaye|mar\s+gaya|mar\s+gaye|inteqal|wafat|"
    r"dehant|swargwas[ie]?)\b",
    flags=re.IGNORECASE,
)

# Death phrase used inside sentence-level evidence checks.
_DEATH_PHRASE = (
    r"(?:died|dead|death|passed\s+away|no\s+more|nahi\s+rahe|"
    r"gujar\s+gaye|guzar\s+gaye|mar\s+gaya|mar\s+gaye|inteqal|wafat|"
    r"dehant|swargwas[ie]?)"
)

# Negation guard: a death word near one of these words is a debunk, not a
# confirmation ("family denied reports of X's death", "X death rumour").
_NEGATION_WORDS = (
    r"(?:denied|denies|refuted|refutes|rubbish(?:ed)?|dismissed|"
    r"quash(?:ed)?|debunk(?:ed|s)?|hoax|rumou?rs?|afwah|jhooth|jhuth[ai]|"
    r"farzi|nakli|baseless|misleading|fake|false)"
)


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
    """A discovered source with its evidence signals and ML prediction.

    article_has_debunk / article_has_confirm reflect pattern checks run
    against the full scraped article body when available; evidence_quote is
    the sentence that triggered the strongest signal.
    """

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
    article_has_debunk: bool = False
    article_has_confirm: bool = False
    evidence_quote: str = ""

    # Evidence-engine annotations (filled during consensus)
    stance: str = "NEUTRAL"
    credibility_score: float = 1.0
    freshness_score: float = 0.7
    is_independent: bool = True
    # Optional local NLI model assist (None when the model is unavailable)
    nli_stance: str | None = None
    nli_score: float = 0.0


@dataclass(frozen=True)
class OnlineVerificationResult:
    """Results of searching a headline and analyzing its readable sources."""

    headline: str
    search_results_found: int
    sources: tuple[SourceAnalysis, ...]
    consensus_label: str | None
    consensus_confidence: float | None
    supporting_count: int = 0
    contradicting_count: int = 0
    neutral_count: int = 0
    evidence_summary: str = ""
    fact_checks: tuple[FactCheckVerdict, ...] = ()
    wikipedia_note: str = ""
    wikipedia_url: str = ""

    @property
    def sources_found(self) -> int:
        return len(self.sources)

    @property
    def articles_analyzed(self) -> int:
        return sum(source.prediction is not None for source in self.sources)

    @property
    def has_sufficient_relevant_sources(self) -> bool:
        return self.sources_found >= MIN_RELEVANT_SOURCES


# ---------------------------------------------------------------------------
# Search query builder
# ---------------------------------------------------------------------------

_QUESTION_STARTERS = re.compile(
    r"^(is|are|can|does|do|was|were|has|have|did|will|would|should|could|kya)\s+",
    flags=re.IGNORECASE,
)


def build_search_query(headline: str) -> str:
    """Return a cleaner search query derived from the user headline.

    Only surface-level cleanup is applied so the factual meaning is preserved:
      - Strip leading question-starter words (Is/Are/Kya …)
      - Strip trailing question marks
      - Collapse extra whitespace
    """

    query = headline.strip()
    query = re.sub(r"\?+$", "", query).strip()
    query = _QUESTION_STARTERS.sub("", query).strip()
    query = re.sub(r"\s+", " ", query).strip()
    return query or headline.strip()


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def verify_headline(
    headline: str, fast_mode: bool = False, nli_enabled: bool = True
) -> OnlineVerificationResult:
    """Search public news results and classify successfully extracted articles.

    If fast_mode is True, article extraction and ML prediction are skipped;
    only titles and snippets are used for pattern checks. When nli_enabled is
    False the optional local NLI stance assist is skipped entirely.
    """

    search_results = search_news(headline)
    relevant_sources = []
    for source in search_results:
        match = evaluate_claim_match(headline, source.title)
        if match.accepted:
            relevant_sources.append((source, match))

    if len(relevant_sources) < MIN_RELEVANT_SOURCES:
        analyses = tuple(
            to_unanalyzed_source(source, match) for source, match in relevant_sources
        )
        consensus_label = "Inconclusive"
        consensus_confidence = None
        supporting_count, contradicting_count = 0, 0
        neutral_count = len(analyses)
        summary = (
            "INCONCLUSIVE — insufficient relevant sources found to verify "
            "this claim."
        )
        fact_checks: tuple[FactCheckVerdict, ...] = ()
        wikipedia_check: WikipediaCheck | None = None
    else:
        if fast_mode:
            raw_analyses = tuple(
                fast_analyze_source(source, match) for source, match in relevant_sources
            )
        else:
            raw_analyses = _analyze_sources_parallel(headline, relevant_sources)

        if nli_enabled:
            raw_analyses = _apply_nli_assist(headline, raw_analyses)
        fact_checks = _lookup_fact_checks(headline)
        death_subject = extract_death_subject(headline)
        wikipedia_check = (
            wikipedia_death_check(death_subject) if death_subject else None
        )

        (
            consensus_label,
            consensus_confidence,
            analyses,
            supporting_count,
            contradicting_count,
            neutral_count,
            summary,
        ) = calculate_consensus(
            headline, raw_analyses, fact_checks=fact_checks, wikipedia_check=wikipedia_check
        )

    return OnlineVerificationResult(
        headline=headline,
        search_results_found=len(search_results),
        sources=analyses,
        consensus_label=consensus_label,
        consensus_confidence=consensus_confidence,
        supporting_count=supporting_count,
        contradicting_count=contradicting_count,
        neutral_count=neutral_count,
        evidence_summary=summary,
        fact_checks=fact_checks,
        wikipedia_note=(wikipedia_check.note if wikipedia_check else ""),
        wikipedia_url=(wikipedia_check.url if wikipedia_check else ""),
    )


def _apply_nli_assist(
    headline: str, analyses: tuple[SourceAnalysis, ...]
) -> tuple[SourceAnalysis, ...]:
    """Attach local NLI stance scores to each source when the model exists.

    The NLI cross-encoder (~100 MB) compares the claim against each source's
    title + snippet (+ evidence quote). Its stance is only advisory: it is
    consumed in ``_classify_stance`` as a conservative last-resort upgrade
    for sources the keyword patterns left NEUTRAL.
    """

    try:
        from services.nli_stance import score_pairs
    except Exception:
        return analyses

    pairs = [
        (headline, f"{a.title}. {a.snippet} {a.evidence_quote}"[:600])
        for a in analyses
    ]
    try:
        results = score_pairs(pairs)
    except Exception:
        return analyses

    if not results or all(stance is None for stance, _ in results):
        return analyses

    updated = [
        dataclasses.replace(a, nli_stance=stance, nli_score=score)
        for a, (stance, score) in zip(analyses, results)
    ]
    return tuple(updated)


def _analyze_sources_parallel(
    headline: str, relevant_sources: list[tuple[SourceResult, ClaimMatch]]
) -> tuple[SourceAnalysis, ...]:
    """Download and analyze articles concurrently, preserving input order."""

    sources = [source for source, _ in relevant_sources]
    matches = [match for _, match in relevant_sources]
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FETCHES) as pool:
        analyses = list(
            pool.map(
                lambda source, match: analyze_source(source, match, headline),
                sources,
                matches,
            )
        )
    return tuple(analyses)


def _lookup_fact_checks(headline: str) -> tuple[FactCheckVerdict, ...]:
    """Fetch professional fact-checks when a free API key is configured."""

    try:
        api_key = st.secrets.get("FACT_CHECK_API_KEY", "")
    except Exception:
        api_key = ""

    return fetch_fact_checks(headline, api_key)


# ---------------------------------------------------------------------------
# Search: DuckDuckGo → Bing News RSS → Google News RSS
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Search: rotating provider order (round-robin across requests) so no single
# free provider eats every request and hits its rate limit quickly.
# Request 1 → DDG first, request 2 → Bing first, request 3 → Google first,
# then repeat. Within one request the remaining providers are still tried as
# fallbacks, with GDELT always as the last resort.
# ---------------------------------------------------------------------------

_next_provider = {"index": 0}


def _search_via_ddg(query: str) -> list[dict[str, str]]:
    """DuckDuckGo via ddgs: explicit backend, then the auto fallback."""

    try:
        results = DDGS().news(
            query=query, max_results=MAX_SEARCH_RESULTS, backend="duckduckgo"
        )
    except Exception:
        results = []
    if not results:
        try:
            results = DDGS().news(
                query=query, max_results=MAX_SEARCH_RESULTS, backend="auto"
            )
        except Exception:
            results = []
    return results


def _rotated_search_order() -> tuple:
    """Return (providers in rotated order) + GDELT as the final fallback."""

    providers = (_search_via_ddg, search_bing_news_rss, search_google_news_rss)
    start = _next_provider["index"] % len(providers)
    _next_provider["index"] += 1
    rotated = providers[start:] + providers[:start]
    return rotated + (search_gdelt,)


@st.cache_data(ttl=600, show_spinner=False)
def search_news(headline: str) -> tuple[SourceResult, ...]:
    """Find up to twenty public news results for a headline without an API key.

    Providers rotate per request (DDG → Bing → Google → …) to spread load;
    whichever provider is chosen first, the others still act as in-request
    fallbacks, ending with GDELT.
    """

    query = build_search_query(headline)

    raw_results: list[dict[str, str]] = []
    for provider in _rotated_search_order():
        try:
            raw_results = provider(query)
        except Exception:
            raw_results = []
        if raw_results:
            break

    if not raw_results:
        raise NewsSearchError(
            "Free news-search providers are temporarily unavailable or "
            "rate-limiting requests. Your ML model is fine. Wait a few "
            "minutes, then try again."
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


def search_bing_news_rss(query: str) -> list[dict[str, str]]:
    """Use Bing News RSS as a keyless fallback search provider."""

    response = http_get_with_retry(
        "https://www.bing.com/news/search",
        params={"q": query, "format": "RSS", "setlang": "en"},
        headers=HTTP_HEADERS,
        timeout=10,
        retries=2,
    )
    root = ElementTree.fromstring(response.content)

    results: list[dict[str, str]] = []
    for item in root.findall(".//item")[:MAX_SEARCH_RESULTS]:
        title = item.findtext("title", default="").strip()
        url = item.findtext("link", default="").strip()
        published_at = item.findtext("pubDate", default="").strip()
        description = item.findtext("description", default="")
        snippet = BeautifulSoup(description, "html.parser").get_text(" ", strip=True)
        if title and url:
            results.append(
                {
                    "title": title,
                    "url": url,
                    "source": urlparse(url).netloc,
                    "date": published_at,
                    "body": snippet,
                }
            )
    return results


def search_google_news_rss(query: str) -> list[dict[str, str]]:
    """Use Google News RSS as a keyless fallback when DDGS is rate-limited."""

    response = http_get_with_retry(
        "https://news.google.com/rss/search",
        params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
        headers=HTTP_HEADERS,
        timeout=10,
        retries=2,
    )
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


def search_gdelt(query: str) -> list[dict[str, str]]:
    """Use the free GDELT DOC 2.0 API as a keyless global fallback source.

    Endpoint: https://api.gdeltproject.org/api/v2/doc/doc (no key, no cost).
    Returns up to MAX_SEARCH_RESULTS recent articles mentioning the query.
    """

    response = http_get_with_retry(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={
            "query": query,
            "mode": "ArtList",
            "maxrecords": MAX_SEARCH_RESULTS,
            "format": "json",
            "sort": "DateDesc",
        },
        headers=HTTP_HEADERS,
        timeout=12,
        retries=2,
    )

    data = response.json()
    results: list[dict[str, str]] = []
    for article in data.get("articles", [])[:MAX_SEARCH_RESULTS]:
        url = str(article.get("url", "")).strip()
        title = str(article.get("title", "")).strip()
        if not url or not title:
            continue
        # seendate looks like "20260822T101500Z"; normalise to ISO.
        published_at = ""
        seen_date = str(article.get("seendate", "")).strip()
        try:
            parsed_date = datetime.strptime(seen_date, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
            published_at = parsed_date.isoformat()
        except ValueError:
            published_at = seen_date
        results.append(
            {
                "title": title,
                "url": url,
                "source": str(article.get("domain", "")).strip(),
                "date": published_at,
                "body": "",
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


# ---------------------------------------------------------------------------
# Article extraction (cached, retried) and analysis
# ---------------------------------------------------------------------------

@st.cache_data(ttl=EXTRACTION_CACHE_SECONDS, show_spinner=False)
def extract_article_text(url: str) -> str:
    """Retrieve readable paragraph text with a fallback chain.

    Extraction order:
      1. <article> tag paragraphs  — most news sites use this semantic element
      2. <main> tag paragraphs     — common fallback container
      3. All <p> tags on the page  — original behaviour, broadest net
    """

    response = http_get_with_retry(
        url, headers=HTTP_HEADERS, timeout=10, retries=2
    )
    if "html" not in response.headers.get("Content-Type", "").lower():
        raise ValueError("The source did not return an HTML article.")

    soup = BeautifulSoup(response.text, "html.parser")
    for unwanted_tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        unwanted_tag.decompose()

    def _paragraphs_from(tag) -> str:
        if tag is None:
            return ""
        return " ".join(
            p.get_text(" ", strip=True)
            for p in tag.find_all("p")
            if p.get_text(" ", strip=True)
        )

    article_text = _paragraphs_from(soup.find("article"))

    if len(article_text) < MIN_ARTICLE_CHARACTERS:
        article_text = _paragraphs_from(soup.find("main"))

    if len(article_text) < MIN_ARTICLE_CHARACTERS:
        article_text = " ".join(
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if p.get_text(" ", strip=True)
        )

    if len(article_text) < MIN_ARTICLE_CHARACTERS:
        raise ValueError("The article did not provide enough readable text.")

    return article_text


def _first_matching_sentence(text: str, pattern_groups: list[list[str]]) -> str:
    """Return the first sentence matching any pattern, for evidence display."""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        lower = sentence.lower()
        for patterns in pattern_groups:
            if any(re.search(pattern, lower) for pattern in patterns):
                return sentence.strip()[:280]
    return ""


def _check_patterns(text: str) -> tuple[bool, bool, str]:
    """Return (has_debunk, has_confirm, evidence_quote) for a text block."""

    lower = text.lower()
    has_debunk = any(re.search(p, lower) for p in DEBUNK_PATTERNS)
    has_confirm = any(re.search(p, lower) for p in CONFIRM_PATTERNS)

    groups: list[list[str]] = []
    if has_debunk:
        groups.append(DEBUNK_PATTERNS)
    if has_confirm:
        groups.append(CONFIRM_PATTERNS)
    quote = _first_matching_sentence(text, groups) if groups else ""

    return has_debunk, has_confirm, quote


def _death_linked_quote(article_text: str, death_subject: str) -> str:
    """Return the sentence linking the claimed person to a death word."""

    if not death_subject:
        return ""
    last_name = death_subject.split()[-1].lower()
    for sentence in re.split(r"(?<=[.!?])\s+", article_text):
        lower = sentence.lower()
        if last_name in lower and re.search(_DEATH_PHRASE, lower):
            return sentence.strip()[:280]
    return ""


def analyze_source(
    source: SourceResult, match: ClaimMatch, headline: str = ""
) -> SourceAnalysis:
    """Extract one article, check body-level stance signals, and run ML."""

    article_text: str | None = None
    prediction: PredictionResult | None = None
    article_has_debunk = False
    article_has_confirm = False
    evidence_quote = ""

    try:
        article_text = extract_article_text(source.url)
    except Exception:
        pass

    if article_text:
        prediction = predict_news(article_text)
        article_has_debunk, article_has_confirm, evidence_quote = _check_patterns(
            article_text
        )

    if not evidence_quote:
        death_subject = extract_death_subject(headline) if headline else None
        if death_subject and article_text:
            evidence_quote = _death_linked_quote(article_text, death_subject)

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
        article_has_debunk=article_has_debunk,
        article_has_confirm=article_has_confirm,
        evidence_quote=evidence_quote,
        freshness_score=compute_freshness_score(source.published_at),
    )


def fast_analyze_source(source: SourceResult, match: ClaimMatch) -> SourceAnalysis:
    """Analyze a source without downloading the full article.

    Uses only the title + snippet for pattern detection and skips ML.
    """

    snippet_text = f"{source.title} {source.snippet}"
    article_has_debunk, article_has_confirm, evidence_quote = _check_patterns(
        snippet_text
    )
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
        article_has_debunk=article_has_debunk,
        article_has_confirm=article_has_confirm,
        evidence_quote=evidence_quote,
        freshness_score=compute_freshness_score(source.published_at),
    )


def to_unanalyzed_source(source: SourceResult, match: ClaimMatch) -> SourceAnalysis:
    """Represent a relevant source when there is insufficient evidence."""

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
        article_has_debunk=False,
        article_has_confirm=False,
        freshness_score=compute_freshness_score(source.published_at),
    )


# ---------------------------------------------------------------------------
# Freshness scoring
# ---------------------------------------------------------------------------

_RELATIVE_DATE = re.compile(
    r"(\d+)\s*(minute|hour|day|week|month)s?\s+ago", flags=re.IGNORECASE
)


def _parse_published_date(value: str | None) -> datetime | None:
    """Best-effort parse of provider date strings into an aware datetime."""

    if not value:
        return None
    text = value.strip()

    relative = _RELATIVE_DATE.search(text)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2).lower()
        delta = {
            "minute": timedelta(minutes=amount),
            "hour": timedelta(hours=amount),
            "day": timedelta(days=amount),
            "week": timedelta(weeks=amount),
            "month": timedelta(days=amount * 30),
        }.get(unit)
        if delta is not None:
            return datetime.now(timezone.utc) - delta

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    try:
        parsed = parsedate_to_datetime(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def compute_freshness_score(published_at: str | None) -> float:
    """Score recency 0.5–1.0; undated items get a neutral 0.7."""

    published = _parse_published_date(published_at)
    if published is None:
        return 0.7

    age_days = (datetime.now(timezone.utc) - published).days
    if age_days < 0:
        age_days = 0
    if age_days <= 7:
        return 1.0
    if age_days <= 30:
        return 0.85
    if age_days <= 180:
        return 0.7
    if age_days <= 365:
        return 0.6
    return 0.5


# ---------------------------------------------------------------------------
# Credibility (proper domain matching, no substring false positives)
# ---------------------------------------------------------------------------

def registrable_domain(url: str) -> str:
    """Return a comparable domain like 'altnews.in' from any URL."""

    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    parts = [part for part in netloc.split(":")[0].split(".") if part]
    return ".".join(parts[-2:]) if len(parts) >= 2 else netloc


def _domain_matches(domain: str, known_domains: set[str]) -> bool:
    """Exact or parent-suffix match — 'notaltnews.in' must NOT match."""

    return any(
        domain == known or domain.endswith("." + known)
        for known in known_domains
    )


def calculate_credibility(source: SourceAnalysis, domain: str) -> float:
    """Credibility weight for one source (fact-checkers rank highest)."""

    if _domain_matches(domain, FACT_CHECKER_DOMAINS):
        return 2.0
    if _domain_matches(domain, TRUSTED_DOMAINS) or any(
        known in source.publisher.lower() for known in TRUSTED_DOMAINS
    ):
        return 1.5
    return 1.0


# ---------------------------------------------------------------------------
# Location / destination contradiction (space & geographic claims)
# ---------------------------------------------------------------------------

LOCATION_GROUPS: list[frozenset[str]] = [
    frozenset({"sun", "solar", "star"}),
    frozenset({"moon", "lunar"}),
    frozenset({"mars", "martian", "red planet"}),
    frozenset({"earth", "terrestrial"}),
    frozenset({"venus", "venusian"}),
    frozenset({"jupiter", "jovian"}),
    frozenset({"saturn"}),
    frozenset({"mercury"}),
    frozenset({"space station", "iss"}),
    frozenset({"asteroid", "comet"}),
]

_ALL_LOCATION_WORDS: frozenset[str] = frozenset(
    word for group in LOCATION_GROUPS for word in group
)


def _group_for(location_word: str) -> frozenset[str] | None:
    for group in LOCATION_GROUPS:
        if location_word in group:
            return group
    return None


def extract_claim_location(headline: str) -> str | None:
    """Extract a spatial target word ('sun', 'moon', 'mars'…) from the claim."""

    lower = headline.lower()
    patterns = [
        r"\blanded?\s+on\s+(?:the\s+)?(\w+)",
        r"\blanding\s+on\s+(?:the\s+)?(\w+)",
        r"\bcrashed?\s+(?:into|on)\s+(?:the\s+)?(\w+)",
        r"\bon\s+(?:the\s+)?(\w+)",
        r"\bto\s+(?:the\s+)?(\w+)",
        r"\bat\s+(?:the\s+)?(\w+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            candidate = match.group(1).strip()
            if candidate in _ALL_LOCATION_WORDS:
                return candidate
    return None


def _dominant_location(text: str) -> str | None:
    """Return whichever known location word appears most in a text block."""

    lower = text.lower()
    counts: dict[str, int] = {}
    for word in _ALL_LOCATION_WORDS:
        count = len(re.findall(r"\b" + re.escape(word) + r"\b", lower))
        if count:
            counts[word] = count
    if not counts:
        return None
    return max(counts, key=lambda word: counts[word])


def _locations_contradict(claim_loc: str, article_loc: str) -> bool:
    """True when claim and article refer to different known locations."""

    if claim_loc == article_loc:
        return False
    claim_group = _group_for(claim_loc)
    article_group = _group_for(article_loc)
    if claim_group is None or article_group is None:
        return False
    return claim_group != article_group


# ---------------------------------------------------------------------------
# Death-claim engine
# ---------------------------------------------------------------------------

def extract_death_subject(headline: str) -> str | None:
    """Return the subject name if the headline claims someone died."""

    if not _DEATH_WORDS.search(headline):
        return None

    # Role + name first (most specific): the role word is matched
    # case-insensitively (pm/PM), but the captured name must be strictly
    # TitleCase — otherwise IGNORECASE would let '[A-Z]' match lowercase
    # words like 'is', and "Actor Vikram Gokhale" would capture the role.
    role_prefix = re.search(
        r"\b(?:pm|cm|president|minister|actor|cricketer|politician|singer|director)\s+",
        headline,
        flags=re.IGNORECASE,
    )
    if role_prefix:
        name_after_role = re.match(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", headline[role_prefix.end():]
        )
        if name_after_role:
            return name_after_role.group(1).lower()

    name_match = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", headline)
    if name_match:
        return name_match.group(1).lower()

    any_name = re.findall(r"\b([A-Z][a-z]{2,})\b", headline)
    candidates = any_name[1:] if any_name and headline.strip().startswith(any_name[0]) else any_name
    if candidates:
        return candidates[0].lower()

    death_match = _DEATH_WORDS.search(headline)
    if death_match:
        before_death = headline[: death_match.start()].strip()
        if before_death:
            return before_death.split()[-1].lower()

    return None


def article_confirms_person_alive(person_name: str, article_text: str) -> bool:
    """True when an article mentions the person AND shows an alive signal."""

    lower = article_text.lower()

    name_tokens = [token for token in person_name.split() if len(token) > 2]
    if not name_tokens:
        return False
    if not any(token in lower for token in name_tokens):
        return False

    return any(re.search(pattern, lower) for pattern in ALIVE_SIGNALS)


def _negated_death(text: str) -> bool:
    """True when a death word appears near a denial/rumour word (same sentence).

    Character-window guard (~60 chars) so 'family denied reports of X's
    death' is treated as a debunk, not a confirmation.
    """

    return bool(
        re.search(rf"{_NEGATION_WORDS}\b[^.!?]{{0,60}}{_DEATH_PHRASE}", text)
        or re.search(rf"{_DEATH_PHRASE}\b[^.!?]{{0,60}}{_NEGATION_WORDS}", text)
    )


# ---------------------------------------------------------------------------
# Evidence aggregation & consensus
# ---------------------------------------------------------------------------

def _classify_stance(
    source: SourceAnalysis,
    headline_lower: str,
    death_subject: str | None,
    claim_location: str | None,
) -> str:
    """Decide SUPPORTS / CONTRADICTS / NEUTRAL for one source."""

    snippet_text = f"{source.title} {source.snippet}".lower()
    has_debunk = source.article_has_debunk or any(
        re.search(p, snippet_text) for p in DEBUNK_PATTERNS
    )
    has_confirm = source.article_has_confirm or any(
        re.search(p, snippet_text) for p in CONFIRM_PATTERNS
    )

    # 1. Location contradiction → CONTRADICTS
    if claim_location is not None:
        article_location = _dominant_location(snippet_text)
        if article_location is not None and _locations_contradict(claim_location, article_location):
            return "CONTRADICTS"

    # 2. Death contradiction / confirmation (with negation guard)
    if death_subject is not None:
        if article_confirms_person_alive(death_subject, snippet_text):
            return "CONTRADICTS"

        last_name = death_subject.split()[-1]
        linked_death = (
            re.search(
                rf"\b{last_name}\b(?:\s+\w+){{0,4}}\s+{_DEATH_PHRASE}", snippet_text
            )
            or re.search(
                rf"{_DEATH_PHRASE}(?:\s+\w+){{0,4}}\s+\b{last_name}\b", snippet_text
            )
        )

        if linked_death and not has_debunk and not _negated_death(snippet_text):
            return "SUPPORTS"

        # Ignore generic confirmation words for death claims unless the death
        # is explicitly linked to the person (handled above).
        has_confirm = False

    # 3. Scientific hard rules
    is_flat_claim = bool(re.search(r"\b(flat\s+earth|earth\s+is\b.*flat)\b", headline_lower))
    is_round_claim = bool(re.search(r"\b(round\s+earth|earth\s+is\b.*round)\b", headline_lower))
    if is_flat_claim:
        if "not flat" in snippet_text or "round" in snippet_text or has_debunk:
            return "CONTRADICTS"
    elif is_round_claim:
        if "round" in snippet_text or "not flat" in snippet_text or has_confirm:
            return "SUPPORTS"
        if has_debunk:
            return "CONTRADICTS"

    # 4. Explicit patterns
    if has_debunk:
        return "CONTRADICTS"
    if has_confirm:
        return "SUPPORTS"

    # 5. Local NLI assist — only for otherwise-NEUTRAL sources, and only
    #    when the small model is confident enough to be trusted.
    if (
        source.nli_stance in {"SUPPORTS", "CONTRADICTS"}
        and source.nli_score >= 0.80
    ):
        return source.nli_stance

    return "NEUTRAL"


def calculate_consensus(
    headline: str,
    sources: tuple[SourceAnalysis, ...],
    fact_checks: tuple[FactCheckVerdict, ...] = (),
    wikipedia_check: WikipediaCheck | None = None,
) -> tuple[str | None, float | None, tuple[SourceAnalysis, ...], int, int, int, str]:
    """Return verdict, confidence, updated sources, counts, and explanation.

    Verdict labels: 'Real News', 'Fake News', 'Inconclusive', or None when
    nothing at all could be evaluated. Evidence that is weak, conflicting, or
    single-sourced yields 'Inconclusive' instead of a forced verdict.
    """

    if not sources and not fact_checks and wikipedia_check is None:
        return None, None, sources, 0, 0, 0, "No sources available for analysis."

    headline_lower = headline.lower().strip()
    claim_location = extract_claim_location(headline)
    death_subject = extract_death_subject(headline)

    updated_sources = []
    seen_domains: set[str] = set()

    real_score = 0.0
    fake_score = 0.0

    supporting_count = 0
    contradicting_count = 0
    neutral_count = 0

    # Hard-science claims (space destinations, earth shape) rest on the
    # contradiction engine, so a single decisive source may carry them.
    is_science_claim = bool(
        re.search(
            r"\b(flat\s+earth|round\s+earth|earth\s+is\b.*(?:flat|round))\b",
            headline_lower,
        )
    )
    strong_evidence = (
        any(check.stance in {"SUPPORTS", "CONTRADICTS"} for check in fact_checks)
        or (wikipedia_check is not None and wikipedia_check.stance != "NEUTRAL")
        or claim_location is not None
        or is_science_claim
    )

    for source in sources:
        domain = registrable_domain(source.url)

        # Satire domains are excluded entirely.
        if _domain_matches(domain, SATIRE_DOMAINS):
            continue

        is_independent = domain not in seen_domains
        seen_domains.add(domain)

        credibility = calculate_credibility(source, domain)
        if not is_independent:
            credibility *= 0.2

        # Freshness dampens stale articles without removing their voice.
        effective_credibility = credibility * (0.7 + 0.3 * source.freshness_score)

        stance = _classify_stance(source, headline_lower, death_subject, claim_location)

        if stance == "SUPPORTS":
            supporting_count += 1
            real_score += effective_credibility * 1.5
        elif stance == "CONTRADICTS":
            contradicting_count += 1
            fake_score += effective_credibility * 1.5
        else:
            neutral_count += 1
            # ML prediction is a supporting signal ONLY for neutral articles,
            # capped so it can never overpower explicit factual stances.
            if source.prediction:
                ml_label = source.prediction.label
                ml_conf = source.prediction.confidence
                ml_weight = (
                    effective_credibility
                    * min(0.5 + source.similarity_score, 1.2)
                    * ml_conf
                )
                if ml_label == "Real News":
                    real_score += ml_weight * 0.8
                elif ml_label == "Fake News":
                    fake_score += ml_weight * 0.8

        updated_sources.append(
            dataclasses.replace(
                source,
                stance=stance,
                credibility_score=credibility,
                is_independent=is_independent,
            )
        )

    # Professional fact-check verdicts (ClaimReview) carry the most weight.
    for check in fact_checks:
        if check.stance == "SUPPORTS":
            supporting_count += 1
            real_score += 2.5 * 1.5
        elif check.stance == "CONTRADICTS":
            contradicting_count += 1
            fake_score += 2.5 * 1.5

    # Wikipedia is exactly one conservative evidence source for death claims.
    if wikipedia_check is not None and wikipedia_check.stance != "NEUTRAL":
        if wikipedia_check.stance == "SUPPORTS":
            supporting_count += 1
            real_score += 2.0 * 1.5
        else:
            contradicting_count += 1
            fake_score += 2.0 * 1.5

    if not updated_sources and not fact_checks and wikipedia_check is None:
        return (
            "Inconclusive", None, sources, 0, 0, 0,
            "All discovered sources were excluded (e.g., satire domains).",
        )

    total_score = real_score + fake_score
    articles_analyzed_count = sum(s.prediction is not None for s in updated_sources)

    # Zero full articles analyzed: snippets may carry signals, but capped and
    # still subject to the 1.5× margin rule — equal or close scores stay
    # Inconclusive rather than defaulting to whichever side tied-breaks.
    if articles_analyzed_count == 0:
        if total_score > 0:
            enough_stances = (
                supporting_count + contradicting_count >= 2 or strong_evidence
            )
            if enough_stances and real_score > fake_score * 1.5:
                confidence = min(real_score / total_score, 0.65)
                return (
                    "Real News", confidence, tuple(updated_sources),
                    supporting_count, contradicting_count, neutral_count,
                    "Verdict based on search snippets and pattern signals only "
                    "(0 full articles analyzed). Confidence is capped.",
                )
            if enough_stances and fake_score > real_score * 1.5:
                confidence = min(fake_score / total_score, 0.65)
                return (
                    "Fake News", confidence, tuple(updated_sources),
                    supporting_count, contradicting_count, neutral_count,
                    "Verdict based on search snippets and pattern signals only "
                    "(0 full articles analyzed). Confidence is capped.",
                )
            return (
                "Inconclusive", None, tuple(updated_sources),
                supporting_count, contradicting_count, neutral_count,
                "INCONCLUSIVE — snippet signals alone were too few, too weak, "
                "or too conflicting to verify this claim.",
            )
        return (
            "Inconclusive", None, tuple(updated_sources),
            supporting_count, contradicting_count, neutral_count,
            "INCONCLUSIVE — no readable articles and no explicit signals were "
            "found for this claim.",
        )

    if total_score < 1.0 or (contradicting_count == 0 and supporting_count == 0):
        return (
            "Inconclusive", None, tuple(updated_sources),
            supporting_count, contradicting_count, neutral_count,
            "INCONCLUSIVE — no explicit factual confirmations or contradictions "
            "found. ML predictions alone are insufficient to verify this claim.",
        )

    # A verdict needs a 1.5× margin AND at least two explicit stances, unless
    # a professional fact-check or explicit Wikipedia evidence is present.
    enough_stances = (supporting_count + contradicting_count) >= 2 or strong_evidence

    if real_score > fake_score * 1.5 and enough_stances:
        confidence = min(real_score / total_score, 0.98)
        summary = (
            f"Strong evidence supports the claim ({supporting_count} supporting "
            f"vs {contradicting_count} contradicting)."
        )
        return (
            "Real News", confidence, tuple(updated_sources),
            supporting_count, contradicting_count, neutral_count, summary,
        )

    if fake_score > real_score * 1.5 and enough_stances:
        confidence = min(fake_score / total_score, 0.98)
        summary = (
            f"Strong evidence contradicts the claim ({contradicting_count} "
            f"contradicting vs {supporting_count} supporting)."
        )
        return (
            "Fake News", confidence, tuple(updated_sources),
            supporting_count, contradicting_count, neutral_count, summary,
        )

    if not enough_stances:
        return (
            "Inconclusive", None, tuple(updated_sources),
            supporting_count, contradicting_count, neutral_count,
            "INCONCLUSIVE — only a single weak signal was found; reliable "
            "evidence is insufficient for a verdict.",
        )

    return (
        "Inconclusive", None, tuple(updated_sources),
        supporting_count, contradicting_count, neutral_count,
        "INCONCLUSIVE — evidence is conflicting or too weak to make a "
        "definitive call.",
    )
