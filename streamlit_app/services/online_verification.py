"""Headline search, article extraction, and source-level ML analysis.

This module deliberately keeps all network work separate from Offline
Prediction. A search or extraction failure never changes the local pipeline.

Improvements applied (A–F):
  A  Article text extraction uses a fallback chain:
       <article> → <main> → all <p> tags
     so more sites yield readable text and fewer predictions are None.
  B  Search queries are cleaned before being sent to the provider:
       leading question words and trailing '?' are removed.
  C  A None prediction is now skipped entirely in consensus scoring instead
     of adding a neutral 0.5 + 0.5 that pollutes the weighted total.
  D  Handled in relevance.py (Hinglish stopwords).
  E  ML confidence is scaled by the article's relevance score before being
     added to the consensus pool (conservative — no accuracy claim).
  F  Debunk and confirm patterns are checked against the full scraped article
     body when available, not just the title + snippet.
"""

from dataclasses import dataclass, field
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

# Trusted fact-check and authoritative news domains.
# Indian sources prioritised for local news verification.
TRUSTED_DOMAINS = {
    # International fact-checkers
    "politifact.com", "snopes.com", "factcheck.org", "fullfact.org",
    # Indian fact-checkers
    "altnews.in", "boomlive.in", "pib.gov.in", "factcrescendo.com",
    "vishvasnews.com", "newschecker.in", "indiatoday.in",
    # International wire services / broadcasters
    "nasa.gov", "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "bloomberg.com", "nytimes.com", "washingtonpost.com", "aljazeera.com",
    "usatoday.com", "livescience.com", "britannica.com", "wikipedia.org",
    "nature.com", "who.int", "un.org",
    # Indian mainstream news
    "thehindu.com", "ndtv.com", "indianexpress.com",
    "timesofindia.com", "hindustantimes.com", "scroll.in",
    "thewire.in", "theprint.in", "moneycontrol.com", "zeenews.india.com",
    "aajtak.in", "abplive.com",
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
    r"\balive\b", r"\bstill\s+alive\b", r"\bis\s+alive\b",
    r"\bdenies\b", r"\bdenied\b", r"\brefutes\b",
    r"\bis\s+fine\b", r"\bis\s+well\b", r"\bis\s+safe\b",
    r"\bappears\s+in\b", r"\bseen\s+at\b", r"\bspotted\b",
    r"\bposts\s+on\b", r"\bshares\s+on\b", r"\btweets\b",
    r"\bnot\s+dead\b", r"\bdeath\s+hoax\b", r"\bdeath\s+rumou?r\b",
    r"\bno\s+truth\b",
]

# Verb/adjective forms that signal a death claim in the headline.
_DEATH_WORDS = re.compile(
    r"\b(died|dead|death|passed\s+away|no\s+more|is\s+no\s+more|nahi\s+rahe|gujar\s+gaye|mar\s+gaye)\b",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Fix 1 — Location / destination contradiction constants
# ---------------------------------------------------------------------------

# Canonical names for known space / geographic targets that claims often use.
# Each word in a group is treated as naming the same body.
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

# All individual location words (flat set for fast membership test).
_ALL_LOCATION_WORDS: frozenset[str] = frozenset(
    word for group in LOCATION_GROUPS for word in group
)


def _group_for(location_word: str) -> frozenset[str] | None:
    """Return the group frozenset that contains a given word, or None."""
    for group in LOCATION_GROUPS:
        if location_word in group:
            return group
    return None


def extract_claim_location(headline: str) -> str | None:
    """Extract a spatial target/destination word from the claim headline.

    Looks for preposition-linked location phrases such as:
      "landed on the sun", "on the moon", "to mars", "at the ISS".
    Returns the first matched location word that belongs to a known group,
    or None when no recognisable location is found.
    """
    lower = headline.lower()
    # Ordered from most specific to most general to reduce false positives.
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
    """Return whichever known location word appears most in a text block.

    Used to decide which spatial target an article is actually about.
    Returns None when no location word appears.
    """
    lower = text.lower()
    counts: dict[str, int] = {}
    for word in _ALL_LOCATION_WORDS:
        # Whole-word match only to avoid 'martial' matching 'mars' etc.
        count = len(re.findall(r"\b" + re.escape(word) + r"\b", lower))
        if count:
            counts[word] = count
    if not counts:
        return None
    return max(counts, key=lambda w: counts[w])


def _locations_contradict(claim_loc: str, article_loc: str) -> bool:
    """Return True when claim and article refer to different known locations.

    Two location words contradict when they belong to different groups —
    e.g. 'sun' and 'moon' are in separate groups, so they contradict.
    Two words from the same group (e.g. 'moon' and 'lunar') do not contradict.
    """
    if claim_loc == article_loc:
        return False
    claim_group = _group_for(claim_loc)
    article_group = _group_for(article_loc)
    if claim_group is None or article_group is None:
        return False
    return claim_group != article_group


# ---------------------------------------------------------------------------
# Death-claim contradiction engine
# (catches "Amitabh Bachchan died" / "PM Modi dead" type hoaxes)
# ---------------------------------------------------------------------------

def extract_death_subject(headline: str) -> str | None:
    """Return the subject name if the headline claims someone died.

    Works by checking whether a death-word appears in the headline alongside
    a plausible named entity (two or more capitalised words, or a known-role
    title followed by a name).  Returns the lowercased name string, or None.

    Examples:
      "Amitabh Bachchan died"       → "amitabh bachchan"
      "PM Modi is no more"          → "modi"
      "Chandrayaan 3 landing"       → None  (no death word)
    """
    if not _DEATH_WORDS.search(headline):
        return None

    # Try to extract a two-word capitalised name (e.g. "Amitabh Bachchan").
    name_match = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", headline)
    if name_match:
        return name_match.group(1).lower()

    # Fall back to a single capitalised name after a role title (e.g. "PM Modi").
    # The role itself is matched case-insensitively; the name must be title-case.
    role_match = re.search(
        r"\b(?:pm|cm|president|minister|actor|cricketer|politician|singer|director)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        headline,
    )
    if role_match:
        return role_match.group(1).lower()

    # Last resort: first capitalised word that appears after the start of the sentence.
    # Avoid picking up the very first word (which may be a sentence-starter).
    any_name = re.findall(r"\b([A-Z][a-z]{2,})\b", headline)
    # Skip the first capitalised word if it starts the sentence
    candidates = any_name[1:] if any_name and headline.strip().startswith(any_name[0]) else any_name
    if candidates:
        return candidates[0].lower()

    return None


def article_confirms_person_alive(person_name: str, article_text: str) -> bool:
    """Return True when an article says the named person is alive / active.

    Strategy:
      1. The article must actually mention the person's name (or a meaningful
         part of it) so we don't trigger on unrelated articles.
      2. At least one ALIVE_SIGNAL must appear anywhere in the article text.

    This is intentionally conservative: both conditions must hold.
    """
    lower = article_text.lower()

    # Check that at least one token of the person's name appears.
    name_tokens = [t for t in person_name.split() if len(t) > 2]
    if not name_tokens:
        return False
    name_present = any(token in lower for token in name_tokens)
    if not name_present:
        return False

    # Check for at least one alive signal.
    return any(re.search(pattern, lower) for pattern in ALIVE_SIGNALS)


# B: Question-starter words to strip from the beginning of a search query.
_QUESTION_STARTERS = re.compile(
    r"^(is|are|can|does|do|was|were|has|have|did|will|would|should|could|kya)\s+",
    flags=re.IGNORECASE,
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
    """A discovered source and its optional model prediction.

    article_has_debunk / article_has_confirm reflect pattern checks run
    against the full scraped article body (F), not only the title + snippet.
    They are False when article text was unavailable.
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
    # F: article-body-level stance signals
    article_has_debunk: bool = False
    article_has_confirm: bool = False
    
    # Evidence Engine extensions
    stance: str = "NEUTRAL"
    credibility_score: float = 1.0
    freshness_score: float = 1.0
    is_independent: bool = True


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
# B: Search query builder
# ---------------------------------------------------------------------------

def build_search_query(headline: str) -> str:
    """Return a cleaner search query derived from the user headline.

    Only surface-level cleanup is applied so the factual meaning is preserved:
      - Strip leading question-starter words (Is/Are/Kya …)
      - Strip trailing question marks
      - Collapse extra whitespace

    The original headline is still used for relevance matching and scoring;
    this cleaned form is sent to the search provider only.
    """
    query = headline.strip()
    query = re.sub(r"\?+$", "", query).strip()
    query = _QUESTION_STARTERS.sub("", query).strip()
    query = re.sub(r"\s+", " ", query).strip()
    return query or headline.strip()


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

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
        supporting_count, contradicting_count, neutral_count = 0, 0, len(analyses)
        summary = "Insufficient relevant sources found to verify this claim."
    else:
        raw_analyses = tuple(analyze_source(source, match) for source, match in relevant_sources)
        (
            consensus_label, 
            consensus_confidence, 
            analyses, 
            supporting_count, 
            contradicting_count, 
            neutral_count, 
            summary
        ) = calculate_consensus(headline, raw_analyses)

    return OnlineVerificationResult(
        headline=headline,
        search_results_found=len(search_results),
        sources=analyses,
        consensus_label=consensus_label,
        consensus_confidence=consensus_confidence,
        supporting_count=supporting_count,
        contradicting_count=contradicting_count,
        neutral_count=neutral_count,
        evidence_summary=summary
    )


# ---------------------------------------------------------------------------
# Search (B: cleaned query used here)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def search_news(headline: str) -> tuple[SourceResult, ...]:
    """Find up to twenty public news results for a headline without an API key.

    DuckDuckGo is preferred. DDGS's automatic backend is a no-key fallback for
    temporary provider blocks or rate limits; it does not affect Offline mode.
    The query sent to providers is cleaned (B) while the original headline
    is kept for all downstream relevance and scoring logic.
    """

    query = build_search_query(headline)  # B

    try:
        raw_results = DDGS().news(
            query=query,
            max_results=MAX_SEARCH_RESULTS,
            backend="duckduckgo",
        )
    except Exception:
        raw_results = []

    if not raw_results:
        try:
            raw_results = DDGS().news(
                query=query,
                max_results=MAX_SEARCH_RESULTS,
                backend="auto",
            )
        except Exception:
            raw_results = []

    if not raw_results:
        try:
            raw_results = search_google_news_rss(query)
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


def search_google_news_rss(query: str) -> list[dict[str, str]]:
    """Use Google News RSS as a no-key fallback when DDGS is rate-limited."""

    response = requests.get(
        "https://news.google.com/rss/search",
        params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
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


# ---------------------------------------------------------------------------
# Article extraction (A) and analysis (F)
# ---------------------------------------------------------------------------

def extract_article_text(url: str) -> str:
    """Retrieve readable paragraph text with a fallback chain (A).

    Extraction order:
      1. <article> tag paragraphs  — most news sites use this semantic element
      2. <main> tag paragraphs     — common fallback container
      3. All <p> tags on the page  — original behaviour, broadest net

    Each level is tried only if the previous one yields fewer than
    MIN_ARTICLE_CHARACTERS of clean text.
    """

    response = requests.get(url, headers=HTTP_HEADERS, timeout=10)
    response.raise_for_status()
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

    # A: Fallback chain
    article_text = _paragraphs_from(soup.find("article"))

    if len(article_text) < MIN_ARTICLE_CHARACTERS:
        article_text = _paragraphs_from(soup.find("main"))

    if len(article_text) < MIN_ARTICLE_CHARACTERS:
        # Broadest fallback: collect all <p> tags on the page
        article_text = " ".join(
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if p.get_text(" ", strip=True)
        )

    if len(article_text) < MIN_ARTICLE_CHARACTERS:
        raise ValueError("The article did not provide enough readable text.")

    return article_text


def _check_patterns(text: str) -> tuple[bool, bool]:
    """Return (has_debunk, has_confirm) for a given text block."""
    lower = text.lower()
    has_debunk = any(re.search(p, lower) for p in DEBUNK_PATTERNS)
    has_confirm = any(re.search(p, lower) for p in CONFIRM_PATTERNS)
    return has_debunk, has_confirm


def analyze_source(source: SourceResult, match: ClaimMatch) -> SourceAnalysis:
    """Extract one article, check body-level stance signals (F), and run ML."""

    article_text: str | None = None
    prediction: PredictionResult | None = None
    article_has_debunk = False
    article_has_confirm = False

    try:
        article_text = extract_article_text(source.url)
    except Exception:
        pass

    if article_text:
        prediction = predict_news(article_text)
        # F: check debunk/confirm patterns on the full article body
        article_has_debunk, article_has_confirm = _check_patterns(article_text)

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
        article_has_debunk=False,
        article_has_confirm=False,
    )


# ---------------------------------------------------------------------------
# Evidence Aggregation & Consensus Engine
# ---------------------------------------------------------------------------

import dataclasses

def _calculate_credibility(source: SourceAnalysis, domain: str) -> float:
    # Fact-checkers get maximum credibility
    fact_checkers = {"altnews.in", "boomlive.in", "factcrescendo.com", "vishvasnews.com", "newschecker.in", "politifact.com", "snopes.com", "factcheck.org", "fullfact.org"}
    if any(fc in domain for fc in fact_checkers):
        return 2.0
    
    # Trusted mainstream/official news get high credibility
    if any(td in domain for td in TRUSTED_DOMAINS) or any(td in source.publisher.lower() for td in TRUSTED_DOMAINS):
        return 1.5
    
    # Generic domains
    return 1.0


def _classify_stance(
    source: SourceAnalysis, 
    headline_lower: str, 
    death_subject: str | None, 
    claim_location: str | None
) -> str:
    snippet_text = f"{source.title} {source.snippet}".lower()
    has_debunk = source.article_has_debunk or any(re.search(p, snippet_text) for p in DEBUNK_PATTERNS)
    has_confirm = source.article_has_confirm or any(re.search(p, snippet_text) for p in CONFIRM_PATTERNS)
    
    # 1. Location Contradiction -> CONTRADICTS
    if claim_location is not None:
        article_location = _dominant_location(snippet_text)
        if article_location is not None and _locations_contradict(claim_location, article_location):
            return "CONTRADICTS"
            
    # 2. Death Contradiction -> CONTRADICTS
    if death_subject is not None:
        if article_confirms_person_alive(death_subject, snippet_text):
            return "CONTRADICTS"
            
    # 3. Scientific Hard Rules
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
        
    return "NEUTRAL"


def calculate_consensus(
    headline: str,
    sources: tuple[SourceAnalysis, ...],
) -> tuple[str | None, float | None, tuple[SourceAnalysis, ...], int, int, int, str]:
    """Return verdict, confidence, updated sources, counts, and explanation."""
    
    if not sources:
        return None, None, sources, 0, 0, 0, "No sources available for analysis."

    headline_lower = headline.lower().strip()
    claim_location = extract_claim_location(headline)
    death_subject = extract_death_subject(headline)
    
    updated_sources = []
    seen_domains = set()
    
    real_score = 0.0
    fake_score = 0.0
    
    supporting_count = 0
    contradicting_count = 0
    neutral_count = 0

    for source in sources:
        domain = urlparse(source.url).netloc.lower()
        
        # Priority 0a: Skip satire domains entirely (0 credibility, effectively hidden/ignored)
        if any(sd in domain for sd in SATIRE_DOMAINS):
            continue
            
        # Source Independence: Check if we've already seen this domain
        is_independent = domain not in seen_domains
        seen_domains.add(domain)
        
        credibility = _calculate_credibility(source, domain)
        # Downweight duplicate domains so they don't inflate confidence
        if not is_independent:
            credibility *= 0.2
            
        stance = _classify_stance(source, headline_lower, death_subject, claim_location)
        
        # Update metrics
        if stance == "SUPPORTS":
            supporting_count += 1
            real_score += credibility * 1.5
        elif stance == "CONTRADICTS":
            contradicting_count += 1
            fake_score += credibility * 1.5
        else:
            neutral_count += 1
            # ML Prediction acts as a supporting signal ONLY for neutral articles
            if source.prediction:
                ml_label = source.prediction.label
                ml_conf = source.prediction.confidence
                # Cap the ML contribution so it doesn't overpower explicit factual stances
                ml_weight = credibility * min(0.5 + source.similarity_score, 1.2) * ml_conf
                if ml_label == "Real News":
                    real_score += ml_weight * 0.8
                elif ml_label == "Fake News":
                    fake_score += ml_weight * 0.8

        updated_sources.append(dataclasses.replace(
            source,
            stance=stance,
            credibility_score=credibility,
            is_independent=is_independent
        ))
        
    articles_analyzed_count = sum(s.prediction is not None for s in updated_sources)
    
    if not updated_sources:
        return None, None, sources, 0, 0, 0, "All discovered sources were excluded (e.g., satire domains)."
        
    total_score = real_score + fake_score
    
    # Zero-evidence safety & Low evidence penalty
    if articles_analyzed_count == 0:
        # If no articles were actually scraped, we cannot have high confidence.
        # If there's strong snippet evidence, we cap it heavily.
        if total_score > 0:
            confidence = min(max(real_score, fake_score) / total_score, 0.65)
            label = "Real News" if real_score > fake_score else "Fake News"
            summary = "Verdict based on search snippets only (0 full articles analyzed). Confidence is capped."
            return label, confidence, tuple(updated_sources), supporting_count, contradicting_count, neutral_count, summary
        return None, None, tuple(updated_sources), supporting_count, contradicting_count, neutral_count, "No articles analyzed and snippets lacked explicit signals."

    if total_score < 1.0 or (contradicting_count == 0 and supporting_count == 0):
        # Only ML signals or very weak signals exist. Do not force a verdict.
        return None, None, tuple(updated_sources), supporting_count, contradicting_count, neutral_count, "No explicit factual confirmations or contradictions found. ML predictions alone are insufficient to verify this claim."

    if real_score > fake_score * 1.5:
        confidence = min(real_score / total_score, 0.98)
        label = "Real News"
        summary = f"Strong evidence supports the claim ({supporting_count} supporting vs {contradicting_count} contradicting)."
    elif fake_score > real_score * 1.5:
        confidence = min(fake_score / total_score, 0.98)
        label = "Fake News"
        summary = f"Strong evidence contradicts the claim ({contradicting_count} contradicting vs {supporting_count} supporting)."
    else:
        # Conflicting or perfectly balanced weak evidence
        label = None
        confidence = None
        summary = f"Evidence is conflicting or too weak to make a definitive call."

    return label, confidence, tuple(updated_sources), supporting_count, contradicting_count, neutral_count, summary
