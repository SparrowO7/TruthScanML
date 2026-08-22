"""Conservative Wikipedia cross-check for death claims.

Rules (deliberately cautious):
  - Wikipedia lead explicitly states the person died → SUPPORTS evidence.
  - Wikipedia lead describes the person in present tense with a birth date
    and no death mention → CONTRADICTS evidence (alive signal).
  - Anything else (past tense, unclear, page missing) → NEUTRAL.

Wikipedia is only ever one evidence source; it never decides a verdict alone.
"""

from dataclasses import dataclass
import re
from urllib.parse import quote

import requests
import streamlit as st


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
HTTP_HEADERS = {
    "User-Agent": (
        "FakeNewsDetectionApp/1.0 (educational project; contact: student)"
    )
}
REQUEST_TIMEOUT = 8


class WikipediaSignal:
    """Stance labels shared with the evidence engine."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class WikipediaCheck:
    """One conservative Wikipedia-derived signal for a death claim."""

    stance: str
    note: str
    quote: str
    url: str


@st.cache_data(ttl=86_400, show_spinner=False)
def _resolve_title(person_name: str) -> str | None:
    """Find the closest English Wikipedia title for a person name."""

    try:
        response = requests.get(
            WIKIPEDIA_API,
            params={
                "action": "opensearch",
                "search": person_name,
                "limit": 1,
                "namespace": 0,
                "format": "json",
            },
            headers=HTTP_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        # opensearch shape: [query, [titles], [descriptions], [urls]]
        titles = data[1] if len(data) > 1 else []
        return titles[0] if titles else None
    except Exception:
        return None


@st.cache_data(ttl=86_400, show_spinner=False)
def _fetch_summary_extract(title: str) -> str:
    """Return the lead extract for a resolved Wikipedia title."""

    try:
        response = requests.get(
            WIKIPEDIA_SUMMARY.format(title=quote(title.replace(" ", "_"))),
            headers=HTTP_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return str(response.json().get("extract", ""))
    except Exception:
        return ""


def _death_sentence(extract: str) -> str | None:
    """Return the first sentence that explicitly states a death."""

    for sentence in re.split(r"(?<=[.!?])\s+", extract):
        if re.search(r"\bdied\b|\bpassed away\b|\bwas killed\b", sentence):
            return sentence
    return None


def _present_tense_bio(extract: str) -> bool:
    """True when the lead describes the subject in present tense."""

    lead = extract[:350]
    has_birth = "born" in lead or re.search(r"\bb\.\s?\d{4}", lead)
    present = re.search(r"\b(?:is|remains|continues to be)\s+(?:a|an|the)\b", lead)
    death_free = not re.search(r"\bdied\b|\bdeath\b|\bwas killed\b", extract[:500])
    return bool(has_birth and present and death_free)


def wikipedia_death_check(person_name: str) -> WikipediaCheck | None:
    """Produce one conservative Wikipedia evidence signal for a death claim.

    Returns None when nothing useful could be established — the claim then
    stays with the ordinary news-evidence pipeline.
    """

    name = " ".join(person_name.split())
    if not name:
        return None

    title = _resolve_title(name)
    if not title:
        return None

    extract = _fetch_summary_extract(title)
    if len(extract) < 80:
        return None

    url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

    death_sentence = _death_sentence(extract)
    if death_sentence:
        return WikipediaCheck(
            stance=WikipediaSignal.SUPPORTS,
            note=f"Wikipedia's lead explicitly states the death of {title}.",
            quote=death_sentence[:300],
            url=url,
        )

    if _present_tense_bio(extract):
        first_sentence = extract.split(". ")[0][:300]
        return WikipediaCheck(
            stance=WikipediaSignal.CONTRADICTS,
            note=(
                f"Wikipedia describes {title} in the present tense with a birth "
                "date and no death mention — alive signal."
            ),
            quote=first_sentence,
            url=url,
        )

    return WikipediaCheck(
        stance=WikipediaSignal.NEUTRAL,
        note=f"Wikipedia has an article for {title}, but no clear death or alive statement.",
        quote="",
        url=url,
    )
