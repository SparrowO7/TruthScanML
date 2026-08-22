"""Lightweight claim-to-title relevance scoring for online verification."""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher


# D: Extended with common Hinglish noise words so they don't pollute
# relevance scoring. These are not translated — just filtered as stop words,
# the same way English filler words like "the", "is", "are" are filtered.
STOP_WORDS = {
    # English stop words
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "with",
    # Common Hinglish noise words (verb forms, postpositions, particles)
    # These carry no entity/event meaning in a news relevance context.
    "hua", "hui", "hue", "hai", "hain", "ho", "tha", "thi",
    "ka", "ki", "ke", "ko", "se", "ne", "par", "kya", "jo", "bhi",
    "aur", "ya", "nahi", "na", "gaya", "gayi", "gaye", "raha", "rahi",
    "wala", "wali", "wale", "mein", "pe",
}

EVENT_GROUPS = {
    # Expanded death group to also accept alive signals so debunking articles aren't discarded.
    # Hindi/Hinglish variants (dehant, swargwasi, inteqal, wafat…) included.
    "death": {"dead", "died", "dies", "death", "killed", "murdered", "passed away",
              "alive", "safe", "well", "hoax", "rumour", "rumor",
              "dehant", "swargwasi", "swargwas", "inteqal", "wafat",
              "mar gaya", "mar gaye", "nahi rahe", "gujar gaye", "guzar gaye"},
    "arrest": {"arrested", "arrest", "detained", "custody", "released",
               "girftar", "giraftaar", "hua giraftaar"},
    "resignation": {"resigns", "resigned", "resignation", "steps down", "istifa"},
    "retirement": {"retires", "retired", "retirement", "retire", "quits cricket",
                   "hangs up his boots", "calls it quits"},
    "election": {"wins", "won", "elected", "election", "vote", "appointed",
                 "jeeta", "jeet", "chune gaye", "vote diya"},
    "disaster": {"earthquake", "flood", "fire", "cyclone", "explosion", "crash",
                 "accident", "bhukamp", "baadh", "toofan", "dhamaka", "durghatna"},
    "ban": {"banned", "ban", "prohibited", "pabandi", "pratibandh"},
    "launch": {"launched", "launch", "liftoff", "mission", "deployed", "udan"},
}
GENERIC_CAPITAL_WORDS = {"former", "indian", "prime", "president", "minister"}
THRESHOLDS = {"entity_only": 0.40, "entity_event": 0.60, "general": 0.45}


@dataclass(frozen=True)
class ClaimMatch:
    claim_type: str
    entity_score: float
    event_score: float
    overall_score: float
    accepted: bool
    reason: str


def relevance_score(claim: str, article_title: str) -> float:
    """Return a transparent 0-1 lexical relevance score.

    This is intentionally a lightweight fuzzy matcher, not a factual verdict or
    a semantic model. It rejects titles that merely mention one named entity.
    """

    normalized_claim = _normalize(claim)
    normalized_title = _normalize(article_title)
    claim_terms = set(_meaningful_terms(normalized_claim))
    title_terms = set(_meaningful_terms(normalized_title))

    if not claim_terms or not title_terms:
        return 0.0

    claim_coverage = len(claim_terms & title_terms) / len(claim_terms)
    text_similarity = SequenceMatcher(None, normalized_claim, normalized_title).ratio()
    return round(max(claim_coverage, text_similarity), 2)


def entities_match(claim: str, article_title: str) -> bool:
    """Require a matching person/entity instead of shared role words alone."""

    claim_entities = _extract_entities(claim)
    title_entities = _extract_entities(article_title)
    if not claim_entities or not title_entities:
        return False

    for claim_entity in claim_entities:
        for title_entity in title_entities:
            claim_tokens = set(claim_entity.split())
            title_tokens = set(title_entity.split())
            if claim_tokens & title_tokens:
                return True
    return False


def events_match(claim: str, article_title: str) -> bool:
    """Require the same event family, such as death or arrest."""

    claim_events = _extract_events(claim)
    title_events = _extract_events(article_title)
    return bool(claim_events and title_events and claim_events & title_events)


def has_event(claim: str) -> bool:
    """Return whether a claim makes a recognisable event assertion."""

    return bool(_extract_events(claim))


def is_entity_only_claim(claim: str) -> bool:
    """Identify short searches such as 'Donald Trump' or 'Narendra Modi'."""

    normalized_terms = " ".join(_meaningful_terms(_normalize(claim)))
    return normalized_terms in _extract_entities(claim)


def evaluate_claim_match(claim: str, article_title: str) -> ClaimMatch:
    """Apply dynamic entity/event/topic rules and return transparent scores."""

    overall_score = relevance_score(claim, article_title)
    entity_score = _entity_score(claim, article_title)
    event_score = _event_score(claim, article_title)

    if has_event(claim):
        claim_type = "entity_event"
        accepted = (
            entity_score > 0
            and event_score > 0
            and overall_score >= THRESHOLDS[claim_type]
        )
        reason = "Matching entity and event" if accepted else "Entity, event, or relevance did not match"
    elif is_entity_only_claim(claim):
        claim_type = "entity_only"
        accepted = entity_score > 0 and overall_score >= THRESHOLDS[claim_type]
        reason = "Matching named entity" if accepted else "Named entity did not match closely enough"
    else:
        claim_type = "general"
        accepted = overall_score >= THRESHOLDS[claim_type]
        reason = "Relevant general topic" if accepted else "Topic relevance below threshold"

    return ClaimMatch(claim_type, entity_score, event_score, overall_score, accepted, reason)


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\bpm\b", "prime minister", text)
    # Map non-standard death phrases to ensure event detection catches them
    text = re.sub(
        r"\b(died|dies|death|killed|passed away|no\s+more|is\s+no\s+more|nahi\s+rahe|gujar\s+gaye|guzar\s+gaye|mar\s+gaya|mar\s+gaye|inteqal|wafat)\b",
        "dead",
        text
    )
    return re.sub(r"[^a-z\s]", " ", text)


def _meaningful_terms(text: str) -> list[str]:
    return [term for term in text.split() if term not in STOP_WORDS]


def _extract_entities(text: str) -> set[str]:
    """Extract practical person/entity candidates without a large NLP download."""

    normalized = _normalize(text)
    entities: set[str] = set()

    # Handles names after roles: "Prime Minister Narendra Modi" and "PM Modi".
    role_match = re.search(
        r"\b(?:prime minister|president|chief minister|pm)\s+([a-z]+(?:\s+[a-z]+){0,2})",
        normalized,
    )
    if role_match:
        candidate = " ".join(
            token
            for token in role_match.group(1).split()
            if token not in STOP_WORDS and token not in {"dead", "died", "dies", "death"}
        )
        if candidate:
            entities.add(candidate)

    # Also handles ordinary two-word personal names in article titles.
    for match in re.finditer(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", text):
        entities.add(_normalize(match.group(1)).strip())

    # For topic/location claims such as "Bangladesh election", use a single
    # capitalized entity only when a stronger multi-word person entity is absent.
    if not entities:
        for match in re.finditer(r"\b([A-Z][a-z]+)\b", text):
            candidate = _normalize(match.group(1)).strip()
            if candidate and candidate not in GENERIC_CAPITAL_WORDS:
                entities.add(candidate)

    return entities


def _extract_events(text: str) -> set[str]:
    normalized = _normalize(text)
    return {
        event_name
        for event_name, keywords in EVENT_GROUPS.items()
        if any(keyword in normalized for keyword in keywords)
    }


def _entity_score(claim: str, article_title: str) -> float:
    claim_entities = _extract_entities(claim)
    title_entities = _extract_entities(article_title)
    best_score = 0.0
    for claim_entity in claim_entities:
        for title_entity in title_entities:
            claim_tokens = set(claim_entity.split())
            title_tokens = set(title_entity.split())
            best_score = max(best_score, len(claim_tokens & title_tokens) / len(claim_tokens))
    return round(best_score, 2)


def _event_score(claim: str, article_title: str) -> float:
    claim_events = _extract_events(claim)
    if not claim_events:
        return 1.0
    title_events = _extract_events(article_title)
    return round(len(claim_events & title_events) / len(claim_events), 2)
