"""Optional local NLI (natural-language inference) stance assistant.

Uses a small cross-encoder NLI model (~100 MB, well under the 200 MB budget)
to score whether one article's title/snippet *entails* or *contradicts* the
user's claim. This upgrades pure keyword-pattern stances with actual meaning
comparison — without any paid API.

The model is optional: if ``sentence-transformers`` is not installed or the
model cannot be downloaded, every function degrades to "not available" and
the evidence engine falls back to its pattern-based behaviour.
"""

from functools import lru_cache

import streamlit as st


MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"
MIN_ACCEPT_SCORE = 0.80  # conservative bar for an NLI-assisted stance

# Fallback label order used by the SBERT cross-encoder NLI family.
_DEFAULT_LABELS = ["contradiction", "entailment", "neutral"]


@lru_cache(maxsize=1)
def _import_crossencoder():
    """Import CrossEncoder lazily so the app runs without the dependency."""

    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_model():
    """Load the NLI cross-encoder once per process, or None when impossible."""

    cross_encoder = _import_crossencoder()
    if cross_encoder is None:
        return None
    try:
        return cross_encoder(MODEL_NAME, max_length=256)
    except Exception:
        return None


def nli_available() -> bool:
    """Return whether the local NLI model can be used right now."""

    return _load_model() is not None


def _label_order(model) -> list[str]:
    """Read the label order from the model config, with a safe fallback."""

    try:
        id2label = model.model.config.id2label
        return [str(id2label[i]).lower() for i in range(len(id2label))]
    except Exception:
        return _DEFAULT_LABELS


def score_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str | None, float]]:
    """Score (claim, evidence) pairs.

    Returns one ``(stance, score)`` per pair where stance is
    "SUPPORTS" / "CONTRADICTS" / "NEUTRAL" and score is that label's
    probability. Pairs get ``(None, 0.0)`` when the model is unavailable.
    """

    if not pairs:
        return []

    model = _load_model()
    if model is None:
        return [(None, 0.0)] * len(pairs)

    try:
        scores = model.predict(list(pairs))
    except Exception:
        return [(None, 0.0)] * len(pairs)

    labels = _label_order(model)
    results: list[tuple[str | None, float]] = []
    for row in scores:
        try:
            probabilities = list(map(float, row))
        except (TypeError, ValueError):
            results.append((None, 0.0))
            continue
        best_index = probabilities.index(max(probabilities))
        best_label = labels[best_index] if best_index < len(labels) else "neutral"
        stance = {
            "entailment": "SUPPORTS",
            "contradiction": "CONTRADICTS",
        }.get(best_label, "NEUTRAL")
        results.append((stance, max(probabilities)))
    return results
