"""Read-only access to the repository's established ML inference pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import streamlit as st

from utils.preprocessing import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "model" / "fake_news_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "model" / "tfidf_vectorizer.pkl"


class ModelLoadError(RuntimeError):
    """Raised when the existing saved artifacts cannot be loaded safely."""


@dataclass(frozen=True)
class PredictionResult:
    """A display-ready result produced by the unchanged trained model."""

    label: str
    confidence: float


@st.cache_resource(show_spinner="Loading the saved prediction model...")
def load_artifacts() -> tuple[Any, Any]:
    """Load the existing model and vectorizer once per Streamlit process.

    The artifacts are opened only for reading. This function never trains,
    serializes, or changes them.
    """

    if not MODEL_PATH.is_file():
        raise ModelLoadError(f"Model file not found at {MODEL_PATH}")
    if not VECTORIZER_PATH.is_file():
        raise ModelLoadError(f"Vectorizer file not found at {VECTORIZER_PATH}")

    try:
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    except Exception as error:
        raise ModelLoadError("Unable to read the saved model artifacts.") from error


def predict_news(text: str) -> PredictionResult:
    """Run the same clean → vectorize → predict sequence as FastAPI."""

    model, vectorizer = load_artifacts()
    cleaned_text = clean_text(text)
    features = vectorizer.transform([cleaned_text])
    prediction = int(model.predict(features)[0])

    label = "Fake News" if prediction == 1 else "Real News"
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))

    return PredictionResult(label=label, confidence=confidence)
