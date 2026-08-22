"""Offline prediction page backed by the existing saved ML artifacts."""

import streamlit as st

from database.history import HistoryDatabaseError, save_prediction
from services.inference import ModelLoadError, predict_news
from ui.theme import inject_theme, render_html, verdict_ring_html


st.set_page_config(page_title="Offline Prediction", page_icon="📝", layout="wide")

inject_theme()

st.title("Offline Prediction")
st.caption("Classify pasted news text without an internet connection.")

st.info("Predictions are generated locally with the repository's saved ML pipeline.")

news_text = st.text_area(
    "News text",
    placeholder="Paste the article or news claim you want to assess...",
    height=260,
    help="Your text will be processed locally when this feature is enabled.",
)

if st.button("Analyze news", type="primary"):
    if not news_text.strip():
        st.warning("Paste some news text before requesting a prediction.")
    else:
        try:
            with st.spinner("Analyzing text with the saved model..."):
                result = predict_news(news_text)
        except ModelLoadError as error:
            st.error(f"The local prediction model could not be loaded: {error}")
        except Exception:
            st.error(
                "The prediction could not be completed. The saved model and "
                "vectorizer have not been changed."
            )
        else:
            kind = "fake" if result.label == "Fake News" else "real"
            icon = "🚫" if kind == "fake" else "✅"
            title = "FAKE NEWS" if kind == "fake" else "REAL NEWS"
            sub = (
                "The model's patterns suggest this text resembles fake news."
                if kind == "fake"
                else "The model's patterns suggest this text resembles real news."
            )
            ring_html = verdict_ring_html(result.confidence, "CONFIDENCE", kind)
            render_html(
                f"""
                <div class="verdict-banner {kind}">
                    {ring_html}
                    <div>
                        <p class="v-title">{icon} {title}</p>
                        <p class="v-sub">{sub}</p>
                    </div>
                </div>
                """
            )

            prediction_column, confidence_column = st.columns(2)
            prediction_column.metric("Model output", result.label)
            confidence_column.metric("Confidence", f"{result.confidence:.1%}")

            st.caption(
                "Confidence expresses the model's certainty for this prediction; "
                "it is not a measure of factual truth."
            )

            try:
                save_prediction(
                    mode="Offline",
                    input_text=news_text,
                    prediction=result.label,
                    confidence=result.confidence,
                )
            except HistoryDatabaseError:
                st.warning("Prediction completed, but it could not be saved to local history.")
