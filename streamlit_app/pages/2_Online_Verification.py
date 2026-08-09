"""Headline-based online verification with web-source discovery."""

import streamlit as st

from database.history import HistoryDatabaseError, save_prediction
from services.online_verification import NewsSearchError, verify_headline


st.set_page_config(page_title="Online Verification", page_icon="🔗", layout="wide")

st.title("Online Verification")
st.caption("Search news coverage for a headline and analyze readable source articles.")

st.warning(
    "Online search is a supporting signal, not automated fact-checking. Search "
    "coverage and ML predictions should be checked against credible sources. "
    "Offline Prediction remains available when the internet is unavailable."
)
st.caption(
    "No API key is needed. The search uses DuckDuckGo first and may use the "
    "DDGS public-search fallback when DuckDuckGo temporarily rejects a request."
)

headline = st.text_input(
    "News headline or claim",
    placeholder="Example: Chandrayaan-3 landed on the Moon",
    help="Up to 20 news results are checked. Only titles relevant to your claim are analyzed.",
)

if st.button("Search and analyze", type="primary"):
    if len(headline.strip()) < 5:
        st.warning("Enter a more specific headline or claim before searching.")
    else:
        try:
            with st.spinner("Searching news sources and analyzing readable articles..."):
                verification = verify_headline(headline)
        except NewsSearchError as error:
            st.error(str(error))
        except Exception:
            st.error(
                "Online verification could not be completed. Offline Prediction "
                "and the saved model are unaffected."
            )
        else:
            metrics = st.columns(4)
            metrics[0].metric("Results searched", verification.search_results_found)
            metrics[1].metric("Relevant sources", verification.sources_found)
            metrics[2].metric(
                "Articles analyzed",
                verification.articles_analyzed,
            )
            metrics[3].metric(
                "Average confidence",
                f"{verification.consensus_confidence:.1%}"
                if verification.consensus_confidence is not None
                else "Not available",
            )

            if not verification.sources_found:
                st.warning(
                    "No sufficiently relevant news articles were found for this claim.\n\n"
                    "Possible reasons: the claim is too generic, very recent, or the "
                    "search engine returned unrelated results."
                )
            elif not verification.has_sufficient_relevant_sources:
                st.warning(
                    "Unable to verify this claim with sufficient relevant sources. "
                    "At least two matching article titles are required."
                )
            elif verification.consensus_label == "Fake News":
                st.error("### Source-model consensus: Fake News")
            elif verification.consensus_label == "Real News":
                st.success("### Source-model consensus: Real News")
            elif verification.articles_analyzed:
                st.info("### Source-model consensus: Mixed or inconclusive")
            else:
                st.warning("Relevant sources were found, but no readable article text could be analyzed.")

            st.caption(
                "Consensus combines multi-source news coverage, fact-checking stance, "
                "domain authority, and trained ML model predictions as a supporting decision-support signal."
            )

            try:
                save_prediction(
                    mode="Online",
                    input_text=headline,
                    prediction=verification.consensus_label or "Inconclusive",
                    confidence=verification.consensus_confidence,
                    sources_found=verification.sources_found,
                    articles_analyzed=verification.articles_analyzed,
                )
            except HistoryDatabaseError:
                st.warning("Verification completed, but it could not be saved to local history.")

            st.subheader("Relevant source matches")
            st.caption(
                "Event claims use entity + event matching; entity-only searches use "
                "entity matching; general topics use relevance matching."
            )
            if not verification.sources:
                st.info("None of the searched article titles matched this claim closely enough.")
            for source in verification.sources:
                with st.expander(source.title, expanded=False):
                    st.caption(
                        " · ".join(
                            value
                            for value in (source.publisher, source.published_at)
                            if value
                        )
                    )
                    if source.snippet:
                        st.write(source.snippet)
                    st.link_button("Open source article", source.url)
                    st.caption(
                        f"Entity match: {source.entity_score:.0%} · "
                        f"Event match: {source.event_score:.0%} · "
                        f"Overall relevance: {source.similarity_score:.0%}"
                    )
                    st.caption(f"Accepted because: {source.acceptance_reason}")

                    if source.prediction is not None:
                        st.success(
                            f"Article ML result: {source.prediction.label} "
                            f"({source.prediction.confidence:.1%} confidence)"
                        )
                    else:
                        st.info("Article not analyzed: readable text was unavailable.")
