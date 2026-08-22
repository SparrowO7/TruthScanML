"""Online Verification - colorful verdict UI on the evidence engine."""

import streamlit as st

from database.history import HistoryDatabaseError, save_prediction
from services.nli_stance import nli_available
from services.online_verification import (
    NewsSearchError,
    build_search_query,
    verify_headline,
)
from ui.theme import inject_theme, render_html, verdict_ring_html


st.set_page_config(page_title="Online Verification", page_icon="🔗", layout="wide")

inject_theme()

st.title("Online Verification")
st.caption("Search news coverage for a headline and analyze readable source articles.")

st.warning(
    "Online search is a supporting signal, not automated fact-checking. Search "
    "coverage and ML predictions should be checked against credible sources. "
    "Offline Prediction remains available when the internet is unavailable."
)
st.caption(
    "Free search providers rotate per request (DDG -> Bing -> Google) so no "
    "single one hits its rate limit, with GDELT as the final fallback. "
    "Articles are extracted in parallel, plus professional fact-check lookup "
    "and an optional local NLI model for stance assistance."
)

if "fast_mode" not in st.session_state:
    st.session_state.fast_mode = False
if "nli_enabled" not in st.session_state:
    st.session_state.nli_enabled = True

fast = st.checkbox(
    "⚡ Fast keyword-only check (skip full article download)",
    value=st.session_state.fast_mode,
    help="Analyse only titles and snippets. Instant, but less thorough.",
)
st.session_state.fast_mode = fast

nli_toggle = st.toggle(
    "🧠 NLI assist",
    value=st.session_state.nli_enabled,
    help="Local AI model that compares your claim's meaning against each "
    "article. Adds a few seconds on first use; the toggle disables it "
    "without uninstalling anything.",
)
st.session_state.nli_enabled = nli_toggle

# Status line reflects both the toggle and whether the model can load.
try:
    nli_ready = nli_available()
except Exception:
    nli_ready = False
if not nli_toggle:
    nli_status = "disabled (toggle is off)"
elif nli_ready:
    nli_status = "active (local model loaded)"
else:
    nli_status = (
        "not installed - run `venv\\Scripts\\pip install sentence-transformers`"
    )
st.caption(f"🧠 NLI assist: {nli_status}")

headline = st.text_input(
    "News headline or claim",
    placeholder="Example: Chandrayaan-3 landed on the Moon",
    help="Up to 20 news results are checked. Only titles relevant to your claim are analyzed.",
)

if st.button("Search and analyze", type="primary"):
    clean_headline = headline.strip()
    if len(clean_headline) < 5:
        st.warning("Enter a more specific headline or claim before searching.")
        st.stop()

    try:
        with st.status(
            f"Verifying: {build_search_query(clean_headline)}...", expanded=True
        ) as status:
            st.write("🔎 Searching news providers (rotating: DDG / Bing / Google)...")
            st.write("🧹 Filtering results relevant to your claim...")
            verification = verify_headline(
                clean_headline,
                fast_mode=st.session_state.fast_mode,
                nli_enabled=st.session_state.nli_enabled,
            )
            st.write(
                f"📰 {verification.search_results_found} results found, "
                f"{verification.sources_found} relevant sources."
            )
            if st.session_state.fast_mode:
                st.write("⚡ Fast mode: analyzing titles and snippets...")
            else:
                st.write("⬇️ Downloading and analyzing articles in parallel...")
            if st.session_state.nli_enabled and nli_ready:
                st.write("🧠 NLI comparing claim meaning against sources...")
            st.write("🧠 Checking stances, credibility and freshness...")
            st.write("⚖️ Weighing evidence and source trust...")
            st.write("🧮 Aggregating consensus...")
            status.update(label="✅ Verification complete", state="complete", expanded=False)
    except NewsSearchError as error:
        st.error(str(error))
        st.stop()
    except Exception:
        st.error(
            "Online verification could not be completed. Offline Prediction "
            "and the saved model are unaffected."
        )
        st.stop()

    label = verification.consensus_label or "Inconclusive"
    confidence = verification.consensus_confidence

    # ---- Metrics row (classic look) ----
    metrics = st.columns(4)
    metrics[0].metric("Results searched", verification.search_results_found)
    metrics[1].metric("Relevant sources", verification.sources_found)
    metrics[2].metric("Articles analyzed", verification.articles_analyzed)
    metrics[3].metric(
        "Average confidence",
        f"{confidence:.1%}" if confidence is not None else "Not available",
    )

    # ---- Colorful verdict banner (green / red / blue) ----
    if not verification.sources_found:
        st.warning(
            "No sufficiently relevant news articles were found for this claim.\n\n"
            "Possible reasons: the claim is too generic, very recent, or the "
            "search engine returned unrelated results."
        )
    elif label == "Fake News":
        st.error("### 🚫 Verdict: Fake News")
    elif label == "Real News":
        st.success("### ✅ Verdict: Real News")
    else:
        st.info("### ⚖️ Verdict: Inconclusive")

    # ---- Why? - glass ring card + evidence summary ----
    if verification.has_sufficient_relevant_sources or verification.evidence_summary:
        st.markdown("#### Why?")
        if confidence is not None:
            kind = {"Real News": "real", "Fake News": "fake"}.get(label, "inconcl")
            ring_html = verdict_ring_html(confidence, "CONFIDENCE", kind)
            render_html(
                f"""
                <div class="verdict-banner {kind}" style="padding:0.9rem 1.2rem;">
                    {ring_html}
                    <div>
                        <p class="v-sub" style="font-size:0.95rem;color:inherit;">
                            {verification.evidence_summary}
                        </p>
                    </div>
                </div>
                """
            )
        else:
            st.write(verification.evidence_summary)

        # ---- Colorful explicit counts ----
        col1, col2, col3 = st.columns(3)
        col1.success(f"🟢 **Supports:** {verification.supporting_count}")
        col2.error(f"🔴 **Contradicts:** {verification.contradicting_count}")
        col3.info(f"⚪ **Neutral:** {verification.neutral_count}")
        st.caption(
            "Stance counts cover every relevant source (snippets included). "
            "'Articles analyzed' counts pages downloaded and run through the "
            "ML model - a lower number means some sites blocked automated reading."
        )

    # ---- Professional fact-checks (ClaimReview) ----
    if verification.fact_checks:
        st.markdown("#### 🏛️ Professional fact-checks")
        for check in verification.fact_checks:
            color = {"SUPPORTS": "green", "CONTRADICTS": "red"}.get(check.stance, "blue")
            with st.expander(f"{check.publisher}: {check.title}", expanded=False):
                st.markdown(f"**Rating:** :{color}[{check.rating}]")
                if check.claim_text:
                    st.caption(f"Checked claim: {check.claim_text}")
                st.link_button("Open fact-check article", check.url)

    # ---- Wikipedia signal ----
    if verification.wikipedia_note:
        with st.expander("📚 Wikipedia death-claim check", expanded=False):
            st.write(verification.wikipedia_note)
            if verification.wikipedia_url:
                st.link_button("Open Wikipedia article", verification.wikipedia_url)

    # ---- Source matches (classic colorful expanders) ----
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
                " · ".join(value for value in (source.publisher, source.published_at) if value)
            )
            if source.snippet:
                st.write(source.snippet)

            if source.evidence_quote:
                st.markdown(f"> {source.evidence_quote}")

            stance_color = {
                "SUPPORTS": "green",
                "CONTRADICTS": "red",
                "NEUTRAL": "blue",
            }.get(source.stance, "gray")
            nli_note = (
                f" · 🧠 NLI {source.nli_stance} ({source.nli_score:.0%})"
                if source.nli_stance
                else ""
            )
            st.markdown(f"**Stance:** :{stance_color}[{source.stance}]{nli_note}")
            st.caption(
                f"Credibility Weight: {source.credibility_score:.1f} · "
                f"Independent Domain: {'Yes' if source.is_independent else 'No (Duplicate)'} · "
                f"Freshness: {source.freshness_score:.0%}"
            )
            st.caption(
                f"Entity match: {source.entity_score:.0%} · "
                f"Event match: {source.event_score:.0%} · "
                f"Overall relevance: {source.similarity_score:.0%}"
            )
            st.caption(f"Accepted because: {source.acceptance_reason}")

            st.link_button("Open source article", source.url)

            if source.prediction is not None:
                st.info(
                    f"Article ML Result: {source.prediction.label} "
                    f"({source.prediction.confidence:.1%} confidence)"
                )
            else:
                st.info("Article not analyzed: readable text was unavailable.")

    # ---- Save to local history (best effort) ----
    try:
        save_prediction(
            mode="Online",
            input_text=clean_headline,
            prediction=label,
            confidence=confidence,
            sources_found=verification.sources_found,
            articles_analyzed=verification.articles_analyzed,
        )
    except HistoryDatabaseError:
        st.warning("Verification completed, but it could not be saved to local history.")
