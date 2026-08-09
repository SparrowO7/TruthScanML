"""Landing page for the AI-Powered Fake News Detection application."""

import streamlit as st


st.set_page_config(
    page_title="AI-Powered Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI-Powered Fake News Detection")
st.caption("Offline & Online News Verification using Machine Learning")

st.info(
    "This application is an educational decision-support tool. Its prediction "
    "should be checked against credible reporting and primary sources."
)

st.subheader("Verify news your way")
offline_column, online_column = st.columns(2, gap="large")

with offline_column:
    with st.container(border=True):
        st.markdown("#### Offline Prediction")
        st.write(
            "Paste news text to receive a local prediction using the existing "
            "TF-IDF and Logistic Regression pipeline."
        )
        st.caption("Designed to work without an internet connection.")
        st.page_link(
            "pages/1_Offline_Prediction.py",
            label="Open Offline Prediction",
            icon="📝",
            use_container_width=True,
        )

with online_column:
    with st.container(border=True):
        st.markdown("#### Online Verification")
        st.write(
            "Enter a headline. The application searches DuckDuckGo, extracts "
            "matching article text, and evaluates it with the same saved model."
        )
        st.caption("Only this mode needs a network connection.")
        st.page_link(
            "pages/2_Online_Verification.py",
            label="Open Online Verification",
            icon="🔗",
            use_container_width=True,
        )

st.subheader("How a prediction is produced")
input_step, clean_step, vector_step, predict_step = st.columns(4, gap="small")

with input_step:
    st.markdown("**1. Input**")
    st.caption("News text or a news headline")
with clean_step:
    st.markdown("**2. Preprocess**")
    st.caption("Normalize text using the established pipeline")
with vector_step:
    st.markdown("**3. TF-IDF**")
    st.caption("Convert words into numeric features")
with predict_step:
    st.markdown("**4. Classify**")
    st.caption("Logistic Regression predicts Fake or Real")

st.subheader("Application sections")
st.write(
    "Use the navigation in the sidebar to open Offline Prediction, Online "
    "Verification, Prediction History, or About Model."
)

with st.expander("Milestone 1 status", expanded=False):
    st.write(
        "Offline Prediction and Online Verification both use the unchanged saved "
        "model and TF-IDF vectorizer. Prediction History is stored locally with "
        "SQLite."
    )
