"""Model information page for the Streamlit interface."""

import streamlit as st

from ui.theme import inject_theme


st.set_page_config(page_title="About Model", page_icon="ℹ️", layout="wide")

inject_theme()

st.title("About the Model")
st.caption("The established machine-learning pipeline behind each prediction.")

model_column, process_column = st.columns(2, gap="large")

with model_column:
    with st.container(border=True):
        st.markdown("#### Classifier")
        st.write("Logistic Regression")
        st.markdown("#### Labels")
        st.write("Real News and Fake News")

with process_column:
    with st.container(border=True):
        st.markdown("#### Feature extraction")
        st.write("TF-IDF Vectorizer")
        st.markdown("#### Text normalization")
        st.write("Lowercase, alphabetic filtering, and whitespace normalization")

st.subheader("Model integrity")
st.write(
    "This Streamlit interface will reuse the repository's existing saved model "
    "and vectorizer. It does not retrain, overwrite, or modify either artifact."
)

st.subheader("Important limitation")
st.write(
    "A machine-learning prediction reflects patterns learned from training data; "
    "it is not a substitute for source checking, editorial verification, or "
    "professional fact-checking."
)
