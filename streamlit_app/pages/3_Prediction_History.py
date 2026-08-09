"""Local SQLite-backed prediction history."""

import streamlit as st

from database.history import HistoryDatabaseError, clear_history, list_history


st.set_page_config(page_title="Prediction History", page_icon="🕘", layout="wide")

st.title("Prediction History")
st.caption("Review recent Offline Prediction and Online Verification results.")

st.info(
    "History is stored locally in SQLite on the machine running the app. It is "
    "not sent to an external service."
)

try:
    records = list_history()
except HistoryDatabaseError:
    st.error("Prediction history could not be opened. Existing predictions are unaffected.")
    records = []

if not records:
    st.info("No predictions have been saved yet. Run an Offline or Online prediction first.")
else:
    st.dataframe(
        [record.to_display_row() for record in records],
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Clear local history"):
        confirmed = st.checkbox(
            "I understand that this removes all saved prediction records from this device.",
            key="confirm_history_clear",
        )
        if st.button("Clear all history", disabled=not confirmed):
            try:
                clear_history()
            except HistoryDatabaseError:
                st.error("History could not be cleared.")
            else:
                st.success("Local prediction history was cleared.")
                st.rerun()
