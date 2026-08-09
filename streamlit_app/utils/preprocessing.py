"""Text preprocessing that preserves the existing FastAPI inference behavior."""

import re


def clean_text(text: str) -> str:
    """Normalize text exactly as ``api.main.clean_text`` does."""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
