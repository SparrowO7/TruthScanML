"""Local, fault-tolerant SQLite storage for prediction history."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "streamlit_app" / "data" / "prediction_history.db"
MAX_HISTORY_RECORDS = 250


class HistoryDatabaseError(RuntimeError):
    """Raised when local history storage cannot be accessed."""


@dataclass(frozen=True)
class HistoryRecord:
    """A display-ready record read from the local prediction database."""

    created_at: str
    mode: str
    input_preview: str
    prediction: str
    confidence: float | None
    sources_found: int | None
    articles_analyzed: int | None

    def to_display_row(self) -> dict[str, str]:
        return {
            "Time (UTC)": self.created_at.replace("T", " ").replace("+00:00", ""),
            "Mode": self.mode,
            "Input": self.input_preview,
            "Prediction": self.prediction,
            "Confidence": f"{self.confidence:.1%}" if self.confidence is not None else "—",
            "Sources": str(self.sources_found) if self.sources_found is not None else "—",
            "Full reads": str(self.articles_analyzed)
            if self.articles_analyzed is not None
            else "—",
        }


def save_prediction(
    *,
    mode: str,
    input_text: str,
    prediction: str,
    confidence: float | None,
    sources_found: int | None = None,
    articles_analyzed: int | None = None,
) -> None:
    """Persist a compact local history record after a successful prediction."""

    if mode not in {"Offline", "Online"}:
        raise HistoryDatabaseError("Unsupported prediction mode.")

    preview = " ".join(input_text.split())[:240]
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    try:
        with _connection() as connection:
            connection.execute(
                """
                INSERT INTO prediction_history (
                    created_at, mode, input_preview, prediction, confidence,
                    sources_found, articles_analyzed
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    mode,
                    preview,
                    prediction,
                    confidence,
                    sources_found,
                    articles_analyzed,
                ),
            )
    except sqlite3.Error as error:
        raise HistoryDatabaseError("Unable to save local prediction history.") from error


def list_history(limit: int = MAX_HISTORY_RECORDS) -> list[HistoryRecord]:
    """Return recent records newest first without exposing article bodies."""

    safe_limit = min(max(limit, 1), MAX_HISTORY_RECORDS)
    try:
        with _connection() as connection:
            rows = connection.execute(
                """
                SELECT created_at, mode, input_preview, prediction, confidence,
                       sources_found, articles_analyzed
                FROM prediction_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
    except sqlite3.Error as error:
        raise HistoryDatabaseError("Unable to read local prediction history.") from error

    return [
        HistoryRecord(
            created_at=row["created_at"],
            mode=row["mode"],
            input_preview=row["input_preview"],
            prediction=row["prediction"],
            confidence=row["confidence"],
            sources_found=row["sources_found"],
            articles_analyzed=row["articles_analyzed"],
        )
        for row in rows
    ]


def clear_history() -> None:
    """Delete saved local records while preserving the database schema."""

    try:
        with _connection() as connection:
            connection.execute("DELETE FROM prediction_history")
    except sqlite3.Error as error:
        raise HistoryDatabaseError("Unable to clear local prediction history.") from error


def _connection() -> sqlite3.Connection:
    """Open and initialize one short-lived local SQLite connection."""

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            mode TEXT NOT NULL CHECK (mode IN ('Offline', 'Online')),
            input_preview TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL,
            sources_found INTEGER,
            articles_analyzed INTEGER
        )
        """
    )
    return connection
