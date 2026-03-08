"""SQLite log of live predictions + running signal-accuracy tracking."""

import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "prediction_log.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    predicted_on TEXT NOT NULL,
    created_at TEXT NOT NULL,
    current_price REAL,
    predicted_price REAL,
    signal TEXT,
    model_signal TEXT,
    sentiment REAL,
    actual_price REAL,
    correct_direction INTEGER,
    UNIQUE(ticker, predicted_on)
);
"""


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as conn:
        conn.execute(SCHEMA)


def log_prediction(ticker, predicted_on, current_price, predicted_price,
                   signal, model_signal=None, sentiment=None):
    """Insert (or ignore) a prediction for the given trading day."""
    init_db()
    with _conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO predictions
               (ticker, predicted_on, created_at, current_price, predicted_price,
                signal, model_signal, sentiment)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, str(predicted_on), datetime.now().isoformat(timespec="seconds"),
             float(current_price), float(predicted_price), signal,
             model_signal, sentiment),
        )


def update_actuals(ticker, df):
    """Fill actual prices for past predictions using a stock frame (Date/Price).

    Also computes whether the predicted direction was right.
    Returns the number of rows updated.
    """
    init_db()
    price_by_date = {
        str(row["Date"].date()): float(row["Price"])
        for _, row in df.iterrows()
    }
    updated = 0
    with _conn() as conn:
        rows = conn.execute(
            """SELECT id, predicted_on, current_price, predicted_price
               FROM predictions
               WHERE ticker = ? AND actual_price IS NULL""",
            (ticker,),
        ).fetchall()
        for r in rows:
            actual = price_by_date.get(r["predicted_on"])
            if actual is None:
                continue
            pred_dir = 1 if r["predicted_price"] > r["current_price"] else -1
            act_dir = 1 if actual > r["current_price"] else -1
            conn.execute(
                "UPDATE predictions SET actual_price = ?, correct_direction = ? WHERE id = ?",
                (actual, int(pred_dir == act_dir), r["id"]),
            )
            updated += 1
    return updated


def get_history(ticker, limit=30):
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            """SELECT * FROM predictions WHERE ticker = ?
               ORDER BY predicted_on DESC LIMIT ?""",
            (ticker, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def accuracy_summary(ticker):
    """Running accuracy of logged predictions that have actual outcomes."""
    init_db()
    with _conn() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS total,
                      SUM(CASE WHEN actual_price IS NOT NULL THEN 1 ELSE 0 END) AS resolved,
                      SUM(CASE WHEN correct_direction = 1 THEN 1 ELSE 0 END) AS correct
               FROM predictions WHERE ticker = ?""",
            (ticker,),
        ).fetchone()
    resolved = row["resolved"] or 0
    correct = row["correct"] or 0
    return {
        "logged": row["total"] or 0,
        "resolved": resolved,
        "correct": correct,
        "accuracy": round(correct / resolved * 100, 2) if resolved else None,
    }


def clear_ticker(ticker):
    init_db()
    with _conn() as conn:
        conn.execute("DELETE FROM predictions WHERE ticker = ?", (ticker,))


# ----------------------------------------------------------------------
# Watchlist (persisted in the same SQLite database)
# ----------------------------------------------------------------------

WATCHLIST_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    ticker TEXT PRIMARY KEY,
    added_at TEXT NOT NULL
);
"""


def _ensure_watchlist():
    with _conn() as conn:
        conn.execute(WATCHLIST_SCHEMA)


def get_watchlist():
    _ensure_watchlist()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT ticker, added_at FROM watchlist ORDER BY added_at"
        ).fetchall()
    return [dict(r) for r in rows]


def add_to_watchlist(ticker):
    _ensure_watchlist()
    with _conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (ticker, added_at) VALUES (?, ?)",
            (ticker, datetime.now().isoformat(timespec="seconds")),
        )
    return get_watchlist()


def remove_from_watchlist(ticker):
    _ensure_watchlist()
    with _conn() as conn:
        conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker,))
    return get_watchlist()
