"""
Shared SQLite database for all signal collectors.
DB file: stock_intel/data/signals.db
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "signals.db")


def get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT NOT NULL,       -- 'google_trends' | 'news' | 'reddit' | 'earnings'
                theme       TEXT NOT NULL,
                keyword     TEXT,                -- specific keyword that triggered (optional)
                score       REAL,                -- normalized 0-100
                raw_value   REAL,                -- raw value from source
                spike_factor REAL,               -- ratio vs baseline (>1.5 = notable)
                trend_dir   TEXT,                -- 'rising' | 'stable' | 'falling'
                evidence    TEXT,                -- JSON: source-specific detail
                ts          INTEGER NOT NULL     -- unix timestamp
            );

            CREATE INDEX IF NOT EXISTS idx_signals_theme  ON signals(theme);
            CREATE INDEX IF NOT EXISTS idx_signals_source ON signals(source);
            CREATE INDEX IF NOT EXISTS idx_signals_ts     ON signals(ts);

            CREATE TABLE IF NOT EXISTS theme_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                theme       TEXT NOT NULL,
                date        TEXT NOT NULL,       -- YYYY-MM-DD
                score_total REAL,                -- aggregated score across all sources
                sources     TEXT,                -- JSON: {source: score}
                status      TEXT,                -- 'emerging' | 'confirmed' | 'fading' | 'stable'
                UNIQUE(theme, date)
            );
        """)


if __name__ == "__main__":
    init_db()
    print(f"DB initialized at {os.path.abspath(DB_PATH)}")
