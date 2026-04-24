"""SQLite storage for analyses, trades, and backtest runs."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    technical_signal TEXT NOT NULL,
    llm_action TEXT,
    llm_confidence REAL,
    llm_reason TEXT,
    final_action TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry REAL NOT NULL,
    exit REAL,
    size REAL NOT NULL,
    pnl REAL,
    analysis_id INTEGER,
    note TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    strategy TEXT NOT NULL,
    metrics_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analyses_symbol_ts ON analyses(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades(closed_at);
"""


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_analysis(
        self,
        symbol: str,
        snapshot: dict,
        technical_signal: str,
        final_action: str,
        llm_action: str | None = None,
        llm_confidence: float | None = None,
        llm_reason: str | None = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO analyses
                   (ts, symbol, snapshot_json, technical_signal,
                    llm_action, llm_confidence, llm_reason, final_action)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.utcnow().isoformat(),
                    symbol,
                    json.dumps(snapshot),
                    technical_signal,
                    llm_action,
                    llm_confidence,
                    llm_reason,
                    final_action,
                ),
            )
            return cur.lastrowid

    def save_trade(
        self,
        symbol: str,
        side: str,
        entry: float,
        size: float,
        opened_at: datetime,
        analysis_id: int | None = None,
        note: str | None = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO trades
                   (opened_at, symbol, side, entry, size, analysis_id, note)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (opened_at.isoformat(), symbol, side, entry, size, analysis_id, note),
            )
            return cur.lastrowid

    def close_trade(
        self, trade_id: int, exit_price: float, closed_at: datetime
    ) -> None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT side, entry, size FROM trades WHERE id = ?", (trade_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Trade {trade_id} not found")
            direction = 1 if row["side"] == "BUY" else -1
            pnl = (exit_price - row["entry"]) * row["size"] * direction
            conn.execute(
                """UPDATE trades SET exit = ?, closed_at = ?, pnl = ?
                   WHERE id = ?""",
                (exit_price, closed_at.isoformat(), pnl, trade_id),
            )

    def save_backtest(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        strategy: str,
        metrics: dict,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO backtest_runs
                   (ts, symbol, interval, start_date, end_date, strategy, metrics_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.utcnow().isoformat(),
                    symbol,
                    interval,
                    start_date,
                    end_date,
                    strategy,
                    json.dumps(metrics),
                ),
            )
            return cur.lastrowid

    def recent_trades(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT t.*, a.snapshot_json, a.llm_reason
                   FROM trades t LEFT JOIN analyses a ON t.analysis_id = a.id
                   WHERE t.closed_at IS NOT NULL
                   ORDER BY t.closed_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def recent_analyses(self, symbol: str | None = None, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM analyses WHERE symbol = ? ORDER BY ts DESC LIMIT ?",
                    (symbol, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM analyses ORDER BY ts DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
