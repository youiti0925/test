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

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    entry_price REAL NOT NULL,
    action TEXT NOT NULL,
    confidence REAL,
    reason TEXT,
    expected_direction TEXT NOT NULL,
    expected_magnitude_pct REAL NOT NULL,
    horizon_bars INTEGER NOT NULL,
    invalidation_price REAL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    actual_direction TEXT,
    actual_magnitude_pct REAL,
    max_favorable_pct REAL,
    max_adverse_pct REAL,
    invalidation_hit INTEGER,
    evaluated_at TEXT,
    evaluation_note TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_ts ON predictions(symbol, ts);

CREATE TABLE IF NOT EXISTS postmortems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL UNIQUE,
    ts TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    narrative TEXT NOT NULL,
    proposed_rule TEXT,
    tags TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

CREATE INDEX IF NOT EXISTS idx_postmortems_root_cause ON postmortems(root_cause);
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

    def save_prediction(
        self,
        analysis_id: int,
        symbol: str,
        interval: str,
        entry_price: float,
        action: str,
        confidence: float | None,
        reason: str | None,
        expected_direction: str,
        expected_magnitude_pct: float,
        horizon_bars: int,
        invalidation_price: float | None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO predictions
                   (analysis_id, ts, symbol, interval, entry_price, action,
                    confidence, reason, expected_direction,
                    expected_magnitude_pct, horizon_bars, invalidation_price)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis_id,
                    datetime.utcnow().isoformat(),
                    symbol,
                    interval,
                    entry_price,
                    action,
                    confidence,
                    reason,
                    expected_direction,
                    expected_magnitude_pct,
                    horizon_bars,
                    invalidation_price,
                ),
            )
            return cur.lastrowid

    def pending_predictions(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE status = 'PENDING' ORDER BY ts ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def update_prediction_evaluation(
        self,
        prediction_id: int,
        status: str,
        actual_direction: str | None,
        actual_magnitude_pct: float | None,
        max_favorable_pct: float | None,
        max_adverse_pct: float | None,
        invalidation_hit: bool,
        note: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE predictions
                   SET status = ?, actual_direction = ?, actual_magnitude_pct = ?,
                       max_favorable_pct = ?, max_adverse_pct = ?,
                       invalidation_hit = ?, evaluated_at = ?, evaluation_note = ?
                   WHERE id = ?""",
                (
                    status,
                    actual_direction,
                    actual_magnitude_pct,
                    max_favorable_pct,
                    max_adverse_pct,
                    1 if invalidation_hit else 0,
                    datetime.utcnow().isoformat(),
                    note,
                    prediction_id,
                ),
            )

    def wrong_predictions_without_postmortem(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT p.*, a.snapshot_json
                   FROM predictions p
                   LEFT JOIN analyses a ON p.analysis_id = a.id
                   LEFT JOIN postmortems pm ON pm.prediction_id = p.id
                   WHERE p.status = 'WRONG' AND pm.id IS NULL
                   ORDER BY p.evaluated_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def save_postmortem(
        self,
        prediction_id: int,
        root_cause: str,
        narrative: str,
        proposed_rule: str | None,
        tags: str | None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO postmortems
                   (prediction_id, ts, root_cause, narrative, proposed_rule, tags)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    prediction_id,
                    datetime.utcnow().isoformat(),
                    root_cause,
                    narrative,
                    proposed_rule,
                    tags,
                ),
            )
            return cur.lastrowid

    def relevant_postmortems(self, symbol: str, limit: int = 5) -> list[dict]:
        """Most recent post-mortems for this symbol — injected into the next
        analysis prompt so Claude sees its own past failures."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT pm.root_cause, pm.narrative, pm.proposed_rule, pm.tags,
                          p.symbol, p.action, p.expected_direction,
                          p.expected_magnitude_pct, p.actual_direction,
                          p.actual_magnitude_pct, p.ts AS predicted_at
                   FROM postmortems pm
                   JOIN predictions p ON pm.prediction_id = p.id
                   WHERE p.symbol = ?
                   ORDER BY pm.ts DESC LIMIT ?""",
                (symbol, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def lesson_summary(self) -> list[dict]:
        """Aggregate counts by root cause — quick view of where we go wrong."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT root_cause, COUNT(*) AS n
                   FROM postmortems
                   GROUP BY root_cause ORDER BY n DESC"""
            ).fetchall()
            return [dict(r) for r in rows]

    def dashboard_stats(self) -> dict:
        """Aggregate counters for the web dashboard."""
        with self._conn() as conn:
            total_analyses = conn.execute(
                "SELECT COUNT(*) AS n FROM analyses"
            ).fetchone()["n"]
            status_rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM predictions GROUP BY status"
            ).fetchall()
            by_status = {r["status"]: r["n"] for r in status_rows}
            settled = (
                by_status.get("CORRECT", 0)
                + by_status.get("PARTIAL", 0)
                + by_status.get("WRONG", 0)
            )
            wins = by_status.get("CORRECT", 0) + 0.5 * by_status.get("PARTIAL", 0)
            win_rate = (wins / settled) if settled else 0.0
            backtests = conn.execute(
                "SELECT COUNT(*) AS n FROM backtest_runs"
            ).fetchone()["n"]
            postmortems = conn.execute(
                "SELECT COUNT(*) AS n FROM postmortems"
            ).fetchone()["n"]
        return {
            "total_analyses": total_analyses,
            "predictions_by_status": by_status,
            "weighted_win_rate": win_rate,
            "backtests": backtests,
            "postmortems": postmortems,
        }

    def get_analysis(self, analysis_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
            if row is None:
                return None
            prediction = conn.execute(
                "SELECT * FROM predictions WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
            return {
                "analysis": dict(row),
                "prediction": dict(prediction) if prediction else None,
            }

    def list_predictions(
        self,
        status: str | None = None,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        sql = "SELECT * FROM predictions"
        clauses: list[str] = []
        args: list = []
        if status:
            clauses.append("status = ?")
            args.append(status)
        if symbol:
            clauses.append("symbol = ?")
            args.append(symbol)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
            return [dict(r) for r in rows]

    def list_postmortems(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT pm.*, p.symbol, p.action, p.expected_direction,
                          p.expected_magnitude_pct, p.actual_direction,
                          p.actual_magnitude_pct, p.ts AS predicted_at
                   FROM postmortems pm
                   JOIN predictions p ON pm.prediction_id = p.id
                   ORDER BY pm.ts DESC LIMIT ?""",
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
