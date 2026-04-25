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
    -- `action` is the LLM's advisory action, kept for backward compat &
    -- evaluation. The two columns below split out spec §13's distinction:
    --   final_decision_action  : what the Decision Engine returned (BUY/SELL/HOLD)
    --   executed_action        : what was actually placed (HOLD if engine HELD,
    --                            even when llm_action=BUY)
    action TEXT NOT NULL,
    final_decision_action TEXT,
    executed_action TEXT,
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
    -- spec §13 postmortem-friendly fields (nullable; back-filled when known)
    spread_at_entry REAL,
    spread_at_exit REAL,
    slippage REAL,
    rule_version TEXT,
    detected_pattern TEXT,
    trend_state TEXT,
    higher_timeframe_trend TEXT,
    event_risk_level TEXT,
    economic_events_nearby TEXT,
    sentiment_score REAL,
    sentiment_volume_spike INTEGER,
    blocked_by TEXT,
    final_reason TEXT,
    rule_chain TEXT,
    risk_reward REAL,
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
    -- spec §12 loss classification: A=正しい負け, B=入ってはいけない負け,
    -- C=利確/損切り設計ミス, D=波形認識ミス, E=指標依存ミス, F=実行系問題
    loss_category TEXT,
    -- Whether this postmortem is being treated as a system-accident (immediate
    -- fix) or a regular loss that must repeat ≥5 times before any rule edit.
    is_system_accident INTEGER NOT NULL DEFAULT 0,
    -- Free-form structured snapshot of the moment it was wrong:
    --   spread_at_entry, slippage, mfe/mae, rule_version, etc. (spec §13).
    context_json TEXT,
    -- Whether the proposed_rule has been promoted into a rule update.
    rule_applied INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

CREATE INDEX IF NOT EXISTS idx_postmortems_root_cause ON postmortems(root_cause);
CREATE INDEX IF NOT EXISTS idx_postmortems_loss_category ON postmortems(loss_category);

CREATE TABLE IF NOT EXISTS subscribers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backend TEXT NOT NULL,            -- "line" | "email" | ...
    user_id TEXT NOT NULL,            -- backend-specific recipient ID
    display_name TEXT,
    subscribed_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    notify_signal INTEGER NOT NULL DEFAULT 1,
    notify_verdict INTEGER NOT NULL DEFAULT 1,
    notify_lesson INTEGER NOT NULL DEFAULT 1,
    notify_summary INTEGER NOT NULL DEFAULT 1,
    min_confidence REAL NOT NULL DEFAULT 0.6,
    UNIQUE(backend, user_id)
);

CREATE INDEX IF NOT EXISTS idx_subscribers_active ON subscribers(active);
"""


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn) -> None:
        """Idempotent ALTER TABLE migrations for installs that pre-date a column.

        SQLite's CREATE TABLE IF NOT EXISTS only fires when the table is
        absent — it never adds new columns to a table created earlier.
        We list the columns each table should have and add anything missing.
        """
        targets = {
            "predictions": [
                ("spread_at_entry", "REAL"),
                ("spread_at_exit", "REAL"),
                ("slippage", "REAL"),
                ("rule_version", "TEXT"),
                ("detected_pattern", "TEXT"),
                ("trend_state", "TEXT"),
                ("higher_timeframe_trend", "TEXT"),
                ("event_risk_level", "TEXT"),
                ("economic_events_nearby", "TEXT"),
                ("sentiment_score", "REAL"),
                ("sentiment_volume_spike", "INTEGER"),
                ("blocked_by", "TEXT"),
                ("final_reason", "TEXT"),
                ("final_decision_action", "TEXT"),
                ("executed_action", "TEXT"),
                ("rule_chain", "TEXT"),
                ("risk_reward", "REAL"),
            ],
            "postmortems": [
                ("loss_category", "TEXT"),
                ("is_system_accident", "INTEGER NOT NULL DEFAULT 0"),
                ("context_json", "TEXT"),
                ("rule_applied", "INTEGER NOT NULL DEFAULT 0"),
            ],
        }
        for table, cols in targets.items():
            existing = {r["name"] for r in conn.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()}
            for name, sql_type in cols:
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")

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
        *,
        # spec §13: separate the LLM's advisory action from what the
        # Decision Engine concluded and what was actually executed.
        # Default `final_decision_action`/`executed_action` to `action`
        # so older callers behave the same.
        final_decision_action: str | None = None,
        executed_action: str | None = None,
        blocked_by: str | None = None,
        final_reason: str | None = None,
        rule_chain: str | None = None,
        risk_reward: float | None = None,
        detected_pattern: str | None = None,
        trend_state: str | None = None,
        higher_timeframe_trend: str | None = None,
        event_risk_level: str | None = None,
        economic_events_nearby: str | None = None,
        sentiment_score: float | None = None,
        sentiment_volume_spike: int | None = None,
        spread_at_entry: float | None = None,
        rule_version: str | None = None,
    ) -> int:
        fdec = final_decision_action if final_decision_action is not None else action
        execd = executed_action if executed_action is not None else action
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO predictions
                   (analysis_id, ts, symbol, interval, entry_price, action,
                    final_decision_action, executed_action,
                    confidence, reason, expected_direction,
                    expected_magnitude_pct, horizon_bars, invalidation_price,
                    blocked_by, final_reason, rule_chain, risk_reward,
                    detected_pattern, trend_state, higher_timeframe_trend,
                    event_risk_level, economic_events_nearby,
                    sentiment_score, sentiment_volume_spike,
                    spread_at_entry, rule_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis_id,
                    datetime.utcnow().isoformat(),
                    symbol,
                    interval,
                    entry_price,
                    action,
                    fdec,
                    execd,
                    confidence,
                    reason,
                    expected_direction,
                    expected_magnitude_pct,
                    horizon_bars,
                    invalidation_price,
                    blocked_by,
                    final_reason,
                    rule_chain,
                    risk_reward,
                    detected_pattern,
                    trend_state,
                    higher_timeframe_trend,
                    event_risk_level,
                    economic_events_nearby,
                    sentiment_score,
                    sentiment_volume_spike,
                    spread_at_entry,
                    rule_version,
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
        *,
        loss_category: str | None = None,
        is_system_accident: bool = False,
        # Free-form structured snapshot (spec §13). The caller assembles
        # whatever context is useful — final_decision/blocked_by/rule_chain/
        # pattern/event_risk/sentiment_spike/spread_pct/risk_reward — and
        # passes it as a dict; we serialise to JSON.
        context: dict | None = None,
    ) -> int:
        ctx_json = json.dumps(context, default=str) if context else None
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO postmortems
                   (prediction_id, ts, root_cause, narrative, proposed_rule, tags,
                    loss_category, is_system_accident, context_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction_id,
                    datetime.utcnow().isoformat(),
                    root_cause,
                    narrative,
                    proposed_rule,
                    tags,
                    loss_category,
                    1 if is_system_accident else 0,
                    ctx_json,
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

    # --- Subscribers (notification recipients) ---

    def add_subscriber(
        self, backend: str, user_id: str, display_name: str | None = None
    ) -> int:
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM subscribers WHERE backend = ? AND user_id = ?",
                (backend, user_id),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE subscribers SET active = 1, display_name = COALESCE(?, display_name) WHERE id = ?",
                    (display_name, existing["id"]),
                )
                return existing["id"]
            cur = conn.execute(
                """INSERT INTO subscribers
                   (backend, user_id, display_name, subscribed_at, active)
                   VALUES (?, ?, ?, ?, 1)""",
                (backend, user_id, display_name, datetime.utcnow().isoformat()),
            )
            return cur.lastrowid

    def deactivate_subscriber(self, backend: str, user_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE subscribers SET active = 0 WHERE backend = ? AND user_id = ?",
                (backend, user_id),
            )
            return cur.rowcount > 0

    def active_subscribers(
        self, backend: str | None = None, kind: str | None = None
    ) -> list[dict]:
        sql = "SELECT * FROM subscribers WHERE active = 1"
        args: list = []
        if backend:
            sql += " AND backend = ?"
            args.append(backend)
        if kind:
            sql += f" AND notify_{kind} = 1"
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
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
