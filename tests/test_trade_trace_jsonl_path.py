"""Tests for PR #13: persist live decision_trace JSONL path on trades.

Pinned guarantees:
  * `trades.trace_jsonl_path` column exists on a brand-new DB.
  * `Storage._migrate()` adds the column idempotently to a pre-existing
    DB that lacks it (simulates an installed user upgrading).
  * `Storage.save_trade(trace_jsonl_path=...)` round-trips.
  * Existing callers that don't pass `trace_jsonl_path` keep working.
  * `cmd_trade --broker paper --trace-out-default` records the resolved
    decision_traces.jsonl path on the saved trade row.
  * `cmd_trade --broker paper` without trace flags leaves
    `trade.trace_jsonl_path` as NULL.
  * HOLD / dry-run produce no trade row (existing behavior preserved).

Out of scope (later PRs):
  * predictions.trace_jsonl_path
  * cmd_trade calling save_prediction()
  * live_trace_runs table for HOLD/dry-run
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.storage import Storage


# ─── Schema / migration ────────────────────────────────────────────────────


def _trade_columns(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("PRAGMA table_info(trades)").fetchall()
        return {r[1] for r in rows}
    finally:
        conn.close()


def test_new_db_has_trace_jsonl_path_column(tmp_path: Path):
    """Storage(...) on a brand-new file must produce a `trades` table that
    already includes `trace_jsonl_path`."""
    db = tmp_path / "new.db"
    Storage(db)
    assert "trace_jsonl_path" in _trade_columns(db)


def test_migration_adds_trace_jsonl_path_to_pre_existing_db(tmp_path: Path):
    """Simulate an installed user whose `trades` table predates this PR:
    create the table without `trace_jsonl_path`, then open via Storage and
    confirm `_migrate()` ALTERs the column in."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE trades (
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
                note TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    assert "trace_jsonl_path" not in _trade_columns(db)
    Storage(db)  # triggers _migrate()
    assert "trace_jsonl_path" in _trade_columns(db)


def test_migration_is_idempotent(tmp_path: Path):
    """Opening the same DB twice must not raise (no duplicate ADD COLUMN)."""
    db = tmp_path / "twice.db"
    Storage(db)
    Storage(db)  # second open re-runs _migrate; must be a no-op
    assert "trace_jsonl_path" in _trade_columns(db)


# ─── save_trade kwarg ──────────────────────────────────────────────────────


def test_save_trade_persists_trace_jsonl_path(tmp_path: Path):
    storage = Storage(tmp_path / "fx.db")
    base = datetime(2025, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    path = "/tmp/runs/live/bt_x/decision_traces.jsonl"
    tid = storage.save_trade(
        symbol="USDJPY=X", side="BUY",
        entry=150.0, size=1.0, opened_at=base,
        broker="paper",
        trace_jsonl_path=path,
    )
    assert tid >= 1
    with sqlite3.connect(tmp_path / "fx.db") as conn:
        row = conn.execute(
            "SELECT trace_jsonl_path FROM trades WHERE id = ?", (tid,)
        ).fetchone()
    assert row[0] == path


def test_save_trade_without_trace_jsonl_path_stores_null(tmp_path: Path):
    """Existing callers (no kwarg) must keep working; column stays NULL."""
    storage = Storage(tmp_path / "fx.db")
    base = datetime(2025, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    tid = storage.save_trade(
        symbol="USDJPY=X", side="BUY",
        entry=150.0, size=1.0, opened_at=base,
        broker="paper",
    )
    with sqlite3.connect(tmp_path / "fx.db") as conn:
        row = conn.execute(
            "SELECT trace_jsonl_path FROM trades WHERE id = ?", (tid,)
        ).fetchone()
    assert row[0] is None


# ─── cmd_trade integration ─────────────────────────────────────────────────


def _ohlcv(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3,
         "close": close, "volume": [1000] * n},
        index=idx,
    )


def _build_trade_namespace(**overrides):
    ns = argparse.Namespace(
        symbol="USDJPY=X",
        interval="1h",
        period="60d",
        capital=10_000.0,
        risk_pct=None,
        stop_atr=None,
        tp_atr=None,
        broker="paper",
        dry_run=True,
        confirm_demo=False,
        trace_out=None,
        trace_out_default=False,
        overwrite=False,
        no_news=True,
        news_limit=0,
        no_correlation=True,
        no_events=False,
        event_window_hours=48,
        no_lessons=True,
        lesson_limit=0,
        no_sentiment=True,
        no_higher_tf=True,
        no_llm=True,
        strategy_config=None,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def cmd_trade_with_real_storage(monkeypatch, tmp_path):
    """Like the cmd_trade_with_stub fixture in test_cmd_trade_live_trace.py
    but with a *real* Storage so we can read the persisted trade row back.

    Returns (cmd_fn, storage, db_path, cwd, cfg).
    """
    from src.fx import cli as cli_mod

    df = _ohlcv(200, seed=11)

    def fake_fetch_ohlcv(symbol, interval, period=None, start=None, end=None):
        return df

    monkeypatch.setattr(cli_mod, "fetch_ohlcv", fake_fetch_ohlcv)
    monkeypatch.setattr(cli_mod, "_print", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)

    db_path = tmp_path / "fx.db"
    storage = Storage(db_path)

    class Cfg:
        anthropic_api_key = None
        model = "claude-sonnet-4-6"
        effort = "medium"
        db_path = ":memory:"

    return cli_mod.cmd_trade, storage, db_path, tmp_path, Cfg()


def _force_buy(monkeypatch):
    from src.fx import cli as cli_mod
    from src.fx.decision_engine import Decision

    def stub_buy(*, technical_signal, pattern, higher_timeframe_trend,
                 risk_reward, risk_state, llm_signal=None,
                 waveform_bias=None, min_confidence=0.6, min_risk_reward=1.5):
        return Decision(
            action="BUY", confidence=0.9, reason="forced BUY for test",
            blocked_by=(),
            rule_chain=("risk_gate", "technical_directionality",
                        "pattern_check", "higher_tf_alignment",
                        "risk_reward_floor"),
            advisory={},
        )
    monkeypatch.setattr(cli_mod, "decide_action", stub_buy)


def _read_trade_row(db_path: Path) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def test_paper_actual_with_trace_default_records_jsonl_path(
    cmd_trade_with_real_storage, monkeypatch
):
    """The real-execution paper path with --trace-out-default must store
    the resolved decision_traces.jsonl path on the trade row."""
    cmd, storage, db_path, cwd, cfg = cmd_trade_with_real_storage
    _force_buy(monkeypatch)

    args = _build_trade_namespace(dry_run=False, trace_out_default=True)
    rc = cmd(args, cfg, storage)
    assert rc == 0

    row = _read_trade_row(db_path)
    assert row is not None
    assert row["trace_jsonl_path"], (
        f"trade.trace_jsonl_path must be non-empty when --trace-out-default "
        f"is set; row={row!r}"
    )
    p = Path(row["trace_jsonl_path"])
    assert p.exists(), f"trace path {p} must point at a file that exists"
    assert p.name == "decision_traces.jsonl"
    # Must point inside runs/live/<run_id>/
    assert "runs" in p.parts and "live" in p.parts


def test_paper_actual_with_explicit_trace_out_records_jsonl_path(
    cmd_trade_with_real_storage, monkeypatch
):
    """`--trace-out <dir>` (explicit) must also be persisted."""
    cmd, storage, db_path, cwd, cfg = cmd_trade_with_real_storage
    _force_buy(monkeypatch)

    explicit = cwd / "explicit_dest"
    args = _build_trade_namespace(
        dry_run=False, trace_out=str(explicit),
    )
    rc = cmd(args, cfg, storage)
    assert rc == 0

    row = _read_trade_row(db_path)
    assert row is not None
    assert row["trace_jsonl_path"], "explicit --trace-out path was not persisted"
    p = Path(row["trace_jsonl_path"])
    assert p == (explicit / "decision_traces.jsonl").resolve()


def test_paper_actual_without_trace_flags_leaves_path_null(
    cmd_trade_with_real_storage, monkeypatch
):
    """Without --trace-out / --trace-out-default the trade row's
    trace_jsonl_path must be NULL — never autocreate a path."""
    cmd, storage, db_path, cwd, cfg = cmd_trade_with_real_storage
    _force_buy(monkeypatch)

    args = _build_trade_namespace(dry_run=False)  # neither flag
    rc = cmd(args, cfg, storage)
    assert rc == 0

    row = _read_trade_row(db_path)
    assert row is not None
    assert row["trace_jsonl_path"] is None
    # Sanity: no runs/ tree produced either
    assert not (cwd / "runs").exists()


def test_hold_path_creates_no_trade_row(
    cmd_trade_with_real_storage,
):
    """HOLD must not insert a trade row regardless of trace flags."""
    cmd, storage, db_path, cwd, cfg = cmd_trade_with_real_storage
    args = _build_trade_namespace(dry_run=False, trace_out_default=True)
    rc = cmd(args, cfg, storage)
    assert rc == 0

    conn = sqlite3.connect(db_path)
    try:
        n = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    finally:
        conn.close()
    assert n == 0, "HOLD must not produce a trade row"


def test_dry_run_path_creates_no_trade_row(
    cmd_trade_with_real_storage, monkeypatch
):
    """Even with a forced BUY, --dry-run must not insert a trade row."""
    cmd, storage, db_path, cwd, cfg = cmd_trade_with_real_storage
    _force_buy(monkeypatch)

    args = _build_trade_namespace(dry_run=True, trace_out_default=True)
    rc = cmd(args, cfg, storage)
    assert rc == 0

    conn = sqlite3.connect(db_path)
    try:
        n = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    finally:
        conn.close()
    assert n == 0, "dry-run must not produce a trade row"


def test_trace_export_failure_does_not_crash_save_trade(
    cmd_trade_with_real_storage, monkeypatch
):
    """When --trace-out points at a directory that already has files and
    --overwrite is not set, the second run still saves the trade row
    (broker already executed) but with trace_jsonl_path NULL, and exit
    code 2 is surfaced via PR #12's trace_export_error path."""
    cmd, storage, db_path, cwd, cfg = cmd_trade_with_real_storage
    _force_buy(monkeypatch)

    explicit = cwd / "shared_dest"
    # First run succeeds
    args1 = _build_trade_namespace(dry_run=False, trace_out=str(explicit))
    assert cmd(args1, cfg, storage) == 0

    # Second run hits FileExistsError (no --overwrite)
    args2 = _build_trade_namespace(dry_run=False, trace_out=str(explicit))
    rc2 = cmd(args2, cfg, storage)
    assert rc2 == 2

    # Two trade rows exist (broker executed both), but the second has
    # trace_jsonl_path NULL because trace export failed.
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT trace_jsonl_path FROM trades ORDER BY id ASC"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 2
    assert rows[0]["trace_jsonl_path"], "first run should have stored path"
    assert rows[1]["trace_jsonl_path"] is None, (
        "second run's trace export failed; path must be NULL"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
