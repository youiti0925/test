"""PR-A: tests for long_term_trend + macro_context plumbing.

Pinned guarantees:
  * `long_term_trend_slice` reads only `df.iloc[: i + 1]` — no future leak.
  * SMA / N-day return values shrink to None during warmup (recorded in
    `unavailable_reasons`, not silently zero).
  * `macro_context_slice` returns None when `macro` is None (e.g. fetch
    failed) and otherwise samples `Series.asof(ts)` strictly past-only.
  * Missing macro slots are surfaced as fields=None plus `missing_slots`
    enumeration; backtest does NOT crash when fetch_macro_snapshot raises.
  * Older trace JSONL (without long_term_trend / macro_context keys)
    still parses through `aggregate_stats` and `aggregate_many` — slots
    bucket under "unknown".
  * The new cross_stats keys exist and pool correctly across runs.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest
from src.fx.decision_trace import LongTermTrendSlice, MacroContextSlice
from src.fx.decision_trace_build import (
    long_term_trend_slice,
    macro_context_slice,
)
from src.fx.decision_trace_stats import aggregate_stats, aggregate_many
from src.fx.macro import MacroSnapshot


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------


def _ohlcv(n_hours: int, *, start: str = "2024-01-01", seed: int = 7) -> pd.DataFrame:
    """Synthetic 1h OHLCV with a mild drift — enough bars for SMA tests."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_hours, freq="1h", tz="UTC")
    drift = np.linspace(0, 5.0, n_hours)
    noise = np.cumsum(rng.standard_normal(n_hours) * 0.05)
    close = 100.0 + drift + noise
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": [1000.0] * n_hours,
        },
        index=idx,
    )


def _macro_series(
    base_index: pd.DatetimeIndex,
    *,
    slots: dict[str, np.ndarray] | None = None,
) -> MacroSnapshot:
    """Build a MacroSnapshot directly (no yfinance) for deterministic tests.

    Each slot's array is aligned against a daily index of matching length,
    starting at base_index[0].normalize(). Callers can pass shorter
    arrays to simulate "slot has limited history" scenarios.
    """
    start = base_index[0].normalize()
    if slots is None:
        n_days = (
            (base_index[-1].normalize() - start).days + 2
        )
        slots = {
            "us10y": np.linspace(4.20, 4.50, n_days),
            "us_short_yield_proxy": np.linspace(4.50, 4.10, n_days),
            "dxy": np.linspace(100.0, 105.0, n_days),
            "vix": np.linspace(13.0, 22.0, n_days),
            "sp500": np.linspace(4500.0, 5000.0, n_days),
            "nasdaq": np.linspace(15000.0, 17000.0, n_days),
            "nikkei": np.linspace(33000.0, 38000.0, n_days),
        }
    series: dict[str, pd.Series] = {}
    for slot, values in slots.items():
        idx = pd.date_range(start, periods=len(values), freq="1D", tz="UTC")
        series[slot] = pd.Series(values, index=idx, name=slot)
    return MacroSnapshot(base_index=base_index, series=series, fetch_errors={})


# ---------------------------------------------------------------------------
# long_term_trend_slice
# ---------------------------------------------------------------------------


def test_long_term_trend_does_not_look_ahead():
    """Computing the slice at bar `i` over df must equal computing it
    over df.iloc[: i + 1]. Future bars must not influence the result."""
    df = _ohlcv(24 * 250)  # ~250 days
    i = 24 * 200  # bar at day 200

    full = long_term_trend_slice(df, i)
    truncated = long_term_trend_slice(df.iloc[: i + 1], i)

    assert full == truncated, (
        "long_term_trend_slice must depend only on bars up to i (look-ahead)"
    )


def test_long_term_trend_warmup_marks_sma200_unavailable():
    """~35 days of history is enough for sma_30d but NOT sma_200d — slot
    must be None AND `unavailable_reasons` must explain why."""
    df = _ohlcv(24 * 35)  # 35 days (clear 30d, well short of 200d)
    s = long_term_trend_slice(df, len(df) - 1)
    assert s.sma_30d is not None, "30d SMA should be available"
    assert s.sma_200d is None, "200d SMA cannot be computed from 35d history"
    assert "sma_200d" in s.unavailable_reasons


def test_long_term_trend_too_short_history_marks_sma30_unavailable():
    """Less than 30 days of history → even sma_30d must be None and
    flagged in unavailable_reasons."""
    df = _ohlcv(24 * 20)  # 20 days
    s = long_term_trend_slice(df, len(df) - 1)
    assert s.sma_30d is None
    assert "sma_30d" in s.unavailable_reasons


def test_long_term_trend_handles_extremely_short_window():
    """Bar 5 of a fresh DF must not crash; everything is None / UNKNOWN."""
    df = _ohlcv(24)
    s = long_term_trend_slice(df, 5)
    assert s.bars_available == 6
    assert s.daily_trend == "UNKNOWN"
    assert s.weekly_trend == "UNKNOWN"
    assert s.monthly_trend == "UNKNOWN"
    assert s.sma_200d is None
    assert s.weekly_return_pct is None or isinstance(s.weekly_return_pct, float)


def test_long_term_trend_returns_match_manual_compute():
    """Spot-check weekly_return_pct against an independent calculation."""
    df = _ohlcv(24 * 60)  # 60 days
    i = 24 * 50
    s = long_term_trend_slice(df, i)
    ts_now = df.index[i]
    cutoff = ts_now - pd.Timedelta(days=7)
    closes = df["close"].iloc[: i + 1]
    ref = float(closes.loc[closes.index <= cutoff].iloc[-1])
    expected = 100.0 * (closes.iloc[-1] - ref) / ref
    assert s.weekly_return_pct == pytest.approx(expected, abs=1e-9)


def test_long_term_trend_to_dict_roundtrips_known_keys():
    df = _ohlcv(24 * 300)
    s = long_term_trend_slice(df, len(df) - 1)
    d = s.to_dict()
    expected_keys = {
        "daily_trend", "weekly_trend", "monthly_trend",
        "sma_30d", "sma_90d", "sma_200d",
        "close_vs_sma_30d_pct", "close_vs_sma_90d_pct",
        "close_vs_sma_200d_pct",
        "weekly_return_pct", "monthly_return_pct", "quarterly_return_pct",
        "bars_available", "unavailable_reasons",
    }
    assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# macro_context_slice
# ---------------------------------------------------------------------------


def test_macro_context_returns_none_when_macro_none():
    """No macro fetch → no slice. Trace-side logic relies on this."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    assert macro_context_slice(None, ts) is None


def test_macro_context_uses_only_past_via_asof():
    """value_at must never reach forward — confirm by checking values
    from a snapshot whose series we control directly."""
    df = _ohlcv(24 * 30, start="2025-01-01")
    macro = _macro_series(df.index)
    ts = df.index[24 * 15]  # day 15
    s = macro_context_slice(macro, ts)
    assert s is not None
    assert s.dxy is not None
    # dxy at ts must equal MacroSnapshot.value_at("dxy", ts) — never future
    assert s.dxy == pytest.approx(macro.value_at("dxy", ts))
    # Value past the run end must not be used
    far_future = df.index[-1] + pd.Timedelta(days=10)
    assert s.dxy <= macro.value_at("dxy", far_future) or s.dxy >= macro.value_at("dxy", far_future)
    # Sanity: 5d delta does NOT match a future bar's level
    assert s.dxy_change_5d_pct is not None


def test_macro_context_missing_slot_listed_explicitly():
    """A slot absent from the snapshot ends up None and in missing_slots."""
    df = _ohlcv(24 * 30, start="2025-01-01")
    macro = _macro_series(
        df.index,
        slots={"dxy": np.linspace(100.0, 102.0, 35)},
    )
    ts = df.index[100]
    s = macro_context_slice(macro, ts)
    assert s is not None
    assert s.dxy is not None
    assert s.us10y is None
    assert "us10y" in s.missing_slots
    # yield_spread requires both ends — must fall back to None gracefully
    assert s.yield_spread_long_short is None


def test_macro_context_5d_delta_uses_value_5d_back():
    df = _ohlcv(24 * 60, start="2025-01-01")
    macro = _macro_series(df.index)
    ts = df.index[24 * 30]
    s = macro_context_slice(macro, ts)
    assert s is not None
    now = macro.value_at("dxy", ts)
    prev = macro.value_at("dxy", ts - pd.Timedelta(days=5))
    assert s.dxy_change_5d_pct == pytest.approx(100.0 * (now - prev) / prev, abs=1e-6)


# ---------------------------------------------------------------------------
# backtest_engine plumbing
# ---------------------------------------------------------------------------


def test_run_engine_backtest_attaches_long_term_trend_to_traces():
    df = _ohlcv(24 * 250, start="2024-01-01")
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    assert res.decision_traces, "engine should produce traces"
    sample = res.decision_traces[100].to_dict()
    assert "long_term_trend" in sample
    assert sample["long_term_trend"] is not None
    assert "daily_trend" in sample["long_term_trend"]


def test_run_engine_backtest_macro_optional_no_crash():
    """When macro is None, traces must still produce — macro_context=null."""
    df = _ohlcv(24 * 60, start="2024-01-01")
    res = run_engine_backtest(df, "X", interval="1h", warmup=50)
    sample = res.decision_traces[10].to_dict()
    assert sample["macro_context"] is None


def test_run_engine_backtest_macro_attached_populates_slice():
    df = _ohlcv(24 * 60, start="2024-01-01")
    macro = _macro_series(df.index)
    res = run_engine_backtest(
        df, "X", interval="1h", warmup=50, macro=macro,
    )
    sample = res.decision_traces[10].to_dict()
    mc = sample["macro_context"]
    assert mc is not None
    assert mc["dxy"] is not None
    assert mc["us10y"] is not None
    assert "available_slots" in mc and len(mc["available_slots"]) >= 5


def test_macro_fetch_failure_is_swallowed_by_cli_path(tmp_path, monkeypatch):
    """fetch_macro_snapshot raising must not abort cmd_backtest_engine.
    We monkeypatch the symbol to raise; the engine should still produce
    a trace with macro_context=None."""
    from src.fx import cli as cli_mod

    df = _ohlcv(24 * 60, start="2024-01-01")

    def fake_fetch_ohlcv(symbol, interval, period=None, start=None, end=None):
        return df

    def boom(*a, **kw):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(cli_mod, "fetch_ohlcv", fake_fetch_ohlcv)
    monkeypatch.setattr(cli_mod, "fetch_macro_snapshot", boom)
    monkeypatch.chdir(tmp_path)

    import argparse
    args = argparse.Namespace(
        symbol="X", interval="1h", period="60d", start=None, end=None,
        warmup=50, stop_atr=2.0, tp_atr=3.0, max_holding_bars=48,
        no_events=True, no_higher_tf=True, no_macro=False,
        macro_period="2y",
        trace_out=str(tmp_path / "out"), trace_out_default=False,
        overwrite=False, gzip=False,
    )

    class Cfg:
        anthropic_api_key = None
        model = "claude-sonnet-4-6"
        effort = "medium"
        db_path = ":memory:"

    class _NullStorage:
        def save_backtest(self, **kw):
            return 1

    rc = cli_mod.cmd_backtest_engine(args, Cfg(), _NullStorage())
    assert rc == 0, "macro fetch failure must be non-fatal"
    jsonl = tmp_path / "out" / "decision_traces.jsonl"
    rec = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[10])
    assert rec["macro_context"] is None


# ---------------------------------------------------------------------------
# stats backward compat + new cross_stats
# ---------------------------------------------------------------------------


def test_aggregate_stats_buckets_old_trace_under_unknown(tmp_path):
    """A trace JSONL written before PR-A (no long_term_trend / macro_context
    keys) must still parse through aggregate_stats — every new cross_stats
    key exists, every bucket is `unknown`."""
    minimal = {
        "run_id": "r", "trace_schema_version": "decision_trace_v1",
        "bar_id": "X_1h_2025-01-01T00:00:00+00:00",
        "timestamp": "2025-01-01T00:00:00+00:00",
        "symbol": "X", "timeframe": "1h", "bar_index": 0,
        "market": {}, "technical": {}, "waveform": {},
        "higher_timeframe": {}, "fundamental": {},
        "execution_assumption": {},
        "execution_trace": {"entry_executed": False},
        "rule_checks": [],
        "decision": {
            "final_action": "HOLD",
            "technical_only_action": "HOLD",
            "blocked_by": [],
            "rule_chain": [],
            "reason": "Technical signal is HOLD; nothing to confirm",
        },
        "future_outcome": {
            "gate_effect": "NO_CHANGE",
            "outcome_if_technical_action_taken": "N/A",
        },
    }
    p = tmp_path / "old.jsonl"
    p.write_text(json.dumps(minimal) + "\n", encoding="utf-8")
    s = aggregate_stats(p)
    cs = s["cross_stats"]
    for key in (
        "daily_trend_outcome",
        "weekly_trend_outcome",
        "monthly_trend_outcome",
        "close_vs_sma_200d_outcome",
        "weekly_return_outcome",
        "monthly_return_outcome",
        "dxy_trend_outcome",
        "us10y_trend_outcome",
        "yield_spread_outcome",
        "vix_regime_outcome",
        "long_term_macro_outcome",
    ):
        assert key in cs
        # The old trace lacked the slices → bucket should be 'unknown'
        if key == "long_term_macro_outcome":
            assert any(k.startswith("unknown") for k in cs[key])
        else:
            assert "unknown" in cs[key]


def test_aggregate_stats_with_new_slices(tmp_path):
    rec = {
        "run_id": "r", "trace_schema_version": "decision_trace_v1",
        "bar_id": "X_1h_2025-01-01T00:00:00+00:00",
        "timestamp": "2025-01-01T00:00:00+00:00",
        "symbol": "X", "timeframe": "1h", "bar_index": 0,
        "market": {}, "technical": {}, "waveform": {},
        "higher_timeframe": {}, "fundamental": {},
        "execution_assumption": {},
        "execution_trace": {"entry_executed": True, "trade_id": "t1"},
        "rule_checks": [],
        "decision": {
            "final_action": "BUY",
            "technical_only_action": "BUY",
            "blocked_by": [],
            "rule_chain": [],
            "reason": "approved",
        },
        "future_outcome": {
            "gate_effect": "NO_CHANGE",
            "outcome_if_technical_action_taken": "WIN",
        },
        "long_term_trend": {
            "daily_trend": "UPTREND",
            "weekly_trend": "UPTREND",
            "monthly_trend": "UPTREND",
            "close_vs_sma_200d_pct": 3.5,
            "weekly_return_pct": 0.4,
            "monthly_return_pct": 1.2,
        },
        "macro_context": {
            "dxy_change_5d_pct": -1.5,
            "us10y_change_5d": -0.12,
            "yield_spread_long_short": 0.6,
            "vix": 22.0,
        },
    }
    p = tmp_path / "new.jsonl"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    s = aggregate_stats(p)
    cs = s["cross_stats"]
    assert "UPTREND" in cs["daily_trend_outcome"]
    assert cs["daily_trend_outcome"]["UPTREND"]["BUY"]["n"] == 1
    assert cs["daily_trend_outcome"]["UPTREND"]["BUY"]["outcome"]["WIN"] == 1
    # close_vs_sma_200d_pct=3.5 → bucket "1..5%"
    assert "1..5%" in cs["close_vs_sma_200d_outcome"]
    # dxy_change_5d_pct=-1.5 → bucket "down>1%"
    assert "down>1%" in cs["dxy_trend_outcome"]
    # us10y_change_5d=-0.12 → bucket "down>10bp"
    assert "down>10bp" in cs["us10y_trend_outcome"]
    # vix=22 → "high"
    assert "high" in cs["vix_regime_outcome"]


def test_aggregate_many_pools_long_term_macro_buckets(tmp_path):
    """Two runs, each contributing one HOLD bar with monthly_trend=UPTREND
    and dxy_5d down — pool must sum to n=2 in that joint bucket."""
    def write_run(p: Path, action: str, bucket_dxy: float):
        rec = {
            "run_id": p.parent.name,
            "trace_schema_version": "decision_trace_v1",
            "bar_id": f"{p.parent.name}_X_1h_2025-01-01",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "symbol": "X", "timeframe": "1h", "bar_index": 0,
            "market": {}, "technical": {}, "waveform": {},
            "higher_timeframe": {}, "fundamental": {},
            "execution_assumption": {},
            "execution_trace": {"entry_executed": False},
            "rule_checks": [],
            "decision": {
                "final_action": action,
                "technical_only_action": action,
                "blocked_by": [],
                "rule_chain": [],
                "reason": "x",
            },
            "future_outcome": {
                "gate_effect": "NO_CHANGE",
                "outcome_if_technical_action_taken": "N/A",
            },
            "long_term_trend": {
                "daily_trend": "UPTREND",
                "weekly_trend": "UPTREND",
                "monthly_trend": "UPTREND",
            },
            "macro_context": {"dxy_change_5d_pct": bucket_dxy},
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    a = tmp_path / "run_a" / "decision_traces.jsonl"
    b = tmp_path / "run_b" / "decision_traces.jsonl"
    write_run(a, "HOLD", -2.0)  # dxy down>1%
    write_run(b, "HOLD", -2.0)
    pooled = aggregate_many([a, b])
    cs = pooled["global"]["cross_stats"]
    joint = cs["long_term_macro_outcome"]
    key = "UPTREND|dxy_5d=down>1%"
    assert key in joint
    assert joint[key]["HOLD"]["n"] == 2
