"""PR #17 tests: --context-days observation-only context for backtest-engine.

Covers T1–T8 from the design:
  T1 no_trade_in_context_period
  T2 metrics_computed_from_test_period_only
  T3 trace_jsonl_only_test_bars
  T4 test_decisions_byte_identical_with_or_without_context  ← core invariance
  T5 long_term_trend_populated_from_test_start_with_400d_context
  T6 long_term_trend_unavailable_without_context  (compare-baseline for T5)
  T7 no_lookahead_in_long_term_trend_at_test_bar
  T8 run_metadata_context_section_emitted

Plus validation cases (overlap / duplicates / tz mismatch) and the
CLI-level decisions-invariance (engine call) check.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import (
    _validate_and_concat_context,
    run_engine_backtest,
)
from src.fx.decision_trace_build import long_term_trend_slice
from src.fx.decision_trace_io import export_run


def _ohlcv(n: int, *, start: str = "2025-01-01", seed: int = 1,
           freq: str = "1h") -> pd.DataFrame:
    """Synthetic OHLCV. Random walk on close, ±0.3 high/low band."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close, "high": close + 0.3, "low": close - 0.3,
            "close": close, "volume": [1000] * n,
        },
        index=idx,
    )


# ─── T1 / T2 / T3 ───────────────────────────────────────────────────────────


def test_no_trade_in_context_period():
    """T1: trades / equity_curve / decision_traces all confined to df_test."""
    df_context = _ohlcv(24 * 30, start="2024-12-01", seed=1)
    df_test = _ohlcv(200, start="2024-12-31", seed=2)
    res = run_engine_backtest(
        df_test, "X", interval="1h", warmup=60,
        df_context=df_context, context_days=30,
    )
    test_min = df_test.index.min()
    for trade in res.trades:
        assert trade.entry_ts >= test_min, (
            f"trade.entry_ts {trade.entry_ts} fell into context period "
            f"(test_start={test_min})"
        )
    for ts, _ in res.equity_curve:
        assert ts >= test_min
    for trace in res.decision_traces:
        assert pd.Timestamp(trace.timestamp) >= test_min


def test_metrics_computed_from_test_period_only():
    """T2: bars_processed and metrics() count only test bars."""
    df_context = _ohlcv(24 * 30, start="2024-12-01", seed=1)
    df_test = _ohlcv(200, start="2024-12-31", seed=2)
    res = run_engine_backtest(
        df_test, "X", interval="1h", warmup=60,
        df_context=df_context, context_days=30,
    )
    expected = len(df_test) - 60  # warmup bars excluded
    assert res.bars_processed == expected
    assert res.metrics()["bars_processed"] == expected


def test_trace_jsonl_only_test_bars(tmp_path):
    """T3: exported decision_traces.jsonl carries no context-period rows."""
    df_context = _ohlcv(24 * 30, start="2024-12-01", seed=1)
    df_test = _ohlcv(200, start="2024-12-31", seed=2)
    res = run_engine_backtest(
        df_test, "X", interval="1h", warmup=60,
        df_context=df_context, context_days=30,
    )
    paths = export_run(res, out_dir=tmp_path / "out")
    test_min = df_test.index.min()
    n = 0
    with open(paths["decision_traces"]) as f:
        for line in f:
            r = json.loads(line)
            ts = pd.Timestamp(r["timestamp"])
            assert ts >= test_min, (
                f"trace ts {ts} predates test_start {test_min}"
            )
            n += 1
    assert n > 0
    assert n == len(df_test) - 60


# ─── T4: byte-identical decisions ───────────────────────────────────────────


def _trade_summary(res):
    return [
        (t.entry_ts, t.exit_ts, t.side, t.exit_reason,
         round(t.pnl, 8), round(t.return_pct, 8), t.bars_held)
        for t in res.trades
    ]


def test_decisions_byte_identical_with_or_without_context():
    """T4 (CORE): the same df_test must produce the same trade list whether
    df_context is attached or not. Pins the Option-α invariant: context is
    observation-only. Run the engine twice, compare every field of every
    trade plus hold_reasons."""
    df_test = _ohlcv(400, start="2025-01-15", seed=3)
    df_context = _ohlcv(24 * 60, start="2024-11-15", seed=1)

    res_no = run_engine_backtest(
        df_test, "X", interval="1h", warmup=60,
    )
    res_yes = run_engine_backtest(
        df_test, "X", interval="1h", warmup=60,
        df_context=df_context, context_days=60,
    )
    assert _trade_summary(res_no) == _trade_summary(res_yes), (
        "Trade list diverged when df_context was attached — Option α "
        "invariant violated."
    )
    assert res_no.hold_reasons == res_yes.hold_reasons
    assert res_no.metrics()["n_trades"] == res_yes.metrics()["n_trades"]
    assert res_no.metrics()["win_rate"] == res_yes.metrics()["win_rate"]
    assert (
        res_no.metrics()["total_return_pct"]
        == res_yes.metrics()["total_return_pct"]
    )
    # Equity curve must be identical bar-by-bar.
    assert len(res_no.equity_curve) == len(res_yes.equity_curve)
    for (t1, e1), (t2, e2) in zip(
        res_no.equity_curve, res_yes.equity_curve,
    ):
        assert t1 == t2 and e1 == e2


# ─── T5 / T6: long_term_trend visibility with vs without context ────────────


def test_long_term_trend_populated_from_test_start_with_400d_context():
    """T5: with ~400d context, the FIRST test trace's long_term_trend
    has bars_available much higher than what test alone could supply,
    so daily_trend is no longer UNKNOWN at test_start."""
    df_test = _ohlcv(200, start="2026-01-01", seed=2)
    df_context = _ohlcv(24 * 365, start="2025-01-01", seed=1)
    res = run_engine_backtest(
        df_test, "X", interval="1h", warmup=60,
        df_context=df_context, context_days=365,
    )
    first_trace = res.decision_traces[0]
    ltt = first_trace.long_term_trend.to_dict()
    # Way more daily history than the 200 test bars alone could yield.
    assert ltt["bars_available"] > 200
    # daily_trend should resolve to one of the populated states; the
    # exact value depends on the random walk fixture, so accept any
    # non-UNKNOWN value (which is what we'd get without context).
    assert ltt["daily_trend"] in {"RANGE", "UPTREND", "DOWNTREND",
                                  "VOLATILE"}


def test_long_term_trend_unavailable_without_context():
    """T6: same test df, but no context — first trace shows the legacy
    starvation pattern (small bars_available, daily_trend=UNKNOWN). This
    asserts the symmetry that T5 is observing a *real* effect of context."""
    df_test = _ohlcv(200, start="2026-01-01", seed=2)
    res = run_engine_backtest(df_test, "X", interval="1h", warmup=60)
    first_trace = res.decision_traces[0]
    ltt = first_trace.long_term_trend.to_dict()
    assert ltt["bars_available"] <= 100
    # Without context, sma_200d is structurally unreachable on a 200-bar
    # test frame regardless of the random seed.
    assert ltt["sma_200d"] is None


# ─── T7: no future leak in the long-term trend ───────────────────────────────


def test_no_lookahead_in_long_term_trend_at_test_bar():
    """T7: re-derive long_term_trend at bar i from `df_context + df_test[:i+1]`
    and check that the engine emitted exactly the same slice. Anything later
    than bar i in df_test must NOT have leaked into the trend computation."""
    df_test = _ohlcv(150, start="2026-01-01", seed=2)
    df_context = _ohlcv(24 * 200, start="2025-06-01", seed=1)
    res = run_engine_backtest(
        df_test, "X", interval="1h", warmup=60,
        df_context=df_context, context_days=200,
    )
    # Pick a few bars across the test period; each must match the
    # independently-recomputed point-in-time slice.
    for offset in (0, 5, 30, 60, len(res.decision_traces) - 1):
        trace = res.decision_traces[offset]
        i_test = trace.bar_index
        df_full_until = pd.concat([
            df_context.sort_index(),
            df_test.iloc[: i_test + 1],
        ])
        expected = long_term_trend_slice(
            df_full_until, len(df_full_until) - 1,
        )
        assert trace.long_term_trend.to_dict() == expected.to_dict()


# ─── T8: run_metadata.context block ─────────────────────────────────────────


def test_run_metadata_context_section_emitted_with_context():
    df_test = _ohlcv(200, start="2026-01-01", seed=2)
    df_context = _ohlcv(24 * 60, start="2025-11-01", seed=1)
    res = run_engine_backtest(
        df_test, "X", interval="1h", warmup=60,
        df_context=df_context, context_days=60,
    )
    rm = res.run_metadata.to_dict()
    assert "context" in rm and rm["context"] is not None
    ctx = rm["context"]
    expected_keys = {
        "context_enabled", "context_days", "context_start",
        "test_start", "test_end", "n_context_bars", "n_test_bars",
        "metrics_scope", "trace_scope", "long_term_context_attached",
    }
    assert set(ctx.keys()) == expected_keys
    assert ctx["context_enabled"] is True
    assert ctx["context_days"] == 60
    assert ctx["n_context_bars"] == len(df_context)
    assert ctx["n_test_bars"] == len(df_test)
    assert ctx["metrics_scope"] == "test_only"
    assert ctx["trace_scope"] == "test_only"
    assert ctx["long_term_context_attached"] is True
    assert ctx["context_start"] == df_context.index.min().isoformat()
    assert ctx["test_start"] == df_test.index.min().isoformat()
    assert ctx["test_end"] == df_test.index.max().isoformat()


def test_run_metadata_context_section_emitted_without_context():
    df_test = _ohlcv(200, start="2026-01-01", seed=2)
    res = run_engine_backtest(df_test, "X", interval="1h", warmup=60)
    ctx = res.run_metadata.to_dict()["context"]
    assert ctx["context_enabled"] is False
    assert ctx["context_days"] == 0
    assert ctx["n_context_bars"] == 0
    assert ctx["context_start"] is None
    assert ctx["long_term_context_attached"] is False
    # test_start / test_end / n_test_bars still populated.
    assert ctx["test_start"] == df_test.index.min().isoformat()
    assert ctx["n_test_bars"] == len(df_test)
    # metrics/trace scope are constants, regardless of context state.
    assert ctx["metrics_scope"] == "test_only"
    assert ctx["trace_scope"] == "test_only"


# ─── Validation: overlap / duplicate / tz-mismatch must fail loud ───────────


def test_context_overlap_with_test_raises():
    df_test = _ohlcv(50, start="2025-02-01", seed=2)
    df_overlap = _ohlcv(60, start="2025-01-30", seed=1)  # last 10 overlap
    with pytest.raises(ValueError, match="overlaps"):
        _validate_and_concat_context(df_overlap, df_test)


def test_context_duplicate_index_raises():
    df_test = _ohlcv(50, start="2025-02-01", seed=2)
    base = _ohlcv(50, start="2025-01-01", seed=1)
    df_dup = pd.concat([base, base.iloc[:5]])  # 5 dupes
    with pytest.raises(ValueError, match="duplicate"):
        _validate_and_concat_context(df_dup, df_test)


def test_context_tz_mismatch_raises():
    df_test = _ohlcv(50, start="2025-02-01", seed=2)
    naive = _ohlcv(50, start="2025-01-01", seed=1)
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone"):
        _validate_and_concat_context(naive, df_test)


def test_engine_rejects_overlapping_context():
    """The engine itself must surface the overlap, not silently drop bars."""
    df_test = _ohlcv(50, start="2025-02-01", seed=2)
    df_overlap = _ohlcv(60, start="2025-01-30", seed=1)
    with pytest.raises(ValueError):
        run_engine_backtest(
            df_test, "X", interval="1h", warmup=20,
            df_context=df_overlap, context_days=2,
        )


# ─── Default-zero compatibility: existing tests untouched by --context-days ─


def test_context_days_zero_keeps_legacy_behaviour():
    """--context-days 0 (default) must produce the same result as never
    passing the kwarg. Pins backwards compatibility for every pre-PR-#17
    call site (cmd_trade live, existing tests, downstream notebooks)."""
    df_test = _ohlcv(200, start="2026-01-01", seed=2)
    res_default = run_engine_backtest(
        df_test, "X", interval="1h", warmup=60,
    )
    res_zero = run_engine_backtest(
        df_test, "X", interval="1h", warmup=60,
        df_context=None, context_days=0,
    )
    assert _trade_summary(res_default) == _trade_summary(res_zero)
    assert res_default.hold_reasons == res_zero.hold_reasons
    # context block populated either way; the kwargless path goes through
    # the same code, and context_enabled stays False.
    assert (
        res_default.run_metadata.to_dict()["context"]["context_enabled"]
        is False
    )
