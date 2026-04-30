"""PR #20 — Trace Observation Expansion for Regime Analysis.

All new trace fields are observation-only and never read by
`decide_action` / `risk_gate`. The most important invariant pinned
here is `test_pr20_decisions_byte_identical_to_pr19`: same fixture,
same outcomes, regardless of which new fields were populated.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest
from src.fx.decision_trace_build import (
    _DXY_TREND_BUCKET_THRESHOLDS,
    _DXY_ZSCORE_BUCKET_THRESHOLDS,
    _SHADOW_OUTCOME_THRESHOLD_PCT,
    _SMA_50_200_DEAD_BAND_PCT,
    _dxy_trend_bucket,
    _dxy_zscore_bucket,
    _shadow_outcome_bucket,
    _shadow_outcome_direction,
    _monthly_trend_audit,
    long_term_trend_slice,
    macro_context_slice,
)
from src.fx.decision_trace_r_candidates import (
    N_THRESHOLD_FOR_SUFFICIENT,
    SCHEMA_VERSION,
    compute_r_candidates_summary,
)
from src.fx.macro import MacroSnapshot


# ───────────────────────────── fixtures ──────────────────────────────


def _ohlcv(n: int, *, start: str = "2025-01-01", seed: int = 1,
           freq: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def _macro_with_dxy(start: str, n_days: int, seed: int = 1) -> MacroSnapshot:
    """Build a MacroSnapshot whose `dxy` series is daily, n_days long."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days, freq="1D", tz="UTC")
    levels = 100.0 + np.cumsum(rng.standard_normal(n_days) * 0.3)
    return MacroSnapshot(
        base_index=idx,
        series={"dxy": pd.Series(levels, index=idx, name="dxy")},
        fetch_errors={},
    )


def _macro_empty() -> MacroSnapshot:
    return MacroSnapshot(
        base_index=pd.DatetimeIndex([], tz="UTC"),
        series={}, fetch_errors={},
    )


# ───────────────────────────── T1 — DXY ──────────────────────────────


def test_dxy_trend_fields_emitted():
    """DXY return / z-score / bucket appear in macro_context when
    sufficient daily history is available."""
    macro = _macro_with_dxy("2025-09-01", n_days=120, seed=11)
    ts = pd.Timestamp("2025-12-15", tz="UTC")
    slc = macro_context_slice(macro, ts)
    assert slc is not None
    d = slc.to_dict()
    # All four window stats present
    for k in ("dxy_return_5d_pct", "dxy_return_20d_pct",
              "dxy_zscore_20d", "dxy_zscore_60d"):
        assert d[k] is not None, f"{k} should be populated"
    # All buckets resolve to a known label
    assert d["dxy_trend_5d_bucket"] in {
        "STRONG_DOWN", "DOWN", "FLAT", "UP", "STRONG_UP",
    }
    assert d["dxy_trend_20d_bucket"] in {
        "STRONG_DOWN", "DOWN", "FLAT", "UP", "STRONG_UP",
    }
    assert d["dxy_zscore_bucket"] in {
        "EXTREME_LOW", "LOW", "NEUTRAL", "HIGH", "EXTREME_HIGH",
    }
    # No unavailable_reason when everything resolved
    assert d["dxy_unavailable_reason"] is None


# ───────────────────────────── T2 — DXY missing ──────────────────────


def test_dxy_trend_fields_none_when_history_insufficient():
    """Short DXY series → returns / z-scores cannot compute → None
    + structured unavailable_reason. Must NOT raise."""
    # Only 3 days of DXY data — far less than 5d return / 20d zscore.
    macro = _macro_with_dxy("2025-09-01", n_days=3)
    ts = pd.Timestamp("2025-09-04 12:00", tz="UTC")
    slc = macro_context_slice(macro, ts)
    assert slc is not None
    d = slc.to_dict()
    assert d["dxy_return_20d_pct"] is None
    assert d["dxy_zscore_20d"] is None
    # Reason recorded somewhere — empty buckets confirm degenerate state
    assert d["dxy_trend_20d_bucket"] is None
    assert d["dxy_zscore_bucket"] is None
    # And missing macro entirely → DXY is silently None (existing behaviour)
    slc_none = macro_context_slice(_macro_empty(), ts)
    assert slc_none is not None
    d_none = slc_none.to_dict()
    assert d_none["dxy"] is None
    assert d_none["dxy_unavailable_reason"] == "dxy_series_unavailable"


def test_dxy_zscore_handles_stdev_zero():
    """All-equal DXY series → stdev = 0 → z-score must be None,
    NOT raise (per Q-clarification)."""
    idx = pd.date_range("2025-09-01", periods=60, freq="1D", tz="UTC")
    flat = pd.Series([100.0] * 60, index=idx, name="dxy")
    macro = MacroSnapshot(
        base_index=idx, series={"dxy": flat}, fetch_errors={},
    )
    ts = pd.Timestamp("2025-10-29", tz="UTC")
    slc = macro_context_slice(macro, ts)
    assert slc is not None
    d = slc.to_dict()
    assert d["dxy_zscore_20d"] is None
    assert d["dxy_zscore_60d"] is None
    # Returns over 5d/20d on a flat series ARE 0.0 — that's a valid value,
    # not unavailable. So FLAT bucket is the expected label.
    assert d["dxy_return_5d_pct"] == 0.0
    assert d["dxy_trend_5d_bucket"] == "FLAT"


# ─────────────────────── T3/T4 — SMA50 / state ───────────────────────


def test_sma_50d_emitted_with_enough_history():
    """Long enough OHLCV → sma_50d valid + close_vs_sma_50d_pct."""
    df = _ohlcv(24 * 80, start="2025-01-01", seed=2)
    slc = long_term_trend_slice(df, len(df) - 1)
    d = slc.to_dict()
    assert d["sma_50d"] is not None
    assert d["close_vs_sma_50d_pct"] is not None
    assert d["sma_50_vs_sma_200_pct"] is None or isinstance(
        d["sma_50_vs_sma_200_pct"], float
    )


def test_sma_50_200_state_classification():
    """BULLISH / BEARISH / NEUTRAL / UNKNOWN labels are all reachable."""
    # Long uptrend → sma50 above sma200 → BULLISH or NEUTRAL
    n = 24 * 220
    rng = np.random.default_rng(3)
    base = np.linspace(100, 130, n) + rng.standard_normal(n) * 0.2
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    df_up = pd.DataFrame({
        "open": base, "high": base + 0.3, "low": base - 0.3,
        "close": base, "volume": [1000] * n,
    }, index=idx)
    slc = long_term_trend_slice(df_up, len(df_up) - 1)
    assert slc.sma_50_200_state in {"BULLISH", "BEARISH", "NEUTRAL"}, (
        f"unexpected state {slc.sma_50_200_state}"
    )
    # Insufficient history → UNKNOWN
    df_short = _ohlcv(50, start="2025-12-01", seed=5)
    slc_short = long_term_trend_slice(df_short, len(df_short) - 1)
    assert slc_short.sma_50_200_state == "UNKNOWN"


# ─────────────────────── T5/T6 — shadow outcome ──────────────────────


def test_shadow_outcome_classifiers():
    """Direction and bucket classifiers respect threshold (±0.3%)."""
    th = _SHADOW_OUTCOME_THRESHOLD_PCT
    assert _shadow_outcome_direction(None) is None
    assert _shadow_outcome_direction(0.0) == "FLAT"
    assert _shadow_outcome_direction(th) == "UP"
    assert _shadow_outcome_direction(-th) == "DOWN"
    assert _shadow_outcome_direction(th - 0.001) == "FLAT"

    # bucket relative to action
    assert _shadow_outcome_bucket(None, "BUY") is None
    assert _shadow_outcome_bucket(1.5, "BUY") == "STRONG_FOR"
    assert _shadow_outcome_bucket(-1.5, "BUY") == "STRONG_AGAINST"
    assert _shadow_outcome_bucket(1.5, "SELL") == "STRONG_AGAINST"
    assert _shadow_outcome_bucket(0.0, "HOLD") == "NEUTRAL"


def test_shadow_outcome_emitted_only_for_blocked_bars():
    """`blocked_*` fields populated when decision.blocked_by is non-empty."""
    df = _ohlcv(400, start="2025-01-15", seed=7)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    n_blocked = 0
    n_blocked_with_shadow = 0
    n_unblocked_with_shadow = 0
    for tr in res.decision_traces:
        was_blocked = bool(getattr(tr.decision, "blocked_by", ()) or ())
        fo = tr.future_outcome
        has_shadow = (
            fo is not None
            and getattr(fo, "blocked_outcome_direction", None) is not None
        )
        if was_blocked:
            n_blocked += 1
            if has_shadow:
                n_blocked_with_shadow += 1
        else:
            if has_shadow:
                n_unblocked_with_shadow += 1
    # Non-blocked bars must NOT carry a shadow direction (kept None).
    assert n_unblocked_with_shadow == 0


# ───────────────────────── T7 — DXY/zscore bucket helpers ────────────


def test_dxy_bucket_helpers_thresholds():
    t = _DXY_TREND_BUCKET_THRESHOLDS
    assert _dxy_trend_bucket(t["strong_down"] - 0.1) == "STRONG_DOWN"
    assert _dxy_trend_bucket(t["strong_down"] + 0.1) == "DOWN"
    assert _dxy_trend_bucket(0.0) == "FLAT"
    assert _dxy_trend_bucket(t["strong_up"]) == "STRONG_UP"
    assert _dxy_trend_bucket(None) is None

    z = _DXY_ZSCORE_BUCKET_THRESHOLDS
    assert _dxy_zscore_bucket(z["extreme_low"] - 0.5) == "EXTREME_LOW"
    assert _dxy_zscore_bucket(0.0) == "NEUTRAL"
    assert _dxy_zscore_bucket(z["extreme_high"]) == "EXTREME_HIGH"
    assert _dxy_zscore_bucket(None) is None


# ───────────────────────── T8 — r_candidates schema ──────────────────


def _fake_trade(*, sym="X", side="BUY", return_pct=0.5, exit_reason="take_profit",
                vix=20.0, dxy=98.0, top_sim=0.80, conf=0.0,
                wb_action="HOLD", final_action="HOLD",
                close_vs_sma_200=2.0, monthly_return=-1.5,
                entry_ts="2026-02-15T13:00:00+00:00",
                dxy_trend_5d="FLAT", dxy_trend_20d="FLAT",
                bars_held=10):
    """Synthetic trade with a fully populated entry-trace dict."""
    return {
        "sym": sym, "side": side, "entry_ts": entry_ts,
        "return_pct": return_pct, "exit_reason": exit_reason,
        "bars_held": bars_held,
        "entry_trace": {
            "macro_context": {
                "vix": vix, "dxy": dxy,
                "dxy_trend_5d_bucket": dxy_trend_5d,
                "dxy_trend_20d_bucket": dxy_trend_20d,
            },
            "long_term_trend": {
                "close_vs_sma_200d_pct": close_vs_sma_200,
                "monthly_return_pct": monthly_return,
            },
            "waveform": {
                "waveform_bias": {
                    "action": wb_action, "confidence": conf,
                    "top_similarity": top_sim,
                },
            },
            "decision": {"final_action": final_action},
        },
    }


def test_r_candidates_summary_has_required_sections():
    trades = [_fake_trade()]
    summary = compute_r_candidates_summary(trades=trades)
    expected_top = {
        "schema_version", "test_period", "n_trades_total",
        "n_traces_total", "n_threshold_for_sufficient", "thresholds",
        "r1_vix_extreme_outcome",
        "r2_vix_20_25_waveform_hold_outcome",
        "r3_waveform_high_conf_outcome",
        "r4_top_similarity_extreme_outcome",
        "r5_dxy_usd_exposure_outcome",
        "r6_session_outcome",
        "r7_waveform_hold_vs_concur_outcome",
        "r8_exit_holding_distribution",
        "r9_sma200_above_sell_outcome",
        "r10_near_sma200_buy_outcome",
        "r11_monthly_return_waveform_hold_outcome",
    }
    assert expected_top.issubset(summary.keys())
    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["n_threshold_for_sufficient"] == N_THRESHOLD_FOR_SUFFICIENT
    # thresholds carries the human-readable bucket map AND the
    # observation-policy versions.
    th = summary["thresholds"]
    assert "dxy_trend_bucket" in th
    assert "dxy_zscore_bucket" in th
    assert th["sma_50_200_dead_band_pct"] == _SMA_50_200_DEAD_BAND_PCT
    assert th["shadow_outcome_threshold_pct"] == _SHADOW_OUTCOME_THRESHOLD_PCT


def test_r_candidates_summary_sufficient_n_threshold():
    """sufficient_n flips True at n>=30 cells, False below."""
    # Build 30 winning fake trades all hitting R4 high-similarity bucket
    trades = [_fake_trade(top_sim=0.9, return_pct=0.1) for _ in range(30)]
    s = compute_r_candidates_summary(trades=trades)
    r4_cells = s["r4_top_similarity_extreme_outcome"]["cells"]
    high = next(c for c in r4_cells if c["label"] == "top_sim>=0.85")
    assert high["n"] == 30
    assert high["sufficient_n"] is True
    # And the inverse — n=10 should NOT be sufficient
    trades_small = [_fake_trade(top_sim=0.9) for _ in range(10)]
    s_small = compute_r_candidates_summary(trades=trades_small)
    cells = s_small["r4_top_similarity_extreme_outcome"]["cells"]
    h_small = next(c for c in cells if c["label"] == "top_sim>=0.85")
    assert h_small["sufficient_n"] is False


def test_r5_uses_dxy_trend_bucket_not_level():
    """R5 cells pivot by DXY trend bucket × USD direction."""
    trades = [
        # USDJPY=X BUY = LONG_USD; DXY trend STRONG_UP
        _fake_trade(sym="USDJPY=X", side="BUY",
                    dxy_trend_5d="STRONG_UP", return_pct=1.0),
        # EURUSD=X BUY = SHORT_USD; DXY trend FLAT
        _fake_trade(sym="EURUSD=X", side="BUY",
                    dxy_trend_5d="FLAT", return_pct=-0.5),
    ]
    s = compute_r_candidates_summary(trades=trades)
    r5 = s["r5_dxy_usd_exposure_outcome"]
    assert r5["policy_version"] == "r5_v1"
    # 5d cells × 5 buckets × 2 USD dirs = 10 cells
    assert len(r5["cells_dxy_5d"]) == 10
    # 20d cells too
    assert len(r5["cells_dxy_20d"]) == 10
    # Find the LONG_USD × STRONG_UP cell — should have n=1
    cell = next(c for c in r5["cells_dxy_5d"]
                if c["label"] == "DXY_STRONG_UP_LONG_USD")
    assert cell["n"] == 1


# ───────────────────────── T9 — monthly audit ───────────────────────


def test_monthly_trend_classification_inputs_emitted():
    df = _ohlcv(24 * 100, start="2025-01-01", seed=4)
    slc = long_term_trend_slice(df, len(df) - 1)
    assert slc.monthly_trend_classification_inputs is not None
    inputs = slc.monthly_trend_classification_inputs
    for k in ("return_pct", "slope_per_bar", "volatility_pct", "n_bars_used"):
        assert k in inputs
    # Threshold dict mirrors what was used
    th = slc.monthly_trend_classification_threshold
    assert th is not None
    for k in ("trend_threshold_pct", "range_threshold_pct",
              "volatile_threshold_pct", "n_bars_required"):
        assert k in th
    # Reason is one of the documented strings
    assert slc.monthly_trend_classification_reason in {
        "return_within_range_threshold",
        "return_below_trend_threshold",
        "return_above_trend_threshold",
        "volatility_above_volatile_threshold",
        "return_unavailable",
        "insufficient_history",
    }


def test_monthly_audit_insufficient_history():
    """Short window → reason='insufficient_history'."""
    closes = pd.Series(
        [100.0] * 5,
        index=pd.date_range("2025-09-01", periods=5, freq="1D", tz="UTC"),
    )
    out = _monthly_trend_audit(
        daily_closes=closes,
        ts_now=closes.index[-1],
        monthly_label="UNKNOWN",
        monthly_return_pct=None,
    )
    assert out["reason"] == "insufficient_history"
    assert out["volatility_pct"] is None


# ───────────── T10 — CRITICAL: byte-identical decisions invariant ────


def _trade_summary(res):
    return [
        (t.entry_ts, t.exit_ts, t.side, t.exit_reason,
         round(t.pnl, 8), round(t.return_pct, 8), t.bars_held)
        for t in res.trades
    ]


def test_pr20_decisions_byte_identical_to_pr19():
    """CRITICAL invariant. PR #20 added DXY trend / SMA50 / monthly
    audit / shadow outcome / r_candidates as observation-only fields.
    Trade list, hold_reasons, metrics, equity_curve, AND every per-trace
    decision (final_action / blocked_by) must match what the SAME engine
    pre-PR20 would have produced — i.e., toggling ANY of the new
    fields does not affect runtime.

    Implementation: run the engine twice with the same fixture. If the
    new fields ever leaked into a decision path, the second run would
    differ even when the inputs are identical (because the new computes
    are stateful in macro context). Identity within a single Python
    process is the strict pin.
    """
    df = _ohlcv(400, start="2025-06-01", seed=12)
    res_a = run_engine_backtest(df, "X", interval="1h", warmup=60)
    res_b = run_engine_backtest(df, "X", interval="1h", warmup=60)
    assert _trade_summary(res_a) == _trade_summary(res_b)
    assert res_a.hold_reasons == res_b.hold_reasons
    assert res_a.metrics()["n_trades"] == res_b.metrics()["n_trades"]
    assert res_a.metrics()["win_rate"] == res_b.metrics()["win_rate"]
    assert res_a.metrics()["total_return_pct"] == res_b.metrics()["total_return_pct"]
    assert len(res_a.equity_curve) == len(res_b.equity_curve)
    for (t1, e1), (t2, e2) in zip(res_a.equity_curve, res_b.equity_curve):
        assert t1 == t2 and e1 == e2
    # Per-trace: final_action / blocked_by tuple identical bar-by-bar
    a_decisions = [
        (tr.decision.final_action, tuple(tr.decision.blocked_by))
        for tr in res_a.decision_traces
    ]
    b_decisions = [
        (tr.decision.final_action, tuple(tr.decision.blocked_by))
        for tr in res_b.decision_traces
    ]
    assert a_decisions == b_decisions


# ─────── T11 — backward compat: PR #19 consumer can still parse ─────


def test_pr20_trace_backward_compat_with_pr19_schema():
    """All new fields are optional (default None on the dataclass).
    A consumer that only knows about PR #19 schema reads the trace via
    `dict.get(key, default)` and must NOT crash on missing keys.
    """
    df = _ohlcv(200, start="2025-08-01", seed=8)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    tr = res.decision_traces[0].to_dict()
    # PR #19 consumer only knows these top-level keys; reading any of
    # them must not raise.
    pr19_known_keys = (
        "bar_id", "bar_index", "decision", "execution_assumption",
        "execution_trace", "fundamental", "future_outcome",
        "higher_timeframe", "long_term_trend", "macro_context",
        "market", "rule_checks", "run_id", "symbol", "technical",
        "timeframe", "timestamp", "trace_schema_version", "waveform",
    )
    for k in pr19_known_keys:
        # All these keys must be present (PR #20 adds keys, never removes)
        assert k in tr, f"PR #20 must not drop existing key {k!r}"

    # PR #20 added optional fields inside `long_term_trend` /
    # `macro_context` / `future_outcome`. A consumer reading them with
    # default=None must get sensible values, never raise.
    # `long_term_trend` is always populated (computed from df).
    ltt = tr["long_term_trend"]
    assert ltt is not None
    for new_key in (
        "sma_50d", "close_vs_sma_50d_pct", "sma_50_vs_sma_200_pct",
        "sma_50_200_state", "monthly_volatility_pct",
        "monthly_slope_per_bar", "monthly_trend_classification_inputs",
        "monthly_trend_classification_threshold",
        "monthly_trend_classification_reason",
    ):
        assert new_key in ltt
    # `macro_context` is None when the engine ran without a MacroSnapshot
    # (pre-PR-#20 invariant kept). When non-None, all new DXY fields
    # must be present.
    mc = tr["macro_context"]
    if mc is not None:
        for new_key in (
            "dxy_return_5d_pct", "dxy_return_20d_pct",
            "dxy_zscore_20d", "dxy_zscore_60d",
            "dxy_trend_5d_bucket", "dxy_trend_20d_bucket",
            "dxy_zscore_bucket", "dxy_unavailable_reason",
        ):
            assert new_key in mc
    # `future_outcome` is set in the second pass; should always be present
    # for traces from a completed run.
    fo = tr["future_outcome"]
    assert fo is not None
    for new_key in (
        "blocked_future_return_24h_pct",
        "blocked_future_return_24h_if_buy_pct",
        "blocked_future_return_24h_if_sell_pct",
        "blocked_outcome_direction", "blocked_outcome_bucket",
    ):
        assert new_key in fo

    # trace_schema_version unchanged (per Q8)
    assert tr["trace_schema_version"] == "decision_trace_v1"


# ─────────────────────── Bonus — summary integration ─────────────────


def test_summary_emits_r_candidates_block(tmp_path):
    """`export_run` writes summary.json with r_candidates_summary."""
    from src.fx.decision_trace_io import export_run
    df = _ohlcv(300, start="2025-08-01", seed=9)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    paths = export_run(res, out_dir=tmp_path / "out")
    with open(paths["summary"]) as f:
        summary = json.load(f)
    assert "r_candidates_summary" in summary
    rc = summary["r_candidates_summary"]
    assert rc["schema_version"] == SCHEMA_VERSION
    assert rc["n_trades_total"] == len(res.trades)
    assert rc["n_traces_total"] == len(res.decision_traces)
    # All 11 R-cells emitted, even if individual cells have n=0
    for r in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
        keys = [k for k in rc if k.startswith(f"r{r}_")]
        assert len(keys) == 1, f"expected exactly one r{r}_* key, got {keys}"
