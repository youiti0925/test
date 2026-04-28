"""PR #15 — waveform-match wiring into backtest / decision_trace.

Pinned guarantees:
  * `_filter_library_no_lookahead` excludes any sample whose forward-return
    label has not yet been realised by `bar_ts` (i.e. requires
    `sample.end_ts + horizon_dur <= bar_ts`). Boundary-case test.
  * `waveform_bias_dict` returns a dict (never None) so consumers always
    get a recorded reason — `unavailable_reason` when the lookup cannot
    run. None is reserved for "lookup not requested".
  * Backtest with `waveform_library=None` produces the same trades and
    metrics as before (regression for "decisions unchanged"). And with
    a library attached the trades are STILL identical — bias is
    observability only and is never forwarded to `decide_action`.
  * Old trace JSONL (pre-PR-#15) reads through `aggregate_stats` with
    every new cross_stats key bucketing as "unknown".
  * `aggregate_many` pools the new buckets correctly.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest
from src.fx.decision_trace_build import (
    _filter_library_no_lookahead,
    _horizon_duration,
    compute_library_id,
    waveform_bias_dict,
)
from src.fx.decision_trace_stats import aggregate_stats, aggregate_many
from src.fx.waveform_library import WaveformSample, build_library, write_library
from src.fx.waveform_matcher import WaveformSignature, compute_signature


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ohlcv(n: int, *, start: str = "2024-01-01", seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    drift = np.linspace(0, 4.0, n)
    noise = np.cumsum(rng.standard_normal(n) * 0.05)
    close = 100.0 + drift + noise
    return pd.DataFrame(
        {"open": close, "high": close + 0.05, "low": close - 0.05,
         "close": close, "volume": [1000.0] * n},
        index=idx,
    )


def _toy_library(n_samples: int, start: str = "2023-01-01") -> list[WaveformSample]:
    """Synthesise a small library — ts strictly before backtest period."""
    df = _ohlcv(24 * 60, start=start, seed=42)
    return build_library(
        df, symbol="X", timeframe="1h",
        window_bars=60, step_bars=24, forward_horizons=(24,),
    )


# ---------------------------------------------------------------------------
# _horizon_duration
# ---------------------------------------------------------------------------


def test_horizon_duration_known_intervals():
    assert _horizon_duration("1h", 24) == pd.Timedelta(hours=24)
    assert _horizon_duration("15m", 4) == pd.Timedelta(hours=1)
    assert _horizon_duration("1d", 5) == pd.Timedelta(days=5)


def test_horizon_duration_unknown_interval_returns_none():
    """Safe-side behaviour: if we can't resolve the interval, the caller
    must disable lookup rather than apply a weaker filter."""
    assert _horizon_duration("3h", 24) is None
    assert _horizon_duration("???", 24) is None
    assert _horizon_duration("1h", 0) is None
    assert _horizon_duration("1h", -1) is None


# ---------------------------------------------------------------------------
# _filter_library_no_lookahead — the critical boundary test
# ---------------------------------------------------------------------------


def _mk_sample(end_ts: pd.Timestamp, *, return_pct: float = 1.0) -> WaveformSample:
    """Minimal WaveformSample with a synthetic signature."""
    sig = WaveformSignature(
        vector=np.array([0.0, 1.0, 0.0]),
        structure=["HH", "HL", "LH"],
        swing_structure=["H", "L"],
        trend_state="UPTREND",
        detected_pattern=None,
        length=3,
        method="z_score",
        atr=None,
    )
    end_dt = end_ts.to_pydatetime() if isinstance(end_ts, pd.Timestamp) else end_ts
    return WaveformSample(
        symbol="X", timeframe="1h",
        start_ts=end_dt - timedelta(hours=60),
        end_ts=end_dt,
        signature=sig,
        forward_returns_pct={24: return_pct},
        max_favorable_pct=1.5, max_adverse_pct=-0.5,
    )


def test_filter_excludes_sample_with_unrealised_horizon():
    """sample.end_ts < bar_ts BUT sample.end_ts + horizon > bar_ts must
    be excluded — the label was not yet known at bar_ts."""
    bar_ts = pd.Timestamp("2025-01-15 10:00", tz="UTC")
    horizon_dur = pd.Timedelta(hours=24)

    # end_ts = bar_ts − 12h (younger than 1 day) → end_ts + 24h = bar_ts + 12h (future).
    s_unrealised = _mk_sample(bar_ts - pd.Timedelta(hours=12))
    out = _filter_library_no_lookahead([s_unrealised], bar_ts, horizon_dur)
    assert out == [], (
        "sample whose horizon has not completed by bar_ts must be excluded"
    )


def test_filter_includes_sample_with_horizon_completed_exactly_at_bar_ts():
    """end_ts + horizon == bar_ts: label is realised AT bar_ts → eligible."""
    bar_ts = pd.Timestamp("2025-01-15 10:00", tz="UTC")
    horizon_dur = pd.Timedelta(hours=24)
    s_boundary = _mk_sample(bar_ts - pd.Timedelta(hours=24))
    out = _filter_library_no_lookahead([s_boundary], bar_ts, horizon_dur)
    assert out == [s_boundary]


def test_filter_includes_sample_with_horizon_completed_earlier():
    bar_ts = pd.Timestamp("2025-01-15 10:00", tz="UTC")
    horizon_dur = pd.Timedelta(hours=24)
    s_old = _mk_sample(bar_ts - pd.Timedelta(days=10))
    out = _filter_library_no_lookahead([s_old], bar_ts, horizon_dur)
    assert out == [s_old]


def test_filter_handles_tz_naive_sample_ts():
    bar_ts = pd.Timestamp("2025-01-15 10:00", tz="UTC")
    horizon_dur = pd.Timedelta(hours=24)
    naive = pd.Timestamp("2025-01-13 10:00")  # tz-naive, two days before bar_ts
    sig = WaveformSignature(
        vector=np.array([0.0, 1.0]), structure=[], swing_structure=[],
        trend_state="UPTREND", detected_pattern=None, length=2,
        method="z_score", atr=None,
    )
    s = WaveformSample(
        symbol="X", timeframe="1h",
        start_ts=naive.to_pydatetime() - timedelta(hours=60),
        end_ts=naive.to_pydatetime(),
        signature=sig, forward_returns_pct={24: 0.5},
        max_favorable_pct=None, max_adverse_pct=None,
    )
    out = _filter_library_no_lookahead([s], bar_ts, horizon_dur)
    assert out == [s]


# ---------------------------------------------------------------------------
# waveform_bias_dict — never returns None when lookup is requested
# ---------------------------------------------------------------------------


def test_waveform_bias_dict_returns_unavailable_when_no_library_attached():
    df = _ohlcv(24 * 30)
    out = waveform_bias_dict(
        None, df, len(df) - 1, interval="1h",
    )
    assert out["unavailable_reason"] == "no library attached"


def test_waveform_bias_dict_returns_unavailable_for_unknown_interval():
    df = _ohlcv(24 * 30)
    lib = _toy_library(10)
    out = waveform_bias_dict(
        lib, df, len(df) - 1, interval="3h",
    )
    assert "unknown interval" in out["unavailable_reason"]


def test_waveform_bias_dict_returns_unavailable_when_window_too_short():
    df = _ohlcv(20)  # less than default window_bars=60
    lib = _toy_library(10)
    out = waveform_bias_dict(
        lib, df, len(df) - 1, interval="1h",
    )
    assert "bars available" in out["unavailable_reason"]


def test_waveform_bias_dict_populates_enriched_fields_when_lookup_runs():
    """When the lookup actually runs, the dict must carry all the
    PR-#15 enriched aliases / analytics."""
    df = _ohlcv(24 * 90, start="2024-01-01", seed=11)  # 90 days of 1h
    # Library from PRIOR period (out-of-sample), denser sampling so we
    # have enough samples after horizon-aware filter at any bar.
    lib = build_library(
        _ohlcv(24 * 200, start="2023-01-01", seed=7),
        symbol="X", timeframe="1h",
        window_bars=60, step_bars=4, forward_horizons=(24,),
    )
    out = waveform_bias_dict(
        lib, df, len(df) - 1, interval="1h",
        library_id="lib_v1", min_samples=5, min_score=0.0,
    )
    # Either lookup ran (full dict) or unavailable_reason is set;
    # ensure no silent missing key.
    assert "library_id" in out
    if "unavailable_reason" not in out:
        for k in (
            "matched_count", "expected_direction", "avg_future_return_pct",
            "median_future_return_pct", "top_similarity",
            "bullish_match_ratio", "bearish_match_ratio", "neutral_match_ratio",
            "library_size", "eligible_size",
        ):
            assert k in out, f"missing enriched key {k!r}"


# ---------------------------------------------------------------------------
# Decisions-unchanged regression: trades must NOT depend on waveform_library
# ---------------------------------------------------------------------------


def _metrics_tuple(res):
    m = res.metrics()
    return (
        m["n_trades"], m["win_rate"], m["profit_factor"],
        m["total_return_pct"], m["max_drawdown_pct"],
    )


def test_decisions_unchanged_with_or_without_waveform_library():
    """Same df, same events, same warmup — running with a library
    attached MUST produce the same trades and metrics as running
    without it. waveform_bias is observability only."""
    df = _ohlcv(24 * 100, start="2024-01-01", seed=11)
    lib = build_library(
        _ohlcv(24 * 200, start="2023-01-01", seed=7),
        symbol="X", timeframe="1h",
        window_bars=60, step_bars=8, forward_horizons=(24,),
    )

    res_no_lib = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
    )
    res_with_lib = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        waveform_library=lib, waveform_library_id="lib_v1",
    )

    assert _metrics_tuple(res_no_lib) == _metrics_tuple(res_with_lib), (
        "decisions changed when waveform library was attached — this PR "
        "must keep BUY/SELL/HOLD logic invariant"
    )
    # Trade-by-trade equality (sanity)
    assert len(res_no_lib.trades) == len(res_with_lib.trades)
    for t1, t2 in zip(res_no_lib.trades, res_with_lib.trades):
        assert t1.entry_ts == t2.entry_ts
        assert t1.side == t2.side
        assert t1.exit_reason == t2.exit_reason


def test_backtest_without_library_keeps_waveform_bias_null():
    df = _ohlcv(24 * 60)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    sample = res.decision_traces[20].to_dict()
    assert sample["waveform"]["waveform_bias"] is None


def test_backtest_with_library_attaches_waveform_bias_dict():
    df = _ohlcv(24 * 90, start="2024-01-01", seed=11)
    lib = build_library(
        _ohlcv(24 * 200, start="2023-01-01", seed=7),
        symbol="X", timeframe="1h",
        window_bars=60, step_bars=8, forward_horizons=(24,),
    )
    res = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        waveform_library=lib, waveform_library_id="lib_v1",
    )
    # Pick a bar after warmup so the window covers ≥60 bars.
    sample = res.decision_traces[80].to_dict()
    bias = sample["waveform"]["waveform_bias"]
    assert isinstance(bias, dict)
    assert bias["library_id"] == "lib_v1"
    # Either ran a lookup OR carries an explicit unavailable_reason.
    assert "matched_count" in bias or "unavailable_reason" in bias


def test_run_metadata_records_waveform_config():
    df = _ohlcv(24 * 60)
    lib = build_library(
        _ohlcv(24 * 200, start="2023-01-01", seed=7),
        symbol="X", timeframe="1h",
        window_bars=60, step_bars=20, forward_horizons=(24,),
    )
    res = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        waveform_library=lib, waveform_library_id="lib_v1",
        waveform_window_bars=60, waveform_horizon_bars=24,
        waveform_min_samples=20, waveform_min_score=0.55,
        waveform_min_share=0.6, waveform_method="dtw",
    )
    md = res.run_metadata.to_dict()
    # config_payload is hashed into strategy_config_hash; the full
    # config_payload isn't surfaced as a top-level key, but the SHA
    # must change vs no-library so we know it's recorded.
    res2 = run_engine_backtest(df, "X", interval="1h", warmup=60)
    md2 = res2.run_metadata.to_dict()
    assert md["strategy_config_hash"] != md2["strategy_config_hash"], (
        "waveform config must be part of strategy_config_hash"
    )


# ---------------------------------------------------------------------------
# Future-spike no-leak — full pipeline test
# ---------------------------------------------------------------------------


def test_waveform_lookup_in_backtest_uses_only_past_samples_at_each_bar():
    """Build a library that includes BOTH past samples (pre-backtest) and
    future samples (post-backtest). Run a backtest and verify each bar's
    `eligible_size` never includes any sample whose horizon hadn't
    completed by that bar."""
    df = _ohlcv(24 * 60, start="2024-06-01", seed=11)

    # Past library (well before backtest)
    past_lib = build_library(
        _ohlcv(24 * 200, start="2023-01-01", seed=7),
        symbol="X", timeframe="1h",
        window_bars=60, step_bars=10, forward_horizons=(24,),
    )
    # Future library (well AFTER backtest end). Their forward returns
    # are computed from data we have not "seen" in the backtest's
    # timeline — must be filtered out at every bar.
    future_lib = build_library(
        _ohlcv(24 * 200, start="2025-06-01", seed=13),
        symbol="X", timeframe="1h",
        window_bars=60, step_bars=10, forward_horizons=(24,),
    )
    lib = past_lib + future_lib

    res = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        waveform_library=lib, waveform_library_id="mixed",
        waveform_min_samples=5, waveform_min_score=0.0,
    )
    # Inspect every bar with a populated bias and confirm the eligible
    # set never grew faster than the horizon-aware cutoff would allow.
    backtest_end = df.index[-1]
    for trace in res.decision_traces:
        bias = trace.waveform.waveform_bias
        if not bias or "unavailable_reason" in bias:
            continue
        eligible_size = bias.get("eligible_size")
        # The future library has 14+ samples whose end_ts is past the
        # backtest's last bar; if eligible_size ever exceeded the past
        # library's count, we'd be using future data.
        assert eligible_size <= len(past_lib), (
            f"eligible_size {eligible_size} exceeds past-library size "
            f"{len(past_lib)} at bar ts {trace.timestamp} — look-ahead leak"
        )


# ---------------------------------------------------------------------------
# stats — backward compat & new buckets
# ---------------------------------------------------------------------------


def _legacy_record() -> dict:
    """Minimal trace record (pre-PR-#15)."""
    return {
        "run_id": "r", "trace_schema_version": "decision_trace_v1",
        "bar_id": "X_1h_2025-01-01T00:00:00+00:00",
        "timestamp": "2025-01-01T00:00:00+00:00",
        "symbol": "X", "timeframe": "1h", "bar_index": 0,
        "market": {}, "technical": {},
        "waveform": {"waveform_bias": None},  # explicit None
        "higher_timeframe": {}, "fundamental": {},
        "execution_assumption": {},
        "execution_trace": {"entry_executed": False},
        "rule_checks": [],
        "decision": {
            "final_action": "HOLD",
            "technical_only_action": "HOLD",
            "blocked_by": [],
            "rule_chain": [],
            "reason": "x",
        },
        "future_outcome": {
            "gate_effect": "NO_CHANGE",
            "outcome_if_technical_action_taken": "N/A",
        },
    }


def test_aggregate_stats_buckets_pre_pr15_trace_under_unknown(tmp_path):
    p = tmp_path / "old.jsonl"
    p.write_text(json.dumps(_legacy_record()) + "\n", encoding="utf-8")
    s = aggregate_stats(p)
    cs = s["cross_stats"]
    for key in (
        "waveform_bias_direction_outcome",
        "waveform_confidence_bucket_outcome",
        "top_similarity_bucket_outcome",
        "matched_count_bucket_outcome",
        "waveform_bias_by_technical_action_outcome",
        "symbol_waveform_bias_outcome",
    ):
        assert key in cs
    assert "unknown" in cs["waveform_bias_direction_outcome"]
    assert "unknown" in cs["waveform_confidence_bucket_outcome"]
    # Symbol cross uses composite key — symbol|wf_bias=unknown
    assert any(k.endswith("|wf_bias=unknown") for k in cs["symbol_waveform_bias_outcome"])


def test_aggregate_stats_with_waveform_bias_present(tmp_path):
    rec = _legacy_record()
    rec["decision"]["final_action"] = "BUY"
    rec["decision"]["technical_only_action"] = "BUY"
    rec["execution_trace"]["entry_executed"] = True
    rec["future_outcome"]["outcome_if_technical_action_taken"] = "WIN"
    rec["waveform"]["waveform_bias"] = {
        "action": "BUY",
        "expected_direction": "BUY",
        "confidence": 0.72,
        "matched_count": 25,
        "top_similarity": 0.81,
    }
    p = tmp_path / "new.jsonl"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    s = aggregate_stats(p)
    cs = s["cross_stats"]
    # bias=BUY → bucket "BUY", final=BUY, outcome=WIN
    assert cs["waveform_bias_direction_outcome"]["BUY"]["BUY"]["n"] == 1
    assert cs["waveform_bias_direction_outcome"]["BUY"]["BUY"]["outcome"]["WIN"] == 1
    # confidence 0.72 → "high>=0.6"
    assert "high>=0.6" in cs["waveform_confidence_bucket_outcome"]
    # similarity 0.81 → "strong>=0.75"
    assert "strong>=0.75" in cs["top_similarity_bucket_outcome"]
    # matched_count 25 → "10-29"
    assert "10-29" in cs["matched_count_bucket_outcome"]


def test_aggregate_many_pools_waveform_buckets(tmp_path):
    def write_run(p: Path, action: str):
        rec = _legacy_record()
        rec["decision"]["final_action"] = action
        rec["decision"]["technical_only_action"] = action
        rec["execution_trace"]["entry_executed"] = action != "HOLD"
        rec["future_outcome"]["outcome_if_technical_action_taken"] = "WIN"
        rec["waveform"]["waveform_bias"] = {
            "action": "BUY", "confidence": 0.7,
            "top_similarity": 0.8, "matched_count": 25,
        }
        rec["run_id"] = p.parent.name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    a = tmp_path / "run_a" / "decision_traces.jsonl"
    b = tmp_path / "run_b" / "decision_traces.jsonl"
    write_run(a, "BUY")
    write_run(b, "BUY")
    pooled = aggregate_many([a, b])
    cs = pooled["global"]["cross_stats"]
    assert cs["waveform_bias_direction_outcome"]["BUY"]["BUY"]["n"] == 2
