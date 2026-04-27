"""Tests for the decision_trace v1 schema and backtest_engine wiring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest
from src.fx.decision_trace import (
    ENTRY_SKIPPED_REASONS,
    HORIZON_BARS_TABLE,
    RULE_TAXONOMY,
    TRACE_SCHEMA_VERSION,
    compute_data_snapshot_hash,
    compute_strategy_config_hash,
    horizon_to_bars,
)
from src.fx.indicators import (
    Snapshot,
    build_snapshot,
    technical_signal,
    technical_signal_reasons,
)


def _ohlcv(n: int, seed: int = 1, freq: str = "1h", drift: float = 0.0) -> pd.DataFrame:
    """Cheap synthetic OHLCV that's long enough for warmup + ATR."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5 + drift)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": [1000] * n,
        },
        index=idx,
    )
    return df


# ─── Structural tests ──────────────────────────────────────────────────────


def test_decision_traces_count_matches_bars_processed():
    df = _ohlcv(200, seed=1)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    assert len(res.decision_traces) == res.bars_processed
    assert res.bars_processed > 0


def test_each_trace_has_required_top_level_keys():
    df = _ohlcv(200, seed=2)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    required = {
        "run_id", "trace_schema_version", "bar_id", "timestamp",
        "symbol", "timeframe", "bar_index",
        "market", "technical", "waveform", "higher_timeframe",
        "fundamental", "execution_assumption", "execution_trace",
        "rule_checks", "decision", "future_outcome",
    }
    for trace in res.decision_traces:
        d = trace.to_dict()
        assert required <= set(d.keys()), (
            f"missing keys: {required - set(d.keys())}"
        )
        assert d["trace_schema_version"] == TRACE_SCHEMA_VERSION


def test_each_trace_has_full_19_rule_check_taxonomy():
    df = _ohlcv(200, seed=3)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    expected = set(RULE_TAXONOMY.keys())
    assert len(expected) == 19
    for trace in res.decision_traces:
        rids = {rc.canonical_rule_id for rc in trace.rule_checks}
        assert rids == expected, f"missing/extra rule_ids: {expected ^ rids}"


def test_rule_check_required_fields_present():
    df = _ohlcv(200, seed=4)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    required = {
        "canonical_rule_id", "rule_group", "result", "computed",
        "used_in_decision", "value", "threshold", "evidence_ids",
        "reason", "source_chain_step",
    }
    for trace in res.decision_traces:
        for rc in trace.rule_checks:
            d = rc.to_dict()
            assert required <= set(d.keys()), (
                f"missing keys on rule_check: {required - set(d.keys())}"
            )
            assert isinstance(d["evidence_ids"], list)
            assert isinstance(d["reason"], str)
            assert d["result"] in {"PASS", "BLOCK", "WARN", "SKIPPED",
                                    "NOT_REACHED", "INFO"}


def test_spread_check_skipped_in_synthetic_backtest():
    df = _ohlcv(200, seed=5)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    for trace in res.decision_traces:
        sp = next(rc for rc in trace.rule_checks
                  if rc.canonical_rule_id == "spread_abnormal")
        assert sp.result == "SKIPPED"
        assert "spread_pct is None" in sp.reason


def test_evidence_id_fields_present_with_missing_reason():
    df = _ohlcv(200, seed=6)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    for trace in res.decision_traces:
        f = trace.fundamental
        assert tuple(f.event_evidence_ids) == ()
        assert tuple(f.macro_evidence_ids) == ()
        assert tuple(f.news_evidence_ids) == ()
        assert "source_documents" in f.missing_event_evidence_reason
        assert f.missing_macro_evidence_reason
        assert f.missing_news_evidence_reason


def test_run_metadata_attached_unique_and_complete():
    df = _ohlcv(200, seed=7)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60,
                               data_source="yfinance")
    rm = res.run_metadata
    assert rm is not None
    assert rm.trace_schema_version == TRACE_SCHEMA_VERSION
    assert rm.run_id.startswith("bt_")
    assert "X" in rm.run_id
    assert rm.input_data_source == "yfinance"
    assert rm.synthetic_execution is True
    assert rm.commit_sha_status in {"resolved", "unknown_no_git",
                                     "unknown_subprocess_failed"}
    assert rm.strategy_config_hash and rm.data_snapshot_hash
    # All traces share the same run_id
    ids = {t.run_id for t in res.decision_traces}
    assert ids == {rm.run_id}


# ─── Regression / decision-invariance tests ────────────────────────────────


def _decisions_summary(res):
    return {
        "trades": [
            (t.entry_ts, t.side, t.exit_ts, t.exit_reason,
             round(t.return_pct, 6))
            for t in res.trades
        ],
        "hold_reasons": dict(res.hold_reasons),
        "bars_processed": res.bars_processed,
    }


def test_decisions_unchanged_with_trace_logging():
    df = _ohlcv(300, seed=11)
    res_on = run_engine_backtest(df, "X", interval="1h", warmup=60,
                                  capture_traces=True)
    res_off = run_engine_backtest(df, "X", interval="1h", warmup=60,
                                   capture_traces=False)
    assert _decisions_summary(res_on) == _decisions_summary(res_off)
    assert len(res_off.decision_traces) == 0


def test_hold_reasons_unchanged_with_trace_logging():
    df = _ohlcv(300, seed=12)
    res_on = run_engine_backtest(df, "X", interval="1h", warmup=60,
                                  capture_traces=True)
    res_off = run_engine_backtest(df, "X", interval="1h", warmup=60,
                                   capture_traces=False)
    assert dict(res_on.hold_reasons) == dict(res_off.hold_reasons)


def test_metrics_dict_unchanged_with_trace_logging():
    df = _ohlcv(300, seed=13)
    res_on = run_engine_backtest(df, "X", interval="1h", warmup=60,
                                  capture_traces=True)
    res_off = run_engine_backtest(df, "X", interval="1h", warmup=60,
                                   capture_traces=False)
    m_on = res_on.metrics()
    m_off = res_off.metrics()
    # The synthetic_execution metadata from PR #3 must still be present
    for key in ("synthetic_execution", "fill_model", "spread_mode",
                "slippage_mode", "bid_ask_mode", "sentiment_archive"):
        assert key in m_on and key in m_off
    # PR #4 must not change metric values
    assert m_on == m_off


# ─── technical_signal_reasons cannot drift ─────────────────────────────────


def test_technical_signal_reasons_action_equals_technical_signal():
    rng = np.random.default_rng(42)
    for _ in range(200):
        snap = Snapshot(
            symbol="X",
            last_close=float(rng.uniform(80, 120)),
            change_pct_1h=float(rng.uniform(-2, 2)),
            change_pct_24h=float(rng.uniform(-5, 5)),
            sma_20=float(rng.uniform(80, 120)),
            sma_50=float(rng.uniform(80, 120)),
            ema_12=float(rng.uniform(80, 120)),
            rsi_14=float(rng.uniform(0, 100)),
            macd=float(rng.uniform(-1, 1)),
            macd_signal=float(rng.uniform(-1, 1)),
            macd_hist=float(rng.uniform(-1, 1)),
            bb_upper=float(rng.uniform(100, 120)),
            bb_lower=float(rng.uniform(80, 100)),
            bb_position=float(rng.uniform(0, 1)),
        )
        action_canonical = technical_signal(snap)
        action_traced, _ = technical_signal_reasons(snap)
        assert action_canonical == action_traced


# ─── Future outcome — judgement neutrality + horizon table ─────────────────


def test_future_outcome_does_not_affect_decisions():
    df = _ohlcv(300, seed=21)
    res_with = run_engine_backtest(df, "X", interval="1h", warmup=60,
                                    compute_future_outcome=True)
    res_without = run_engine_backtest(df, "X", interval="1h", warmup=60,
                                       compute_future_outcome=False)
    assert _decisions_summary(res_with) == _decisions_summary(res_without)
    # When disabled, every trace's future_outcome must be None
    assert all(t.future_outcome is None for t in res_without.decision_traces)
    # When enabled, traces (excluding tail) have future_outcome set
    assert any(t.future_outcome is not None
               for t in res_with.decision_traces)


def test_future_outcome_horizons_truncate_near_end():
    df = _ohlcv(150, seed=22)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    last = res.decision_traces[-1]
    # The very last bar cannot have any forward returns
    assert last.future_outcome is not None
    assert last.future_outcome.unavailable_horizons
    assert "24h" in last.future_outcome.unavailable_horizons


@pytest.mark.parametrize("interval,expected", [
    ("15m", {"1h": 4, "4h": 16, "24h": 96}),
    ("1h", {"1h": 1, "4h": 4, "24h": 24}),
    ("4h", {"1h": None, "4h": 1, "24h": 6}),
])
def test_future_outcome_horizon_bars_per_interval(interval, expected):
    for h, bars in expected.items():
        assert horizon_to_bars(interval, h) == bars
        assert HORIZON_BARS_TABLE[interval][h] == bars


# ─── Position / execution trace ────────────────────────────────────────────


def test_entry_skipped_when_position_already_open():
    """Force a long-running position so a directional signal during it
    surfaces position_already_open + entry_execution BLOCK."""
    df = _ohlcv(400, seed=31)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60,
                               max_holding_bars=1000)
    # Find any trace where final_action ∈ BUY/SELL AND position_state BLOCK
    found = False
    for t in res.decision_traces:
        ps = next(rc for rc in t.rule_checks
                  if rc.canonical_rule_id == "position_state")
        if t.decision.final_action in ("BUY", "SELL") and ps.result == "BLOCK":
            assert ps.value == "already_in_position"
            ee = next(rc for rc in t.rule_checks
                      if rc.canonical_rule_id == "entry_execution")
            assert ee.result == "BLOCK"
            assert t.execution_trace.entry_executed is False
            assert t.execution_trace.entry_skipped_reason == "position_already_open"
            found = True
            break
    if not found:
        # Synthetic data may or may not produce this scenario — accept skip.
        pytest.skip("No bar with directional decision while position open in this fixture")


def test_position_state_not_block_when_final_action_hold():
    """When final_action=HOLD, position_state must NOT be BLOCK regardless
    of whether a position is open. Per user-corrected spec."""
    df = _ohlcv(300, seed=32)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60,
                               max_holding_bars=1000)
    for t in res.decision_traces:
        if t.decision.final_action == "HOLD":
            ps = next(rc for rc in t.rule_checks
                      if rc.canonical_rule_id == "position_state")
            assert ps.result != "BLOCK", (
                f"position_state must not be BLOCK when final_action=HOLD; "
                f"got {ps.result} on bar {t.bar_index}"
            )


def test_position_state_block_only_when_directional_with_open_position():
    """The only configuration that produces position_state=BLOCK is
    final_action ∈ {BUY,SELL} AND a position is already open."""
    df = _ohlcv(400, seed=33)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60,
                               max_holding_bars=1000)
    for t in res.decision_traces:
        ps = next(rc for rc in t.rule_checks
                  if rc.canonical_rule_id == "position_state")
        if ps.result == "BLOCK":
            assert t.decision.final_action in ("BUY", "SELL")
            assert t.execution_trace.had_open_position is True


def test_entry_skipped_reason_in_fixed_candidates():
    df = _ohlcv(300, seed=34)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    for t in res.decision_traces:
        assert t.execution_trace.entry_skipped_reason in ENTRY_SKIPPED_REASONS


def test_exit_event_recorded_for_stop_tp_or_max_holding():
    df = _ohlcv(400, seed=35, drift=0.05)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60,
                               max_holding_bars=20)
    found_exit = False
    for t in res.decision_traces:
        if t.execution_trace.exit_event:
            assert t.execution_trace.exit_reason in (
                "stop", "take_profit", "max_holding", "end_of_data",
            )
            assert t.execution_trace.exit_price is not None
            found_exit = True
    if not found_exit and res.trades:
        # If trades exist we should have at least one exit recorded
        pytest.fail("trades exist but no exit_event in any trace")


def test_trade_id_links_trace_and_engine_trade():
    df = _ohlcv(400, seed=36, drift=0.03)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    if not res.trades:
        pytest.skip("no trades produced for this fixture")
    trace_trade_ids = {
        t.execution_trace.trade_id for t in res.decision_traces
        if t.execution_trace.trade_id
    }
    engine_trade_ids = {tr.trade_id for tr in res.trades if tr.trade_id}
    assert engine_trade_ids
    assert engine_trade_ids <= trace_trade_ids


# ─── MarketSlice quality fields ────────────────────────────────────────────


def test_market_slice_quality_fields_populated():
    df = _ohlcv(200, seed=41)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    required = {"data_quality", "bars_available", "missing_ohlcv_count",
                "has_nan", "index_monotonic", "duplicate_timestamp_count",
                "timezone", "gap_detected", "quality_reason"}
    for t in res.decision_traces:
        d = t.market.to_dict()
        assert required <= set(d.keys())
        assert isinstance(d["bars_available"], int)
        assert isinstance(d["has_nan"], bool)
        assert d["timezone"] in ("UTC", "naive") or "UTC" in str(d["timezone"])


def test_market_slice_detects_nan_quality():
    df = _ohlcv(200, seed=42)
    # Inject a NaN into the last 5 bars so MarketSlice flags it.
    df.iloc[-2, df.columns.get_loc("close")] = np.nan
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    t_last = res.decision_traces[-1]
    assert t_last.market.has_nan is True
    assert t_last.market.data_quality != "OK"
    assert t_last.market.quality_reason


# ─── Missing evidence reasons ──────────────────────────────────────────────


def test_missing_evidence_reasons_distinct_from_empty_arrays():
    df = _ohlcv(200, seed=51)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    for t in res.decision_traces:
        f = t.fundamental
        # arrays empty
        assert list(f.event_evidence_ids) == []
        assert list(f.macro_evidence_ids) == []
        assert list(f.news_evidence_ids) == []
        # reasons non-empty AND distinct from empty arrays
        assert f.missing_event_evidence_reason.strip() != ""
        assert f.missing_macro_evidence_reason.strip() != ""
        assert f.missing_news_evidence_reason.strip() != ""


# ─── Hashes ────────────────────────────────────────────────────────────────


def test_data_snapshot_hash_stable_for_same_input():
    df = _ohlcv(150, seed=61)
    h1 = compute_data_snapshot_hash(df, "X", "1h")
    h2 = compute_data_snapshot_hash(df.copy(), "X", "1h")
    assert h1 == h2


def test_data_snapshot_hash_changes_with_ohlcv():
    df = _ohlcv(150, seed=62)
    h1 = compute_data_snapshot_hash(df, "X", "1h")
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("close")] += 0.5
    h2 = compute_data_snapshot_hash(df2, "X", "1h")
    assert h1 != h2


def test_data_snapshot_hash_changes_with_interval():
    df = _ohlcv(150, seed=63)
    assert (compute_data_snapshot_hash(df, "X", "1h")
            != compute_data_snapshot_hash(df, "X", "15m"))


def test_strategy_config_hash_stable_and_sensitive():
    cfg_a = {"warmup": 50, "stop_atr_mult": 2.0}
    cfg_b = {"stop_atr_mult": 2.0, "warmup": 50}  # same content, diff order
    cfg_c = {"warmup": 60, "stop_atr_mult": 2.0}  # different value
    assert compute_strategy_config_hash(cfg_a) == compute_strategy_config_hash(cfg_b)
    assert compute_strategy_config_hash(cfg_a) != compute_strategy_config_hash(cfg_c)


# ─── Hypothetical technical trade fields ───────────────────────────────────


def test_hypothetical_technical_trade_fields_present():
    df = _ohlcv(300, seed=71)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    for t in res.decision_traces:
        if t.future_outcome is None:
            continue
        d = t.future_outcome.to_dict()
        for k in (
            "hypothetical_technical_trade_exit_reason",
            "hypothetical_technical_trade_exit_price",
            "hypothetical_technical_trade_bars_held",
            "hypothetical_technical_trade_return_pct",
        ):
            assert k in d


def test_hypothetical_technical_trade_does_not_affect_decisions():
    """The hypothetical simulation walks future bars but must never feed
    back into BUY/SELL/HOLD. Verified by toggling future_outcome which
    is the only path that produces hypothetical_*."""
    df = _ohlcv(300, seed=72)
    res_with = run_engine_backtest(df, "X", interval="1h", warmup=60,
                                    compute_future_outcome=True)
    res_without = run_engine_backtest(df, "X", interval="1h", warmup=60,
                                       compute_future_outcome=False)
    assert _decisions_summary(res_with) == _decisions_summary(res_without)


# ─── to_dict / record export ───────────────────────────────────────────────


def test_to_decision_trace_records_returns_jsonable_dicts():
    import json
    df = _ohlcv(200, seed=81)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    records = res.to_decision_trace_records()
    assert isinstance(records, list)
    assert len(records) == res.bars_processed
    # Round-trip through json — would raise if any non-serialisable scalar
    blob = json.dumps(records[0], default=str)
    assert "trace_schema_version" in blob
