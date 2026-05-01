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


# ─── PR #16: waveform run_metadata audit fields ────────────────────────────


_PR16_REQUIRED_WAVEFORM_KEYS = {
    "waveform_enabled",
    "waveform_library_id",
    "waveform_library_size",
    "waveform_library_schema",
    "waveform_library_path_basename",
    "waveform_library_first_end_ts",
    "waveform_library_last_end_ts",
    "waveform_window_bars",
    "waveform_horizon_bars",
    "waveform_min_samples",
    "waveform_min_score",
    "waveform_min_share",
    "waveform_method",
}


def test_run_metadata_includes_waveform_section_when_no_library_attached():
    """PR #16: a backtest run WITHOUT --waveform-library still records
    its waveform run config in run_metadata.json so the absence is
    explicit. Otherwise readers cannot distinguish "library wasn't
    attached" from "field was added later and is missing for older
    runs"."""
    df = _ohlcv(200, seed=3)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    rm_dict = res.run_metadata.to_dict()
    assert "waveform" in rm_dict
    assert rm_dict["waveform"] is not None
    wf = rm_dict["waveform"]
    assert set(wf.keys()) == _PR16_REQUIRED_WAVEFORM_KEYS
    # No library → enabled=False, all library-derived fields null/0,
    # but the parameter knobs (window/horizon/method) still surface.
    assert wf["waveform_enabled"] is False
    assert wf["waveform_library_id"] is None
    assert wf["waveform_library_size"] == 0
    assert wf["waveform_library_schema"] is None
    assert wf["waveform_library_path_basename"] is None
    assert wf["waveform_library_first_end_ts"] is None
    assert wf["waveform_library_last_end_ts"] is None
    assert wf["waveform_window_bars"] == 60
    assert wf["waveform_horizon_bars"] == 24
    assert wf["waveform_method"] == "dtw"


def test_run_metadata_includes_waveform_section_when_library_attached():
    """PR #16: with a library attached, run_metadata exposes basename,
    schema (v1/v2), and first/last_end_ts parsed out of library_id so
    a future audit can tell from run_metadata.json alone which library
    fed the run, without having to open decision_traces.jsonl."""
    from src.fx.waveform_library import build_library
    from src.fx.decision_trace_build import compute_library_id

    history = _ohlcv(24 * 200, seed=11)
    lib = build_library(
        history, symbol="X", timeframe="1h",
        window_bars=60, step_bars=8, forward_horizons=(24,),
    )
    assert lib, "fixture build_library produced 0 samples"
    library_id = compute_library_id(lib, "/tmp/somewhere/usdjpy_train.jsonl")

    df = _ohlcv(200, seed=12)
    res = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        waveform_library=lib, waveform_library_id=library_id,
    )
    wf = res.run_metadata.to_dict()["waveform"]
    assert set(wf.keys()) == _PR16_REQUIRED_WAVEFORM_KEYS
    assert wf["waveform_enabled"] is True
    assert wf["waveform_library_id"] == library_id
    assert wf["waveform_library_size"] == len(lib)
    # PR #15+ build → schema=v2; basename only (no absolute path leakage);
    # endpoints match the library's first/last sample end_ts in iso form.
    assert wf["waveform_library_schema"] == "v2"
    assert wf["waveform_library_path_basename"] == "usdjpy_train.jsonl"
    assert "/" not in wf["waveform_library_path_basename"]
    assert wf["waveform_library_first_end_ts"] == lib[0].end_ts.isoformat()
    assert wf["waveform_library_last_end_ts"] == lib[-1].end_ts.isoformat()


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


def _stub_decide_always(action: str, *, blocked_by=()):
    """Build a `decide` stub that returns the same action on every bar.

    Used to make position-management traces deterministic without relying
    on whatever BUY/SELL/HOLD the technical signal happens to produce on
    a synthetic fixture.
    """
    from src.fx.decision_engine import Decision

    def _stub(*, technical_signal, pattern, higher_timeframe_trend,
              risk_reward, risk_state, llm_signal=None, waveform_bias=None,
              min_confidence=0.6, min_risk_reward=1.5):
        if blocked_by:
            return Decision(
                action="HOLD", confidence=0.0,
                reason="forced HOLD for test",
                blocked_by=tuple(blocked_by),
                rule_chain=("risk_gate",), advisory={},
            )
        return Decision(
            action=action, confidence=0.9,
            reason=f"forced {action} for test",
            blocked_by=(),
            rule_chain=(
                "risk_gate", "technical_directionality", "pattern_check",
                "higher_tf_alignment", "risk_reward_floor",
            ),
            advisory={},
        )
    return _stub


def test_entry_skipped_when_position_already_open(monkeypatch):
    """Deterministic: force decide to always return BUY and use stop/TP wide
    enough that the position never closes. Once a position is open, every
    subsequent bar must surface position_already_open + entry_execution BLOCK."""
    from src.fx import backtest_engine as bte

    monkeypatch.setattr(bte, "decide_action", _stub_decide_always("BUY"))
    df = _ohlcv(200, seed=31)
    res = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        max_holding_bars=10_000,
        stop_atr_mult=200.0,
        tp_atr_mult=200.0,
    )
    # First directional bar opens the position; from then on, every bar
    # with a directional decision while the position is open must BLOCK.
    blocked_bars = []
    for t in res.decision_traces:
        if (t.decision.final_action in ("BUY", "SELL")
                and t.execution_trace.had_open_position):
            blocked_bars.append(t)

    assert blocked_bars, (
        "Expected at least one bar with directional decision while position "
        "is open; got none. Stub may not be wired up."
    )
    # Spot-check the first such bar — every required field must hold.
    t = blocked_bars[0]
    ps = next(rc for rc in t.rule_checks
              if rc.canonical_rule_id == "position_state")
    ee = next(rc for rc in t.rule_checks
              if rc.canonical_rule_id == "entry_execution")
    assert t.decision.final_action in ("BUY", "SELL")
    assert t.execution_trace.had_open_position is True
    assert ps.result == "BLOCK"
    assert ps.value == "already_in_position"
    assert ee.result == "BLOCK"
    assert t.execution_trace.entry_executed is False
    assert t.execution_trace.entry_skipped_reason == "position_already_open"


def test_end_of_data_exit_recorded_in_last_trace(monkeypatch):
    """Force a position to remain open until the loop ends; the LAST trace
    must reflect the engine's end_of_data force-close — otherwise EngineTrade
    has the close but decision_trace says the position is still open."""
    from src.fx import backtest_engine as bte

    monkeypatch.setattr(bte, "decide_action", _stub_decide_always("BUY"))
    df = _ohlcv(200, seed=132)
    res = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        max_holding_bars=10_000,
        stop_atr_mult=200.0,
        tp_atr_mult=200.0,
    )
    # The engine must have force-closed exactly one trade with end_of_data.
    eod_trades = [t for t in res.trades if t.exit_reason == "end_of_data"]
    assert eod_trades, "expected end_of_data trade from forced-BUY fixture"
    last_trade = eod_trades[-1]

    last_trace = res.decision_traces[-1]
    et = last_trace.execution_trace
    assert et.exit_event is True
    assert et.exit_reason == "end_of_data"
    assert et.exit_price == pytest.approx(float(df["close"].iloc[-1]))
    assert et.exit_trade_id == last_trade.trade_id
    assert et.position_after is None

    # exit_check rule_check stays consistent with the execution_trace.
    ec = next(rc for rc in last_trace.rule_checks
              if rc.canonical_rule_id == "exit_check")
    assert ec.result == "PASS"
    assert isinstance(ec.value, dict)
    assert ec.value.get("exited") is True
    assert ec.value.get("exit_reason") == "end_of_data"


def test_same_bar_exit_and_entry_records_both_trade_ids(monkeypatch):
    """When a stop closes the existing position AND a new BUY fires on the
    same bar, the trace must carry BOTH trade_ids: exit_trade_id for the
    closed trade and entry_trade_id for the freshly opened one."""
    from src.fx import backtest_engine as bte

    monkeypatch.setattr(bte, "decide_action", _stub_decide_always("BUY"))
    df = _ohlcv(300, seed=133, drift=0.0)
    # Tight stop so positions get knocked out frequently; with stub always
    # BUY, the bar where stop fires also opens a fresh position.
    res = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        max_holding_bars=200,
        stop_atr_mult=0.25,
        tp_atr_mult=10.0,
    )
    same_bar = [
        t for t in res.decision_traces
        if t.execution_trace.exit_event
        and t.execution_trace.entry_executed
    ]
    assert same_bar, (
        "expected at least one bar with simultaneous exit + entry under "
        "tight stop + always-BUY stub"
    )
    t = same_bar[0]
    et = t.execution_trace
    assert et.entry_trade_id is not None
    assert et.exit_trade_id is not None
    assert et.entry_trade_id != et.exit_trade_id
    # Both ids must also link back to EngineTrade entries.
    engine_ids = {tr.trade_id for tr in res.trades}
    assert et.exit_trade_id in engine_ids
    # Self-consistency: position_state must NOT be BLOCK on this bar —
    # the exit cleared the slot before the entry, so entry was allowed.
    ps = next(rc for rc in t.rule_checks
              if rc.canonical_rule_id == "position_state")
    assert ps.result == "PASS"
    assert ps.value == "exit_then_entry"
    ee = next(rc for rc in t.rule_checks
              if rc.canonical_rule_id == "entry_execution")
    assert ee.result == "PASS"


def test_technical_directionality_value_matches_action():
    """rule_check[technical_directionality].value MUST be the actual
    technical_only_action — never derived from advisory or pattern. This
    pins the bug where pattern presence alone could record value='BUY'."""
    df = _ohlcv(300, seed=134)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    # At least one trace must have technical_directionality computed.
    checked = 0
    for t in res.decision_traces:
        # atr_unavailable bars have technical_directionality NOT_REACHED
        # (computed=False); skip those — only check fully-processed bars.
        td = next(rc for rc in t.rule_checks
                  if rc.canonical_rule_id == "technical_directionality")
        if not td.computed:
            continue
        # value must equal the technical_only_action stored on the slice
        # AND on the decision.
        assert td.value == t.technical.technical_only_action
        assert td.value == t.decision.technical_only_action
        assert td.value in ("BUY", "SELL", "HOLD")
        checked += 1
    assert checked > 0, "expected at least one bar with computed technical_directionality"


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
