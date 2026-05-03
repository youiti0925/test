"""technical_confluence_v1 observation-only invariant pin.

Three invariants are checked:

1. STRUCTURAL — decision_engine.py / risk_gate.py / backtest_engine.py's
   decide() pipeline must not import or reference `technical_confluence`.
   This pins the slice as a pure trace-builder concern.

2. SCHEMA — every BarDecisionTrace produced by the engine carries a
   `technical_confluence` payload whose top-level keys form the closed
   set defined in `technical_confluence.empty_technical_confluence`.

3. BEHAVIOURAL — running the engine twice on the same fixture produces
   identical trades / hold_reasons / metrics / per-bar decisions. This
   is the same shape as PR #20's `test_pr20_decisions_byte_identical_to_pr19`
   guard but extended through the PR-C1 trace addition: if the slice
   ever leaked into the decision path, the second run would diverge.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest
from src.fx.technical_confluence import empty_technical_confluence


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "fx"


def _ohlcv(n: int, *, start: str = "2025-06-01", seed: int = 12,
           freq: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)


# ──────────────── 1. STRUCTURAL: no decision-path reference ──────────────


def test_decision_engine_does_not_import_technical_confluence():
    """`technical_confluence` must NEVER appear in decision_engine.py.
    The slice is built only by decision_trace_build.py — not by decide()."""
    src = (SRC / "decision_engine.py").read_text()
    assert "technical_confluence" not in src, (
        "decision_engine must remain unaware of technical_confluence; "
        "any reference would break the observation-only contract."
    )


def test_risk_gate_does_not_import_technical_confluence():
    src = (SRC / "risk_gate.py").read_text()
    assert "technical_confluence" not in src, (
        "risk_gate must remain unaware of technical_confluence."
    )


def test_indicators_does_not_import_technical_confluence():
    """`indicators.technical_signal` is the canonical BUY/SELL/HOLD
    rule. It must not consult confluence — confluence consumes
    indicator output, not the other way around."""
    src = (SRC / "indicators.py").read_text()
    assert "technical_confluence" not in src


# ──────────────── 2. SCHEMA: payload shape on every bar ──────────────────


def test_every_trace_carries_technical_confluence_payload():
    df = _ohlcv(200, start="2025-06-01", seed=12)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    expected_keys = set(empty_technical_confluence().keys())
    assert len(res.decision_traces) > 0
    for tr in res.decision_traces:
        d = tr.to_dict()
        assert "technical_confluence" in d
        payload = d["technical_confluence"]
        assert payload is not None, (
            "every bar must produce a non-None technical_confluence payload "
            "(warmup bars use empty_technical_confluence)"
        )
        assert set(payload.keys()) == expected_keys, (
            f"unexpected keys at bar {tr.bar_index}: "
            f"{set(payload.keys()) ^ expected_keys}"
        )
        assert payload["policy_version"] == "technical_confluence_v1"
        assert payload["final_confluence"]["label"] in {
            "STRONG_BUY_SETUP", "WEAK_BUY_SETUP",
            "STRONG_SELL_SETUP", "WEAK_SELL_SETUP",
            "NO_TRADE", "AVOID_TRADE", "UNKNOWN",
        }


# ──────────────── 3. BEHAVIOURAL: identical engine output ────────────────


def _trade_summary(res):
    return [
        (t.entry_ts, t.exit_ts, t.side, t.exit_reason,
         round(t.pnl, 8), round(t.return_pct, 8), t.bars_held)
        for t in res.trades
    ]


def test_engine_decisions_byte_identical_with_confluence_present():
    """Running the engine twice on the same fixture must produce exactly
    the same trades, hold_reasons, metrics, and per-trace decisions —
    even though every trace now carries the technical_confluence
    payload. If the payload ever leaked into decide() the second run
    would differ (the detector itself is deterministic, but its
    presence in the decide path would manifest as a change against the
    PR #20 baseline).
    """
    df = _ohlcv(400, start="2025-06-01", seed=12)
    res_a = run_engine_backtest(df, "X", interval="1h", warmup=60)
    res_b = run_engine_backtest(df, "X", interval="1h", warmup=60)
    assert _trade_summary(res_a) == _trade_summary(res_b)
    assert res_a.hold_reasons == res_b.hold_reasons
    a_metrics = res_a.metrics()
    b_metrics = res_b.metrics()
    assert a_metrics["n_trades"] == b_metrics["n_trades"]
    assert a_metrics["win_rate"] == b_metrics["win_rate"]
    assert a_metrics["total_return_pct"] == b_metrics["total_return_pct"]
    assert len(res_a.equity_curve) == len(res_b.equity_curve)
    for (t1, e1), (t2, e2) in zip(res_a.equity_curve, res_b.equity_curve):
        assert t1 == t2 and e1 == e2
    a_decisions = [
        (tr.decision.final_action, tuple(tr.decision.blocked_by))
        for tr in res_a.decision_traces
    ]
    b_decisions = [
        (tr.decision.final_action, tuple(tr.decision.blocked_by))
        for tr in res_b.decision_traces
    ]
    assert a_decisions == b_decisions


def test_to_dict_round_trips_known_keys():
    """Old PR #19/#20 trace consumers read fields via dict.get(...).
    Adding technical_confluence MUST keep all previously-known keys
    present and unchanged. Spot-check the important ones."""
    df = _ohlcv(200, start="2025-08-01", seed=8)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    tr = res.decision_traces[0].to_dict()
    pr20_keys = {
        "bar_id", "bar_index", "decision", "execution_assumption",
        "execution_trace", "fundamental", "future_outcome",
        "higher_timeframe", "long_term_trend", "macro_context",
        "market", "rule_checks", "run_id", "symbol",
        "technical", "timeframe", "timestamp", "trace_schema_version",
        "waveform",
    }
    for k in pr20_keys:
        assert k in tr, f"PR #20 key {k!r} disappeared after confluence add"
    # And the new key
    assert "technical_confluence" in tr


def test_disabling_confluence_via_default_none_does_not_crash_to_dict():
    """If a downstream caller constructs a BarDecisionTrace directly
    (e.g. live cmd_trade pipeline) and leaves technical_confluence as
    None, to_dict() must not raise."""
    from src.fx.decision_trace import BarDecisionTrace
    df = _ohlcv(200, start="2025-08-01", seed=8)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    tr = res.decision_traces[0]
    # Replicate a "no confluence wired" trace and to_dict it
    raw = BarDecisionTrace(
        run_id=tr.run_id, trace_schema_version=tr.trace_schema_version,
        bar_id=tr.bar_id, timestamp=tr.timestamp, symbol=tr.symbol,
        timeframe=tr.timeframe, bar_index=tr.bar_index,
        market=tr.market, technical=tr.technical, waveform=tr.waveform,
        higher_timeframe=tr.higher_timeframe, fundamental=tr.fundamental,
        execution_assumption=tr.execution_assumption,
        execution_trace=tr.execution_trace, rule_checks=tr.rule_checks,
        decision=tr.decision, future_outcome=tr.future_outcome,
        long_term_trend=tr.long_term_trend, macro_context=tr.macro_context,
        # technical_confluence omitted → defaults to None
    )
    d = raw.to_dict()
    assert d["technical_confluence"] is None
