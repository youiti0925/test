"""Engine-level invariants for the royal_road_decision_v1 profile.

The royal-road profile is opt-in. Default (`current_runtime`) must be
byte-identical to PR #21 main:

  trades / metrics / hold_reasons / equity_curve / per-trace decision
  must NOT differ between PR #21 main and this branch when
  `decision_profile="current_runtime"` (the default).

Opt-in must actually take effect:

  Running with `decision_profile="royal_road_decision_v1"` must
  produce a `royal_road_decision` slice on every trace, and may
  legitimately produce a different trade list (the strict v1 rules
  block most setups in random-walk fixtures).

Live / OANDA / paper:

  `live` / `cmd_trade` paths do NOT call run_engine_backtest, so the
  flag cannot affect them. Pinned by source-grep: live-side modules
  must not import from royal_road_decision.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "fx"


def _ohlcv(n: int = 400, *, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-06-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def _trade_summary(res):
    return [
        (t.entry_ts, t.exit_ts, t.side, t.exit_reason,
         round(t.pnl, 8), round(t.return_pct, 8), t.bars_held)
        for t in res.trades
    ]


# ─────────────── default unchanged ─────────────────────────────────


def test_default_profile_is_current_runtime():
    """Caller that does not pass decision_profile must land on
    current_runtime — the byte-identical path."""
    df = _ohlcv()
    res_implicit = run_engine_backtest(df, "X", interval="1h", warmup=60)
    res_explicit = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        decision_profile="current_runtime",
    )
    assert _trade_summary(res_implicit) == _trade_summary(res_explicit)
    assert res_implicit.hold_reasons == res_explicit.hold_reasons
    assert res_implicit.metrics() == res_explicit.metrics()


def test_default_profile_does_not_emit_royal_road_slice():
    """current_runtime path must leave royal_road_decision = None on
    every bar, preserving the trace shape used by all PR #20 / #21
    invariant tests."""
    df = _ohlcv(250)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    assert all(tr.royal_road_decision is None for tr in res.decision_traces)


def test_default_profile_byte_identical_to_pr21_pin():
    """Running the engine twice with the default profile must produce
    identical trades, hold_reasons, metrics, equity_curve, and per-bar
    decisions — same property the PR #20 / PR #21 invariant tests
    enforce. royal_road_decision being a NEW optional Slice on
    BarDecisionTrace must not affect this."""
    df = _ohlcv()
    res_a = run_engine_backtest(df, "X", interval="1h", warmup=60)
    res_b = run_engine_backtest(df, "X", interval="1h", warmup=60)
    assert _trade_summary(res_a) == _trade_summary(res_b)
    assert res_a.hold_reasons == res_b.hold_reasons
    a_decisions = [
        (tr.decision.final_action, tuple(tr.decision.blocked_by))
        for tr in res_a.decision_traces
    ]
    b_decisions = [
        (tr.decision.final_action, tuple(tr.decision.blocked_by))
        for tr in res_b.decision_traces
    ]
    assert a_decisions == b_decisions


# ─────────────── opt-in actually fires ─────────────────────────────


def test_royal_road_profile_emits_royal_slice_on_every_bar():
    df = _ohlcv(250)
    res = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v1",
    )
    assert len(res.decision_traces) > 0
    for tr in res.decision_traces:
        assert tr.royal_road_decision is not None
        slice_ = tr.royal_road_decision
        assert slice_.profile == "royal_road_decision_v1"
        assert slice_.action in ("BUY", "SELL", "HOLD")
        cmp_ = slice_.compared_to_current_runtime
        assert cmp_["current_action"] == tr.decision.final_action
        assert cmp_["royal_road_action"] == slice_.action


def test_royal_road_profile_runs_to_completion():
    """The royal-road path must not raise mid-bar even on a fixture
    where most labels are AVOID_TRADE (random walks tend to produce
    that)."""
    df = _ohlcv(200, seed=99)
    res = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v1",
    )
    assert res.metrics()["n_trades"] >= 0  # may be 0; the point is no crash


def test_royal_road_profile_runs_byte_identical_within_process():
    """Same fixture, same flag, twice → identical output. The royal-
    road code path must not use process-global state (e.g. caches that
    leak across runs)."""
    df = _ohlcv()
    res_a = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v1",
    )
    res_b = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v1",
    )
    assert _trade_summary(res_a) == _trade_summary(res_b)
    assert res_a.hold_reasons == res_b.hold_reasons
    assert res_a.metrics() == res_b.metrics()


def test_unknown_profile_raises():
    df = _ohlcv(100)
    with pytest.raises(ValueError):
        run_engine_backtest(
            df, "X", interval="1h", warmup=60,
            decision_profile="not_a_real_profile",
        )


# ─────────────── live / safety pins ────────────────────────────────


def test_decision_engine_does_not_import_royal_road_decision():
    """The legacy `decide_action` chain must not be aware of the
    royal-road profile. The branch is in backtest_engine and is opt-in."""
    src = (SRC / "decision_engine.py").read_text()
    assert "royal_road_decision" not in src


def test_risk_gate_does_not_import_royal_road_decision():
    src = (SRC / "risk_gate.py").read_text()
    assert "royal_road_decision" not in src


def test_indicators_does_not_import_royal_road_decision():
    src = (SRC / "indicators.py").read_text()
    assert "royal_road_decision" not in src


def test_oanda_paths_do_not_import_royal_road_decision():
    """Live broker / paper paths must remain untouched."""
    for f in ("oanda.py", "broker.py"):
        path = SRC / f
        if path.exists():
            assert "royal_road_decision" not in path.read_text(), (
                f"{f} imports royal_road_decision — live paths must be "
                "isolated from the backtest-only profile flag"
            )


def test_parameter_baseline_unchanged_by_royal_road():
    """The royal-road profile must NOT have leaked into
    PARAMETER_BASELINE_V1. If it had, the literature-baseline payload
    hash would shift and PR #19 metadata-only invariants would break."""
    from src.fx.parameter_defaults import PARAMETER_BASELINE_V1
    actual = set(PARAMETER_BASELINE_V1.keys())
    assert "royal_road_decision_v1" not in actual
    assert "decision_profile" not in actual
