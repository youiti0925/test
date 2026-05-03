"""Comparison trace fields produced by royal_road_decision_v1 profile.

When `decision_profile="royal_road_decision_v1"` is active, every
trace must carry a `royal_road_decision` slice whose
`compared_to_current_runtime` block exposes the difference taxonomy:
  same | current_buy_royal_hold | current_sell_royal_hold |
  current_hold_royal_buy | current_hold_royal_sell |
  opposite_direction | other

These tests use a synthetic OHLCV fixture and assert that the
comparison block is internally consistent and that the closed
taxonomy is preserved.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.fx.backtest_engine import run_engine_backtest


_VALID_DIFF_TYPES = {
    "same",
    "current_buy_royal_hold",
    "current_sell_royal_hold",
    "current_hold_royal_buy",
    "current_hold_royal_sell",
    "opposite_direction",
    "other",
}


def _ohlcv(n: int = 250, *, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-06-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def test_compared_block_present_and_self_consistent():
    res = run_engine_backtest(
        _ohlcv(), "X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v1",
    )
    for tr in res.decision_traces:
        slice_ = tr.royal_road_decision
        assert slice_ is not None
        cmp_ = slice_.compared_to_current_runtime
        # Schema
        assert set(cmp_.keys()) == {
            "current_action", "royal_road_action",
            "same_action", "difference_type",
        }
        # Cross-field consistency: when royal-road is the active
        # profile, `tr.decision.final_action` IS the royal-road action
        # (the engine already used it). `compared_to_current_runtime
        # .current_action` records what the legacy decide_action chain
        # would have produced for the same bar.
        assert cmp_["royal_road_action"] == slice_.action
        assert cmp_["royal_road_action"] == tr.decision.final_action
        assert cmp_["same_action"] is (
            cmp_["current_action"] == cmp_["royal_road_action"]
        )
        assert cmp_["difference_type"] in _VALID_DIFF_TYPES


def test_difference_type_distribution_uses_closed_taxonomy():
    """Across an entire run, every observed `difference_type` must
    come from the closed set. Reaching this assertion proves the
    classifier never invents a new label."""
    res = run_engine_backtest(
        _ohlcv(400, seed=12), "X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v1",
    )
    diffs = Counter(
        tr.royal_road_decision.compared_to_current_runtime["difference_type"]
        for tr in res.decision_traces
    )
    assert sum(diffs.values()) == len(res.decision_traces)
    assert set(diffs.keys()).issubset(_VALID_DIFF_TYPES)


def test_to_dict_serialises_comparison_block():
    """`BarDecisionTrace.to_dict()["royal_road_decision"]` must include
    `compared_to_current_runtime` so JSONL exporters preserve it."""
    res = run_engine_backtest(
        _ohlcv(150), "X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v1",
    )
    payload = res.decision_traces[0].to_dict()["royal_road_decision"]
    assert payload is not None
    assert payload["profile"] == "royal_road_decision_v1"
    assert "action" in payload
    assert "score" in payload
    assert "reasons" in payload
    assert "block_reasons" in payload
    cmp_ = payload["compared_to_current_runtime"]
    assert cmp_["difference_type"] in _VALID_DIFF_TYPES


def test_default_profile_to_dict_emits_null_royal_road_decision():
    """When the default profile is used, the trace dict must carry
    `royal_road_decision: null` (not the key missing) so consumers
    can rely on shape stability across runs."""
    res = run_engine_backtest(_ohlcv(150), "X", interval="1h", warmup=60)
    d = res.decision_traces[0].to_dict()
    assert "royal_road_decision" in d
    assert d["royal_road_decision"] is None


def test_pure_random_walk_blocks_most_trades_under_royal_road():
    """A random-walk fixture rarely produces STRONG_BUY_SETUP +
    bullish evidence + invalidation_clear. Royal-road profile should
    therefore hold most bars; this is a behavioural sanity check, NOT
    a performance claim."""
    res = run_engine_backtest(
        _ohlcv(400, seed=42), "X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v1",
    )
    n_total = len(res.decision_traces)
    n_hold = sum(
        1 for tr in res.decision_traces
        if tr.royal_road_decision.action == "HOLD"
    )
    # On random walks, royal-road v1 should hold > 80% of bars.
    # (Not asserting performance — just that strict rules apply.)
    assert n_hold / n_total >= 0.8
