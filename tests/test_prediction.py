"""Tests for the prediction-evaluation logic.

All synthetic — no network, no Claude calls.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.fx.prediction import evaluate_prediction, slice_bars_after


def _bars(closes: list[float], highs=None, lows=None) -> pd.DataFrame:
    n = len(closes)
    highs = highs or [c * 1.001 for c in closes]
    lows = lows or [c * 0.999 for c in closes]
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000] * n,
        },
        index=idx,
    )


def _prediction(
    direction: str = "UP",
    magnitude: float = 1.0,
    horizon: int = 4,
    invalidation: float | None = None,
    entry: float = 100.0,
):
    return {
        "entry_price": entry,
        "expected_direction": direction,
        "expected_magnitude_pct": magnitude,
        "horizon_bars": horizon,
        "invalidation_price": invalidation,
    }


def test_correct_up_prediction():
    # Up 1.5% within 4 bars; expected up 1%
    bars = _bars([101.0, 101.2, 101.5, 101.6])
    v = evaluate_prediction(_prediction("UP", 1.0, 4), bars)
    assert v.status == "CORRECT"
    assert v.actual_direction == "UP"


def test_partial_up_prediction():
    # Up 0.6% but expected up 1.0% — half-target → PARTIAL
    bars = _bars([100.2, 100.3, 100.5, 100.6])
    v = evaluate_prediction(_prediction("UP", 1.0, 4), bars)
    assert v.status == "PARTIAL"
    assert v.actual_direction == "UP"


def test_wrong_direction():
    # Expected UP but actually went DOWN
    bars = _bars([99.5, 99.0, 98.5, 98.0])
    v = evaluate_prediction(_prediction("UP", 1.0, 4), bars)
    assert v.status == "WRONG"
    assert v.actual_direction == "DOWN"


def test_invalidation_hit_marks_wrong_even_if_recovers():
    # Drops below invalidation 98 then recovers — still WRONG
    bars = _bars(
        closes=[99.0, 97.5, 99.0, 100.5],
        highs=[100.0, 99.0, 100.0, 101.0],
        lows=[97.0, 97.0, 98.5, 100.0],
    )
    pred = _prediction("UP", 1.0, 4, invalidation=98.0)
    v = evaluate_prediction(pred, bars)
    assert v.status == "WRONG"
    assert v.invalidation_hit is True


def test_correct_flat_prediction_stays_in_band():
    # Predicted FLAT with ±0.5% tolerance; max excursion is well under
    bars = _bars([100.1, 100.2, 100.05, 100.0])
    v = evaluate_prediction(_prediction("FLAT", 0.5, 4), bars)
    assert v.status == "CORRECT"


def test_wrong_flat_prediction_breaks_band():
    # Predicted FLAT but moved >2% in window
    bars = _bars(
        closes=[101.0, 102.0, 103.0, 102.5],
        highs=[101.5, 102.5, 103.5, 103.0],
        lows=[100.5, 101.5, 102.0, 102.0],
    )
    v = evaluate_prediction(_prediction("FLAT", 0.5, 4), bars)
    assert v.status == "WRONG"


def test_insufficient_data():
    bars = _bars([100.5, 101.0])  # only 2 bars, but horizon=4
    v = evaluate_prediction(_prediction("UP", 1.0, 4), bars)
    assert v.status == "INSUFFICIENT_DATA"


def test_wrong_down_prediction_invalidation():
    # Expected DOWN with invalidation at 102, price spikes to 102.5
    bars = _bars(
        closes=[101.0, 102.0, 101.5, 100.5],
        highs=[101.5, 102.5, 102.0, 101.5],
        lows=[100.5, 101.0, 101.0, 100.0],
    )
    pred = _prediction("DOWN", 1.0, 4, invalidation=102.0)
    v = evaluate_prediction(pred, bars)
    assert v.status == "WRONG"
    assert v.invalidation_hit is True


def test_max_favorable_and_adverse_signs():
    # UP prediction; price goes up 2% then back to up 0.5%
    bars = _bars(
        closes=[101.0, 102.0, 101.5, 100.5],
        highs=[101.5, 102.5, 102.0, 101.5],
        lows=[100.5, 101.0, 101.0, 100.0],
    )
    v = evaluate_prediction(_prediction("UP", 1.0, 4), bars)
    assert v.max_favorable_pct > 2.0  # high reached 102.5 → +2.5%
    # max adverse should be the worst drawdown vs entry (low 100.0)
    assert v.max_adverse_pct == pytest.approx(0.0, abs=1e-3)


def test_slice_bars_after_returns_strict_after():
    bars = _bars([100, 101, 102, 103, 104])
    target_ts = bars.index[1]
    sliced = slice_bars_after(bars, target_ts)
    assert len(sliced) == 3  # bars 2, 3, 4
    assert sliced.index[0] > target_ts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
