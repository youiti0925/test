"""Tests for waveform forward-statistics aggregation (spec §7.3)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.fx.waveform_backtest import (
    WaveformBias,
    aggregate_bias,
    find_similar,
    waveform_lookup,
)
from src.fx.waveform_library import WaveformSample
from src.fx.waveform_matcher import WaveformSignature, compute_signature


def _vec(seed: int, n: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n)


def _signature(vec: np.ndarray, structure: list[str] | None = None,
               trend: str = "RANGE") -> WaveformSignature:
    return WaveformSignature(
        vector=vec, structure=structure or [], swing_structure=[],
        trend_state=trend, detected_pattern=None, length=len(vec),
        method="z_score",
    )


def _sample(vec: np.ndarray, forward: dict[int, float | None]) -> WaveformSample:
    sig = _signature(vec)
    return WaveformSample(
        symbol="X", timeframe="1h",
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_ts=datetime(2024, 1, 2, tzinfo=timezone.utc),
        signature=sig,
        forward_returns_pct=forward,
        max_favorable_pct=None, max_adverse_pct=None,
    )


def test_below_min_sample_count_returns_hold_zero_confidence():
    target = _signature(_vec(0))
    library = [_sample(_vec(1), {24: 0.5}) for _ in range(5)]
    bias = aggregate_bias(
        [type("M", (), {"score": 0.9, "sample": s})() for s in library],
        horizon_bars=24, min_sample_count=20,
    )
    assert bias.action == "HOLD"
    assert bias.confidence == 0.0


def test_unanimous_bullish_yields_buy_with_confidence_capped_by_share():
    target = _signature(_vec(0))
    samples = [_sample(_vec(i + 1), {24: 0.5}) for i in range(25)]
    matches = [type("M", (), {"score": 0.8, "sample": s})() for s in samples]
    bias = aggregate_bias(matches, horizon_bars=24, min_sample_count=20)
    assert bias.action == "BUY"
    # 100% bullish, score 0.8 → confidence ≈ 0.8 (capped by score)
    assert bias.bullish_count == 25
    assert bias.bearish_count == 0
    # confidence = min(agree=1.0, agree*score=0.8) = 0.8
    assert bias.confidence == pytest.approx(0.8, abs=1e-9)


def test_split_60_40_remains_hold_when_share_below_threshold():
    samples = (
        [_sample(_vec(i), {24: 0.5}) for i in range(11)]
        + [_sample(_vec(100 + i), {24: -0.5}) for i in range(9)]
    )
    matches = [type("M", (), {"score": 0.8, "sample": s})() for s in samples]
    # 11/20 = 55% bullish; threshold is 60% → HOLD
    bias = aggregate_bias(
        matches, horizon_bars=24, min_sample_count=20,
        min_directional_share=0.6,
    )
    assert bias.action == "HOLD"


def test_neutral_band_keeps_tiny_returns_out_of_vote():
    """Returns inside ±neutral_band_pct count neutral, not directional."""
    samples = (
        [_sample(_vec(i), {24: 0.001}) for i in range(20)]  # tiny noise
    )
    matches = [type("M", (), {"score": 0.8, "sample": s})() for s in samples]
    bias = aggregate_bias(
        matches, horizon_bars=24, min_sample_count=20,
        neutral_band_pct=0.05,
    )
    assert bias.bullish_count == 0
    assert bias.bearish_count == 0
    assert bias.neutral_count == 20
    assert bias.action == "HOLD"


def test_find_similar_drops_below_min_score():
    """The near-identical shape passes the floor; the inverted one doesn't."""
    target_vec = np.linspace(-1, 1, 60)
    target = _signature(target_vec)
    very_similar = _sample(target_vec.copy(), {24: 0.5})
    very_different = _sample(np.flip(target_vec) * 5, {24: -0.5})
    # structure_weight=0 because our test signatures don't carry structure
    matches = find_similar(
        target, [very_similar, very_different],
        method="cosine", top_k=10, min_score=0.8, structure_weight=0.0,
    )
    assert len(matches) == 1
    assert matches[0].sample is very_similar


def test_waveform_lookup_end_to_end():
    target_vec = np.linspace(-1, 1, 60)
    target = _signature(target_vec)
    library = [
        _sample(target_vec + np.random.default_rng(i).normal(0, 0.05, 60),
                {24: 0.5}) for i in range(25)
    ]
    bias, matches = waveform_lookup(
        target, library, horizon_bars=24, method="cosine",
        top_k=30, min_score=0.5, min_sample_count=20,
    )
    assert bias.action == "BUY"
    assert bias.sample_count >= 20
    assert len(matches) <= 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
