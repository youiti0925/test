"""Tests for the move-cause attribution module (spec §8 / §17).

Pinned guarantees:
  * Closed taxonomy — every emitted code is in ATTRIBUTION_CODES.
  * No event-only spurious lighting up: a tiny move with a CPI nearby
    still triggers a CPI candidate (event was real), but `notable=False`
    tells the consumer not to put weight on it.
  * Pattern candidate requires BOTH neckline_broken AND direction match.
  * UNKNOWN appears when nothing else clears.
"""
from __future__ import annotations

import pytest

from src.fx.attribution import (
    ATTRIBUTION_CODES,
    attribute_move,
)
from src.fx.patterns import PatternResult, TrendState


def test_unknown_when_no_evidence():
    res = attribute_move(0.05)
    codes = [c.code for c in res.candidates]
    assert codes == ["UNKNOWN"]
    assert res.notable is False


def test_cpi_event_lights_up_cpi_candidate():
    res = attribute_move(
        0.8,
        events_nearby=[{"title": "CPI YoY", "impact": "high"}],
        macro_changes_pct={"us10y": 1.5},
    )
    codes = {c.code for c in res.candidates}
    assert "CPI_SURPRISE" in codes
    cpi = [c for c in res.candidates if c.code == "CPI_SURPRISE"][0]
    # 0.6 base + 0.2 (move>0.5) + 0.2 (10Y>=1bp) = 1.0
    assert cpi.weight == pytest.approx(1.0, abs=1e-6)


def test_fomc_and_yield_move_both_emit():
    res = attribute_move(
        0.6,
        events_nearby=[{"title": "FOMC Statement", "impact": "high"}],
        macro_changes_pct={"us10y": 2.0, "dxy": 0.8},
    )
    codes = {c.code for c in res.candidates}
    assert "FOMC_RISK" in codes
    assert "YIELD_MOVE" in codes
    assert "DXY_MOVE" in codes


def test_pattern_only_when_neckline_broken_and_direction_matches():
    pat_top_unbroken = PatternResult(
        detected_pattern="TRIPLE_TOP_CANDIDATE", neckline=99.0,
        neckline_broken=False, pattern_confidence=0.8,
        trend_state=TrendState.RANGE,
    )
    res = attribute_move(-0.7, pattern=pat_top_unbroken)
    assert all(c.code != "TECHNICAL_PATTERN" for c in res.candidates)

    pat_top_broken = PatternResult(
        detected_pattern="TRIPLE_TOP_CANDIDATE", neckline=99.0,
        neckline_broken=True, pattern_confidence=0.8,
        trend_state=TrendState.DOWNTREND,
    )
    res2 = attribute_move(-0.7, pattern=pat_top_broken)
    assert any(c.code == "TECHNICAL_PATTERN" for c in res2.candidates)


def test_pattern_direction_must_match_top_or_bottom():
    """A top breakdown that coincided with an UP move shouldn't credit the pattern."""
    pat_top_broken = PatternResult(
        detected_pattern="TRIPLE_TOP_CANDIDATE", neckline=99.0,
        neckline_broken=True, pattern_confidence=0.8,
        trend_state=TrendState.RANGE,
    )
    res = attribute_move(+0.8, pattern=pat_top_broken)
    assert all(c.code != "TECHNICAL_PATTERN" for c in res.candidates)


def test_sentiment_spike_emitted_above_thresholds():
    res = attribute_move(
        0.6,
        sentiment_snapshot={
            "sentiment_velocity": 0.7,
            "mention_count_24h": 350,
            "notable_posts": [{"text": "fomc panic"}],
        },
    )
    assert any(c.code == "SENTIMENT_SPIKE" for c in res.candidates)


def test_news_spike_emits_only_above_threshold():
    res_low = attribute_move(0.6, news_volume_z=0.5)
    assert all(c.code != "NEWS_SPIKE" for c in res_low.candidates)
    res_high = attribute_move(0.6, news_volume_z=2.5)
    assert any(c.code == "NEWS_SPIKE" for c in res_high.candidates)


def test_spread_spike_requires_baseline_relative_widening():
    res = attribute_move(0.3, spread_pct=0.15, spread_baseline_pct=0.02)
    assert any(c.code == "SPREAD_SPIKE" for c in res.candidates)


def test_notable_flag_off_for_small_moves():
    res = attribute_move(
        0.1,
        events_nearby=[{"title": "FOMC", "impact": "high"}],
        notable_threshold_pct=0.5,
    )
    assert res.notable is False
    # FOMC candidate is still emitted because event was real
    assert any(c.code == "FOMC_RISK" for c in res.candidates)


def test_every_emitted_code_belongs_to_taxonomy():
    res = attribute_move(
        1.0,
        events_nearby=[{"title": "BOJ", "impact": "high"}],
        macro_changes_pct={"us10y": 3.0, "dxy": 1.2, "vix": 8.0,
                           "sp500": -1.5},
        sentiment_snapshot={"sentiment_velocity": 0.8,
                            "mention_count_24h": 400,
                            "notable_posts": []},
        news_volume_z=2.0,
    )
    for c in res.candidates:
        assert c.code in ATTRIBUTION_CODES


def test_top_returns_sorted_subset():
    res = attribute_move(
        0.8,
        events_nearby=[{"title": "FOMC", "impact": "high"}],
        macro_changes_pct={"us10y": 2.0, "dxy": 1.0},
    )
    top3 = res.top(3)
    assert len(top3) <= 3
    weights = [c.weight for c in top3]
    assert weights == sorted(weights, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
