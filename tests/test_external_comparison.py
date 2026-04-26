"""Tests for the external-vendor comparator (spec §9)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.fx.external import (
    ComparisonResult,
    ExternalSignal,
    aggregate_comparisons,
    compare_signals,
    read_external_csv,
)
from src.fx.external.comparison import compare_one


def _ext(action: str, *, symbol: str = "USDJPY=X",
         hours_offset: int = 0,
         source: str = "vendorA") -> ExternalSignal:
    return ExternalSignal(
        source=source, symbol=symbol, timeframe="1h",
        timestamp=datetime(2025, 4, 1, 12 + hours_offset, tzinfo=timezone.utc),
        action=action, confidence=0.7,
    )


def _self(action: str, hours_offset: int = 0,
          symbol: str = "USDJPY=X") -> dict:
    return {
        "symbol": symbol,
        "ts": datetime(2025, 4, 1, 12 + hours_offset, tzinfo=timezone.utc),
        "action": action,
        "advisory": {},
    }


def test_compare_one_action_match_perfect_score():
    res = compare_one(_self("BUY"), _ext("BUY"))
    assert res.action_match is True
    assert res.agreement_score == 1.0


def test_compare_one_action_mismatch_records_reason():
    res = compare_one(_self("BUY"), _ext("SELL"))
    assert res.action_match is False
    assert "actions differ" in (res.disagreement_reason or "")
    assert res.agreement_score < 1.0


def test_compare_signals_pairs_within_window():
    selfs = [_self("BUY", hours_offset=0)]
    externals = [
        _ext("BUY", hours_offset=0),
        _ext("SELL", hours_offset=10),  # too far
    ]
    out = compare_signals(selfs, externals,
                          pair_within=timedelta(hours=2))
    assert len(out) == 1
    assert out[0].action_match is True


def test_compare_signals_skips_when_no_self_for_symbol():
    selfs = [_self("BUY", symbol="USDJPY=X")]
    externals = [_ext("BUY", symbol="EURUSD=X")]
    out = compare_signals(selfs, externals,
                          pair_within=timedelta(hours=2))
    assert out == []


def test_aggregate_per_source_action_match_rate():
    selfs = [_self("BUY", hours_offset=i) for i in range(3)]
    externals = [
        _ext("BUY", hours_offset=0, source="A"),
        _ext("SELL", hours_offset=1, source="A"),
        _ext("BUY", hours_offset=2, source="B"),
    ]
    out = compare_signals(selfs, externals, pair_within=timedelta(hours=1))
    agg = aggregate_comparisons(out)
    assert agg["n"] == 3
    by = agg["by_source"]
    assert by["A"]["action_match_rate"] == pytest.approx(0.5, abs=1e-9)
    assert by["B"]["action_match_rate"] == pytest.approx(1.0, abs=1e-9)


def test_directional_disagreement_counters():
    selfs = [
        _self("HOLD", hours_offset=0),
        _self("BUY", hours_offset=1),
    ]
    externals = [
        _ext("BUY", hours_offset=0),   # external_only_directional
        _ext("HOLD", hours_offset=1),  # self_only_directional
    ]
    out = compare_signals(selfs, externals, pair_within=timedelta(hours=1))
    agg = aggregate_comparisons(out)
    src = agg["by_source"]["vendorA"]
    assert src["self_only_directional"] == 1
    assert src["external_only_directional"] == 1


def test_csv_reader_round_trip(tmp_path: Path):
    p = tmp_path / "vendor.csv"
    p.write_text(
        "symbol,timestamp,action,confidence,pattern,sentiment\n"
        "USDJPY=X,2025-04-01 12:00:00,BUY,0.75,double_top,0.4\n"
        "EURUSD=X,2025-04-01 13:00:00,SELL,0.55,,\n",
        encoding="utf-8",
    )
    sigs = read_external_csv(p, source="myvendor")
    assert len(sigs) == 2
    a = sigs[0]
    assert a.symbol == "USDJPY=X"
    assert a.action == "BUY"
    assert a.confidence == 0.75
    assert a.pattern == "double_top"
    assert a.sentiment_score == 0.4
    # Timezone-naive input gets UTC stamped
    assert a.timestamp.tzinfo is not None


def test_csv_reader_skips_bad_rows(tmp_path: Path):
    p = tmp_path / "vendor.csv"
    p.write_text(
        "symbol,timestamp,action\n"
        ",2025-04-01 12:00:00,BUY\n"        # missing symbol
        "USDJPY=X,not-a-date,BUY\n"          # bad timestamp
        "USDJPY=X,2025-04-01 13:00:00,SELL\n",
        encoding="utf-8",
    )
    sigs = read_external_csv(p, source="x")
    assert len(sigs) == 1
    assert sigs[0].action == "SELL"


def test_csv_reader_alias_resolution(tmp_path: Path):
    p = tmp_path / "vendor.csv"
    p.write_text(
        "ticker,time,signal,score\n"
        "USDJPY=X,2025-04-01 12:00:00,buy,0.6\n",
        encoding="utf-8",
    )
    sigs = read_external_csv(p, source="x")
    assert len(sigs) == 1
    assert sigs[0].symbol == "USDJPY=X"
    assert sigs[0].action == "BUY"
    assert sigs[0].confidence == 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
