"""Tests for the prediction + post-mortem persistence layer."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.fx.storage import Storage


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "fx.db")


def _save_analysis(storage: Storage) -> int:
    return storage.save_analysis(
        symbol="USDJPY=X",
        snapshot={"last_close": 150.0},
        technical_signal="BUY",
        final_action="BUY",
        llm_action="BUY",
        llm_confidence=0.7,
        llm_reason="MACD crossover with RSI confirmation",
    )


def test_save_prediction_round_trip(storage: Storage):
    aid = _save_analysis(storage)
    pid = storage.save_prediction(
        analysis_id=aid,
        symbol="USDJPY=X",
        interval="1h",
        entry_price=150.0,
        action="BUY",
        confidence=0.7,
        reason="confluence",
        expected_direction="UP",
        expected_magnitude_pct=0.5,
        horizon_bars=4,
        invalidation_price=149.0,
    )
    assert pid >= 1

    pending = storage.pending_predictions()
    assert len(pending) == 1
    assert pending[0]["expected_direction"] == "UP"
    assert pending[0]["status"] == "PENDING"


def test_evaluation_updates_status_and_filters_pending(storage: Storage):
    aid = _save_analysis(storage)
    pid = storage.save_prediction(
        analysis_id=aid, symbol="USDJPY=X", interval="1h", entry_price=150.0,
        action="BUY", confidence=0.7, reason="r", expected_direction="UP",
        expected_magnitude_pct=0.5, horizon_bars=4, invalidation_price=149.0,
    )
    storage.update_prediction_evaluation(
        prediction_id=pid,
        status="WRONG",
        actual_direction="DOWN",
        actual_magnitude_pct=-0.4,
        max_favorable_pct=0.1,
        max_adverse_pct=-0.6,
        invalidation_hit=True,
        note="Invalidation hit at bar 2",
    )
    assert storage.pending_predictions() == []
    wrongs = storage.wrong_predictions_without_postmortem()
    assert len(wrongs) == 1
    assert wrongs[0]["status"] == "WRONG"


def test_postmortem_links_to_prediction_and_excludes_from_wrongs(storage: Storage):
    aid = _save_analysis(storage)
    pid = storage.save_prediction(
        analysis_id=aid, symbol="USDJPY=X", interval="1h", entry_price=150.0,
        action="BUY", confidence=0.7, reason="r", expected_direction="UP",
        expected_magnitude_pct=0.5, horizon_bars=4, invalidation_price=149.0,
    )
    storage.update_prediction_evaluation(
        pid, "WRONG", "DOWN", -0.4, 0.1, -0.6, True, "note"
    )
    pm_id = storage.save_postmortem(
        prediction_id=pid,
        root_cause="NEWS_SHOCK",
        narrative="Major news broke 30 minutes after entry",
        proposed_rule="Skip BUYs within 1h of high-impact USD events",
        tags="usd,fomc,news-shock",
    )
    assert pm_id >= 1
    # Already postmortemed → no longer surfaced
    assert storage.wrong_predictions_without_postmortem() == []


def test_relevant_postmortems_filters_by_symbol(storage: Storage):
    # Two symbols, one postmortem each
    for sym in ("USDJPY=X", "EURUSD=X"):
        aid = storage.save_analysis(
            symbol=sym, snapshot={"last_close": 1.0},
            technical_signal="BUY", final_action="BUY",
        )
        pid = storage.save_prediction(
            analysis_id=aid, symbol=sym, interval="1h", entry_price=1.0,
            action="BUY", confidence=0.7, reason="r",
            expected_direction="UP", expected_magnitude_pct=0.5,
            horizon_bars=4, invalidation_price=0.99,
        )
        storage.update_prediction_evaluation(
            pid, "WRONG", "DOWN", -0.4, 0.1, -0.6, True, "note"
        )
        storage.save_postmortem(
            prediction_id=pid, root_cause="TREND_MISREAD",
            narrative=f"misread {sym}", proposed_rule="r", tags="t",
        )
    relevant = storage.relevant_postmortems("USDJPY=X")
    assert len(relevant) == 1
    assert relevant[0]["symbol"] == "USDJPY=X"


def test_lesson_summary_aggregates_root_causes(storage: Storage):
    for cause in ["TREND_MISREAD", "TREND_MISREAD", "NEWS_SHOCK"]:
        aid = _save_analysis(storage)
        pid = storage.save_prediction(
            analysis_id=aid, symbol="USDJPY=X", interval="1h", entry_price=1.0,
            action="BUY", confidence=0.7, reason="r",
            expected_direction="UP", expected_magnitude_pct=0.5,
            horizon_bars=4, invalidation_price=0.99,
        )
        storage.update_prediction_evaluation(
            pid, "WRONG", "DOWN", -0.4, 0.1, -0.6, True, "n"
        )
        storage.save_postmortem(pid, cause, "n", "r", "t")

    summary = {row["root_cause"]: row["n"] for row in storage.lesson_summary()}
    assert summary == {"TREND_MISREAD": 2, "NEWS_SHOCK": 1}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
