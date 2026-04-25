"""Tests for the split-action persistence model (spec §13).

The model is: `action` carries the LLM's advisory decision for backward
compatibility, while `final_decision_action` and `executed_action` carry
the engine-side truth. A BUY-from-Claude that the engine HOLDs must NOT
appear as an executed BUY in the database.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.fx.storage import Storage


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "fx.db")


def _save_analysis(storage: Storage, final: str = "BUY") -> int:
    return storage.save_analysis(
        symbol="USDJPY=X",
        snapshot={"last_close": 150.0},
        technical_signal="BUY",
        final_action=final,
        llm_action="BUY",
        llm_confidence=0.7,
        llm_reason="confluence",
    )


def test_action_columns_default_to_action_when_unspecified(storage: Storage):
    """Old callers (no kwargs) → all three columns mirror `action`."""
    aid = _save_analysis(storage)
    storage.save_prediction(
        analysis_id=aid, symbol="USDJPY=X", interval="1h", entry_price=150.0,
        action="BUY", confidence=0.7, reason="r",
        expected_direction="UP", expected_magnitude_pct=0.5,
        horizon_bars=4, invalidation_price=149.0,
    )
    rows = storage.list_predictions()
    assert rows[0]["action"] == "BUY"
    assert rows[0]["final_decision_action"] == "BUY"
    assert rows[0]["executed_action"] == "BUY"


def test_engine_hold_overrides_llm_buy(storage: Storage):
    """The crucial spec §13 invariant: LLM=BUY, Engine=HOLD → executed=HOLD."""
    aid = _save_analysis(storage, final="HOLD")
    storage.save_prediction(
        analysis_id=aid, symbol="USDJPY=X", interval="1h", entry_price=150.0,
        action="BUY", confidence=0.99, reason="LLM very confident",
        expected_direction="UP", expected_magnitude_pct=0.5,
        horizon_bars=4, invalidation_price=149.0,
        final_decision_action="HOLD",
        executed_action="HOLD",
        blocked_by="event_high",
        final_reason="FOMC within 6h",
        rule_chain="risk_gate",
    )
    rows = storage.list_predictions()
    assert rows[0]["action"] == "BUY"  # LLM advised BUY
    assert rows[0]["final_decision_action"] == "HOLD"
    assert rows[0]["executed_action"] == "HOLD"
    assert rows[0]["blocked_by"] == "event_high"
    assert rows[0]["final_reason"] == "FOMC within 6h"
    assert rows[0]["rule_chain"] == "risk_gate"


def test_postmortem_persists_context_json(storage: Storage):
    aid = _save_analysis(storage)
    pid = storage.save_prediction(
        analysis_id=aid, symbol="USDJPY=X", interval="1h", entry_price=150.0,
        action="BUY", confidence=0.7, reason="r",
        expected_direction="UP", expected_magnitude_pct=0.5,
        horizon_bars=4, invalidation_price=149.0,
        final_decision_action="BUY", executed_action="BUY",
        risk_reward=2.0, detected_pattern=None,
    )
    storage.update_prediction_evaluation(
        pid, "WRONG", "DOWN", -0.4, 0.1, -0.6, True, "n",
    )
    ctx = {
        "final_decision_action": "BUY",
        "blocked_by": None,
        "rule_chain": ["risk_gate", "approve"],
        "spread_pct": 0.01,
        "risk_reward": 2.0,
    }
    pm_id = storage.save_postmortem(
        prediction_id=pid,
        root_cause="EVENT_VOLATILITY",
        narrative="surprise CPI",
        proposed_rule="extend CPI window",
        tags="cpi",
        loss_category="B",
        is_system_accident=True,
        context=ctx,
    )
    assert pm_id >= 1

    with storage._conn() as conn:
        row = conn.execute(
            "SELECT * FROM postmortems WHERE id = ?", (pm_id,),
        ).fetchone()
    assert row["loss_category"] == "B"
    assert row["is_system_accident"] == 1
    assert row["context_json"] is not None
    parsed = json.loads(row["context_json"])
    assert parsed["risk_reward"] == 2.0
    assert parsed["rule_chain"] == ["risk_gate", "approve"]


def test_postmortem_context_optional(storage: Storage):
    """save_postmortem must still work without the new kwargs."""
    aid = _save_analysis(storage)
    pid = storage.save_prediction(
        analysis_id=aid, symbol="USDJPY=X", interval="1h", entry_price=150.0,
        action="BUY", confidence=0.7, reason="r",
        expected_direction="UP", expected_magnitude_pct=0.5,
        horizon_bars=4, invalidation_price=149.0,
    )
    storage.update_prediction_evaluation(
        pid, "WRONG", "DOWN", -0.4, 0.1, -0.6, True, "n",
    )
    pm_id = storage.save_postmortem(
        prediction_id=pid,
        root_cause="X", narrative="n", proposed_rule=None, tags=None,
    )
    assert pm_id >= 1
    with storage._conn() as conn:
        row = conn.execute(
            "SELECT context_json, loss_category FROM postmortems WHERE id = ?",
            (pm_id,),
        ).fetchone()
    assert row["context_json"] is None
    assert row["loss_category"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
