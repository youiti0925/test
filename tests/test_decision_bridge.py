"""Tests for src/fx/decision_bridge.py and the HTML render helper.

Pinned invariants:
  - decision_bridge_v1 is in the audit payload as a top-level key
  - final_action mirrors v2.action
  - wave_shape_review is classified AUDIT_ONLY
  - fibonacci_context_review is classified AUDIT_ONLY
  - masterclass_panels is classified AUDIT_ONLY
  - symbol_macro_briefing (when data_available=False) is classified
    NOT_CONNECTED
  - position_sizing is always classified NOT_CONNECTED (until wired)
  - HOLD case message says 「入れそうに見えるかもしれません」
  - BUY / SELL case message says 「波形・フィボ・Masterclassは
    直接使っていません」
  - Mobile HTML embeds a "★ この判断の読み方" header
  - Mobile HTML labels USED / 表示のみ / 未接続 entries
  - default current_runtime returns no decision_bridge (because
    no v2 audit is built — covered by existing test)
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.fx.decision_bridge import (
    AUDIT_ONLY,
    NOT_CONNECTED,
    PARTIAL,
    SCHEMA_VERSION,
    STATUS_LABEL_JA,
    USED,
    build_decision_bridge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _v2(action: str = "HOLD", *, with_stop: bool = True,
        with_block_reasons: bool = True) -> dict:
    out = {
        "action": action,
        "profile": "royal_road_decision_v2",
        "macro_alignment": {
            "macro_score": 0.0, "currency_bias": "NEUTRAL",
        },
        "support_resistance_v2": {
            "near_strong_support": True,
            "selected_level_zones_top5": [{"kind": "support"}],
        },
    }
    if with_stop:
        out["structure_stop_plan"] = {
            "stop_price": 1.09, "structure_stop_price": 1.09,
            "atr_stop_price": 1.09, "take_profit_price": 1.13,
            "rr_realized": 2.5, "chosen_mode": "structure",
            "outcome": "structure",
        }
    if with_block_reasons:
        out["block_reasons"] = ["insufficient_buy_evidence"]
    return out


def _payload(action: str = "HOLD", **kw) -> dict:
    return {
        "royal_road_decision_v2": _v2(action),
        "wave_shape_review": kw.get("wave_shape_review", {
            "best_pattern": {"kind": "double_bottom"},
        }),
        "wave_derived_lines": kw.get("wave_derived_lines", [
            {"id": "WNL1", "kind": "neckline"},
        ]),
        "masterclass_panels": kw.get("masterclass_panels", {
            "available": True,
            "panels": {
                "fibonacci_context_review": {
                    "available": True, "retracement_zone": "50.0-61.8",
                },
                "daily_roadmap_review": {"available": True},
                "symbol_macro_briefing_review": {
                    "available": False,
                    "unavailable_reason": "macro_briefing_data_missing",
                },
                "parent_bar_lower_tf_anatomy": {
                    "available": False,
                    "unavailable_reason": "lower_tf_missing",
                },
            },
        }),
        "entry_summary": kw.get("entry_summary", {
            "entry_price": 1.10, "stop_price": 1.09,
            "take_profit_price": 1.13, "rr": 2.5,
        }),
    }


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------


def test_schema_and_observation_contract():
    b = build_decision_bridge(_payload("HOLD"))
    assert b["schema_version"] == SCHEMA_VERSION
    assert b["observation_only"] is True
    assert b["used_in_decision"] is False
    assert b["available"] is True


def test_status_label_ja_complete():
    assert STATUS_LABEL_JA[USED] == "判断に使用"
    assert STATUS_LABEL_JA[PARTIAL] == "一部使用"
    assert STATUS_LABEL_JA[AUDIT_ONLY] == "表示のみ"
    assert STATUS_LABEL_JA[NOT_CONNECTED] == "未接続"


def test_final_action_mirrors_v2():
    for action in ("BUY", "SELL", "HOLD"):
        b = build_decision_bridge(_payload(action))
        assert b["final_action"] == action


def test_used_includes_royal_road_core():
    b = build_decision_bridge(_payload("BUY"))
    cats = {u["category"] for u in b["used_for_final_decision"]}
    assert "royal_road_v2_core" in cats


def test_used_includes_stop_plan_when_present():
    b = build_decision_bridge(_payload("BUY"))
    cats = {u["category"] for u in b["used_for_final_decision"]}
    assert "stop_plan" in cats


def test_used_includes_block_reasons_in_hold():
    b = build_decision_bridge(_payload("HOLD"))
    bri = next(
        u for u in b["used_for_final_decision"]
        if u["category"] == "block_reasons"
    )
    assert bri["status"] == USED


def test_wave_shape_review_is_audit_only():
    b = build_decision_bridge(_payload("HOLD"))
    cats = {a["category"] for a in b["audit_only_references"]}
    assert "wave_shape_review" in cats


def test_wave_derived_lines_is_audit_only():
    b = build_decision_bridge(_payload("HOLD"))
    cats = {a["category"] for a in b["audit_only_references"]}
    assert "wave_derived_lines" in cats


def test_fibonacci_is_audit_only_when_available():
    b = build_decision_bridge(_payload("HOLD"))
    cats = {a["category"] for a in b["audit_only_references"]}
    assert "fibonacci_context_review" in cats


def test_masterclass_panels_is_audit_only():
    b = build_decision_bridge(_payload("HOLD"))
    cats = {a["category"] for a in b["audit_only_references"]}
    assert "masterclass_panels" in cats


def test_symbol_macro_briefing_unavailable_is_not_connected():
    b = build_decision_bridge(_payload("HOLD"))
    cats = {n["category"] for n in b["unconnected_or_missing"]}
    assert "symbol_macro_briefing" in cats


def test_position_sizing_is_always_not_connected():
    b = build_decision_bridge(_payload("HOLD"))
    cats = {n["category"] for n in b["unconnected_or_missing"]}
    assert "position_sizing" in cats


def test_news_calendar_is_not_connected():
    b = build_decision_bridge(_payload("HOLD"))
    cats = {n["category"] for n in b["unconnected_or_missing"]}
    assert "news_calendar" in cats


def test_macro_zero_is_not_connected():
    b = build_decision_bridge(_payload("HOLD"))
    cats = {n["category"] for n in b["unconnected_or_missing"]}
    assert "macro_real_data" in cats


def test_lower_tf_anatomy_unavailable_is_not_connected():
    b = build_decision_bridge(_payload("HOLD"))
    cats = {n["category"] for n in b["unconnected_or_missing"]}
    assert "lower_tf_anatomy" in cats


def test_hold_case_message_contains_iresou():
    b = build_decision_bridge(_payload("HOLD"))
    assert "入れそうに見える" in b["action_message_ja"]


def test_buy_case_message_warns_about_audit_only():
    b = build_decision_bridge(_payload("BUY"))
    msg = b["action_message_ja"]
    assert "波形" in msg
    assert "フィボ" in msg
    assert "Masterclass" in msg


def test_sell_case_message_warns_about_audit_only():
    b = build_decision_bridge(_payload("SELL"))
    msg = b["action_message_ja"]
    assert "SELL" in msg
    assert "直接使っていません" in msg


def test_missing_payload_returns_unavailable():
    b = build_decision_bridge(None)
    assert b["available"] is False


def test_missing_v2_returns_unavailable():
    b = build_decision_bridge({"royal_road_decision_v2": None})
    assert b["available"] is False


def test_every_entry_has_status_label_ja():
    b = build_decision_bridge(_payload("HOLD"))
    for collection_key in (
        "used_for_final_decision",
        "audit_only_references",
        "unconnected_or_missing",
    ):
        for e in b[collection_key]:
            assert e.get("status_label_ja"), (
                f"missing status_label_ja in {collection_key}: {e}"
            )


def test_summary_and_plain_answer_present():
    b = build_decision_bridge(_payload("HOLD"))
    assert b["summary_ja"]
    assert "監査表示のみ" in b["summary_ja"]
    assert b["plain_answer_ja"]


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def test_mobile_html_embeds_decision_bridge_section():
    """End-to-end: render a mobile HTML and verify the bridge appears
    at the top of each case section."""
    from src.fx.backtest_engine import run_engine_backtest
    from src.fx.visual_audit import render_visual_audit_mobile_single_file

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(11)
    n = 250
    idx = pd.date_range("2025-06-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    df = pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "m.html"
        render_visual_audit_mobile_single_file(
            traces=res.decision_traces,
            df_by_symbol={"EURUSD=X": df},
            out_path=out, max_cases=2,
        )
        body = out.read_text()
    # ★ この判断の読み方 (decision bridge) ヘッダー
    assert "★ この判断の読み方" in body
    assert "最終判断" in body
    assert "USED" in body or "判断に使用" in body
    assert "表示のみ" in body
    assert "未接続" in body
    # CSS class
    assert "decision-bridge" in body
    assert "bridge-action-" in body  # action color class
    # Tier 1 still present below decision-bridge but slimmed (8 rows)
    assert "今日このケースで見るべきこと" in body
