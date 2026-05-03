"""Phase G follow-up — tests for the new mobile UI pieces:

  - top decision card (state / 結論 / ENTRY / STOP / TP / RR / 次に待つ条件)
  - chart label trimming (only WNL / STOP / TP visible)
  - auto-line list with [非表示] toggle in the right info panel
  - localStorage limitation notice in the manual-line section
  - unconnected-fundamentals block (DXY / 米金利 / ニュース / 経済カレンダー)
  - pattern_level_derivation fallback flags
    (fallback_used / fallback_reason / level_source) and the matching
    UI caution band rendered in the top decision card

These tests are deliberately structural — they do not assert exact
pixel positions or whitespace — so they are robust to non-meaningful
template tweaks while still failing if the user-required signals
disappear.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.fx.pattern_level_derivation import derive_pattern_levels
from src.fx.visual_audit import (
    _render_g5_auto_line_list_html,
    _render_g5_unconnected_fundamentals_html,
    _render_top_decision_card_html,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# Top decision card
# ─────────────────────────────────────────────────────────────────────


def _entry_plan_ready() -> dict:
    return {
        "side": "BUY",
        "entry_status": "READY",
        "entry_price": 1.10500,
        "stop_price": 1.10000,
        "target_price": 1.11500,
        "rr": 2.5,
        "trigger_line_id": "WNL1",
        "trigger_line_price": 1.10500,
        "what_to_wait_for_ja": "条件は揃っています。",
        "reason_ja": "READY: 上昇 confirmation 済み。",
    }


def _v2(action: str = "BUY") -> dict:
    return {
        "action": action,
        "entry_plan": _entry_plan_ready(),
        "pattern_levels": {
            "side": "BUY", "stop_price": 1.10000, "target_price": 1.11500,
            "trigger_line_price": 1.10500,
            "rr_at_extended_target": 2.5,
            "fallback_used": False, "fallback_reason": None,
            "level_source": "wave_derived_lines",
        },
    }


def test_top_card_shows_state_conclusion_entry_stop_tp_rr_next():
    html = _render_top_decision_card_html(
        v2=_v2("BUY"),
        entry_plan=_entry_plan_ready(),
        pattern_levels=_v2("BUY")["pattern_levels"],
        fundamental_sidebar={},
    )
    # 状態 + entry_status pill
    assert "状態:" in html
    assert "READY" in html
    # 結論 — READY のときは「入れる」
    assert "結論:" in html
    assert "入れる" in html
    # ENTRY / STOP / TP / RR labels and values
    assert "ENTRY:" in html and "1.10500" in html
    assert "STOP:" in html and "1.10000" in html
    assert "TP:" in html and "1.11500" in html
    assert "RR:" in html and "2.50" in html
    # Next-wait line
    assert "次に待つ条件:" in html
    # Card colour follows BUY
    assert "gx-card-buy" in html


def test_top_card_wait_breakout_says_blockaware_japanese():
    plan = _entry_plan_ready()
    plan.update({"entry_status": "WAIT_BREAKOUT", "entry_price": None,
                 "what_to_wait_for_ja": "WNL を上抜けるまで待機。"})
    v2 = _v2("HOLD")
    v2["entry_plan"] = plan
    html = _render_top_decision_card_html(
        v2=v2, entry_plan=plan,
        pattern_levels=v2["pattern_levels"], fundamental_sidebar={},
    )
    assert "WAIT_BREAKOUT" in html
    # Waiting flavour text
    assert "ブレイク待ち" in html or "まだ待つ" in html
    assert "gx-card-hold" in html


def test_top_card_wait_event_clear_says_event_wait():
    plan = _entry_plan_ready()
    plan.update({"entry_status": "WAIT_EVENT_CLEAR", "entry_price": None,
                 "what_to_wait_for_ja": "FOMC 通過後に再評価。"})
    v2 = _v2("HOLD")
    v2["entry_plan"] = plan
    html = _render_top_decision_card_html(
        v2=v2, entry_plan=plan,
        pattern_levels=v2["pattern_levels"], fundamental_sidebar={},
    )
    assert "WAIT_EVENT_CLEAR" in html
    assert "イベント待ち" in html


def test_top_card_emits_fallback_warning_when_pattern_levels_used_skeleton():
    plan = _entry_plan_ready()
    pl = {
        "side": "BUY", "stop_price": 1.10000, "target_price": 1.11500,
        "trigger_line_price": 1.10500, "rr_at_extended_target": 2.5,
        "fallback_used": True,
        "fallback_reason": "wave_derived_lines_missing_or_wrong_side",
        "level_source": "skeleton_fallback",
    }
    v2 = _v2("BUY")
    v2["pattern_levels"] = pl
    html = _render_top_decision_card_html(
        v2=v2, entry_plan=plan, pattern_levels=pl, fundamental_sidebar={},
    )
    assert "gx-fallback-warn" in html
    assert "skeleton" in html
    assert "wave_derived_lines_missing_or_wrong_side" in html
    assert "STOP / TP" in html


# ─────────────────────────────────────────────────────────────────────
# Auto-line hide list (right info panel)
# ─────────────────────────────────────────────────────────────────────


def _auto_lines() -> list[dict]:
    return [
        {"id": "WNL1", "kind": "neckline", "role": "entry_confirmation_line",
         "price": 1.10500, "used_in_decision": False},
        {"id": "WSL1", "kind": "pattern_invalidation",
         "role": "stop_candidate", "price": 1.10000,
         "used_in_decision": False},
        {"id": "WTP1", "kind": "pattern_target",
         "role": "target_candidate", "price": 1.11500,
         "used_in_decision": False},
        {"id": "WBR",  "kind": "pattern_breakout",
         "role": "breakout_line", "price": 1.10500,
         "used_in_decision": False},
    ]


def test_auto_line_list_emits_one_button_per_line():
    html = _render_g5_auto_line_list_html(
        wave_derived_lines=_auto_lines(), section_id="case-1",
    )
    # Header
    assert "6. 自動線一覧" in html
    # All 4 ids appear
    for lid in ("WNL1", "WSL1", "WTP1", "WBR"):
        assert lid in html
    # 4 [非表示] buttons (one per line) carrying data-line-id
    for lid in ("WNL1", "WSL1", "WTP1", "WBR"):
        assert (
            f"data-line-id='{lid}'" in html
            and "[非表示]" in html
        )
    assert html.count("[非表示]") == 4
    assert "gx-line-toggle" in html


def test_auto_line_list_handles_empty_input():
    html = _render_g5_auto_line_list_html(
        wave_derived_lines=[], section_id="case-1",
    )
    assert "6. 自動線一覧" in html
    assert "(自動線なし)" in html


# ─────────────────────────────────────────────────────────────────────
# Unconnected fundamentals (red/yellow) block
# ─────────────────────────────────────────────────────────────────────


def test_unconnected_fund_block_lists_all_4_slots():
    html = _render_g5_unconnected_fundamentals_html(
        fundamental_sidebar={
            "macro_drivers": [],
            "missing_data": [
                {"name": "dxy", "label_ja": "DXY", "status": "未接続"},
                {"name": "us10y_yield", "label_ja": "米金利",
                 "status": "未接続"},
                {"name": "news_calendar", "label_ja": "ニュース",
                 "status": "未接続"},
                {"name": "event_calendar", "label_ja": "経済カレンダー",
                 "status": "未接続"},
            ],
        },
    )
    assert "DXY" in html
    assert "米金利" in html
    assert "ニュース" in html
    assert "経済カレンダー" in html
    # All 4 marked missing
    assert html.count("gx-fund-missing") == 4


def test_unconnected_fund_block_marks_present_drivers_connected():
    html = _render_g5_unconnected_fundamentals_html(
        fundamental_sidebar={
            "macro_drivers": [
                {"name": "dxy_alignment", "label_ja": "DXY",
                 "value": "BUY"},
                {"name": "yield_alignment", "label_ja": "米10年金利",
                 "value": "NEUTRAL"},
            ],
            "missing_data": [
                {"name": "news_calendar", "label_ja": "ニュース",
                 "status": "未接続"},
                {"name": "event_calendar", "label_ja": "経済カレンダー",
                 "status": "未接続"},
            ],
        },
    )
    assert "gx-fund-connected" in html
    # Two connected (DXY + 米金利) + two missing (ニュース / 経済カレンダー)
    assert html.count("gx-fund-connected") == 2
    assert html.count("gx-fund-missing") == 2


def test_unconnected_fund_block_treats_no_feed_as_missing():
    """When neither macro_drivers nor missing_data lists the slot, the
    UI must still flag it as 未接続 — silence is never 'connected'."""
    html = _render_g5_unconnected_fundamentals_html(
        fundamental_sidebar={"macro_drivers": [], "missing_data": []},
    )
    # Every one of the 4 named slots must be flagged
    for label in ("DXY", "ニュース", "経済カレンダー"):
        assert label in html
    # All 4 are missing because no driver has a value
    assert html.count("gx-fund-missing") == 4
    assert "gx-fund-connected" not in html


# ─────────────────────────────────────────────────────────────────────
# pattern_level_derivation fallback flags
# ─────────────────────────────────────────────────────────────────────


def _skeleton_db_pre_breakout() -> dict:
    """Skeleton with B1/NL/B2 pivots; no W lines feeding stop/target so
    the derivation must fall back to skeleton-derived levels."""
    return {
        "schema_version": "wave_skeleton_v1",
        "scale": "medium",
        "bars_used": 300,
        "pivots": [
            {"index": 50,  "kind": "L", "price": 1.0900,
             "ts": "2025-01-01T00:00:00+00:00",
             "source": "test", "strength": 1.0},
            {"index": 130, "kind": "H", "price": 1.1000,
             "ts": "2025-01-01T05:00:00+00:00",
             "source": "test", "strength": 1.0},
            {"index": 210, "kind": "L", "price": 1.0910,
             "ts": "2025-01-01T10:00:00+00:00",
             "source": "test", "strength": 1.0},
        ],
        "atr_value": 0.005,
        "trend_hint": "UP",
    }


def test_fallback_used_emitted_when_skeleton_supplies_levels():
    pl = derive_pattern_levels(
        wave_shape_review={
            "best_pattern": {
                "kind": "double_bottom",
                "side_bias": "BUY",
                "status": "forming",
                "matched_parts": {},
                "scale": "medium",
            },
        },
        wave_derived_lines=[],   # ← no W lines → skeleton must fallback
        skeleton=_skeleton_db_pre_breakout(),
        last_close=1.0950,
    )
    assert pl["available"] is True
    assert pl["fallback_used"] is True
    assert pl["fallback_reason"] == "wave_derived_lines_missing"
    assert pl["level_source"] == "skeleton_fallback"
    # And the actual numeric levels still come through
    assert pl["stop_price"] is not None
    assert pl["target_price"] is not None


def test_fallback_not_used_when_w_lines_supply_valid_levels():
    pl = derive_pattern_levels(
        wave_shape_review={
            "best_pattern": {
                "kind": "double_bottom",
                "side_bias": "BUY",
                "status": "neckline_broken",
                "matched_parts": {},
                "scale": "medium",
            },
        },
        wave_derived_lines=[
            {"id": "WNL1", "kind": "neckline",
             "role": "entry_confirmation_line",
             "price": 1.1000, "used_in_decision": False},
            {"id": "WSL1", "kind": "pattern_invalidation",
             "role": "stop_candidate",
             "price": 1.0890, "used_in_decision": False},
            {"id": "WTP1", "kind": "pattern_target",
             "role": "target_candidate",
             "price": 1.1100, "used_in_decision": False},
        ],
        skeleton=_skeleton_db_pre_breakout(),
        last_close=1.1050,
    )
    assert pl["available"] is True
    assert pl["fallback_used"] is False
    assert pl["fallback_reason"] is None
    assert pl["level_source"] == "wave_derived_lines"


def test_fallback_reason_distinguishes_missing_vs_wrongside():
    """W lines exist but on the wrong side of the trigger → reason is
    'wave_derived_lines_missing_or_wrong_side' (not just 'missing')."""
    pl = derive_pattern_levels(
        wave_shape_review={
            "best_pattern": {
                "kind": "double_bottom",
                "side_bias": "BUY",
                "status": "neckline_broken",
                "matched_parts": {},
                "scale": "medium",
            },
        },
        wave_derived_lines=[
            # Stop above NL — wrong side for BUY → recompute from skeleton
            {"id": "WSL1", "kind": "pattern_invalidation",
             "role": "stop_candidate",
             "price": 1.1500, "used_in_decision": False},
            # Target below NL — wrong side for BUY
            {"id": "WTP1", "kind": "pattern_target",
             "role": "target_candidate",
             "price": 1.0500, "used_in_decision": False},
        ],
        skeleton=_skeleton_db_pre_breakout(),
        last_close=1.1050,
    )
    assert pl["fallback_used"] is True
    assert pl["fallback_reason"] == "wave_derived_lines_missing_or_wrong_side"


# ─────────────────────────────────────────────────────────────────────
# End-to-end: localStorage notice + chart label trimming via the
# smoke generator's mobile single-file builder.
# ─────────────────────────────────────────────────────────────────────


def _df_realistic(n: int = 150) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(11)
    close = 1.10 + np.cumsum(rng.standard_normal(n) * 0.001)
    return pd.DataFrame({
        "open": close, "high": close + 0.001, "low": close - 0.001,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def test_mobile_html_localstorage_notice_appears_in_manual_section(
    tmp_path: Path,
):
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from generate_visual_audit_smoke import (
        _build_double_bottom_integrated_buy_demo_mobile,
    )
    out = tmp_path / "ls_notice.html"
    _build_double_bottom_integrated_buy_demo_mobile(out_path=out)
    html = out.read_text(encoding="utf-8")
    # The improved notice block
    assert "gx-localstorage-warn" in html, (
        "localStorage limitation banner missing from manual section"
    )
    assert "ブラウザ内" in html
    assert "annotations.json" in html
    # Sanitization: no naked best=/quality=/score=N.NNN/None/NaN in
    # user-visible body (script/style/pre stripped first).
    body = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    body = re.sub(r"<style[^>]*>.*?</style>",  "", body, flags=re.DOTALL)
    body = re.sub(r"<pre[^>]*>.*?</pre>",      "", body, flags=re.DOTALL)
    assert "best=" not in body.lower()
    assert "quality=" not in body.lower()
    assert not re.search(r"score=-?\d+\.\d{3}", body)
    assert ">None<" not in body
    assert not re.search(r"\bNaN\b", body)


def test_mobile_html_top_card_lives_above_chart(tmp_path: Path):
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from generate_visual_audit_smoke import (
        _build_double_bottom_integrated_buy_demo_mobile,
    )
    out = tmp_path / "top_card.html"
    _build_double_bottom_integrated_buy_demo_mobile(out_path=out)
    html = out.read_text(encoding="utf-8")
    # The card class must appear; and the FIRST occurrence of the card
    # in each section must be BEFORE the first chart-row.
    sections = re.findall(
        r"<section class='case-section'[^>]*>.*?</section>",
        html, re.DOTALL,
    )
    assert sections, "no case sections rendered"
    for sec in sections:
        i_card = sec.find("gx-top-decision-card")
        i_chart = sec.find("g5-chart-row")
        assert i_card != -1, "top decision card missing"
        assert i_chart != -1, "chart row missing"
        assert i_card < i_chart, (
            "top decision card must render BEFORE the chart row "
            "so mobile users see ENTRY/STOP/TP/RR first"
        )


def test_mobile_html_chart_only_shows_compact_wnl_stop_tp_labels(
    tmp_path: Path,
):
    """The candle chart must not surface verbose / fibonacci / pivot
    sub-ids (WB1 / WPH / WFIB382 etc.) on the chart itself — they
    belong in the right-panel inventory.

    Allowed wave-derived-line labels (post Phase G follow-up #2):
      WNL / WSL / WTP    — short identifiers for the W lines
      ENTRY / STOP / TP  — actionable role labels (stacked next to W*)
    Anything else (specifically WB1/WB2/WPH/WPL/WP1/WP2/WLS/WH/WRS/
    WFIB*) is a regression."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from generate_visual_audit_smoke import (
        _build_double_bottom_integrated_buy_demo_mobile,
    )
    out = tmp_path / "chart_labels.html"
    _build_double_bottom_integrated_buy_demo_mobile(out_path=out)
    html = out.read_text(encoding="utf-8")
    visible_labels = re.findall(
        r"<text class='wave-derived-line-label'[^>]*>([^<]+)</text>", html,
    )
    allowed = {"WNL", "WSL", "WTP", "ENTRY", "STOP", "TP"}
    bad = [l for l in visible_labels if l not in allowed]
    assert not bad, (
        f"Chart leaked verbose wave-line labels (mobile-policy violation): "
        f"first={bad[:6]}  total={len(bad)}"
    )
