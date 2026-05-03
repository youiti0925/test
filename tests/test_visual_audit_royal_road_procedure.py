"""Visual-audit tests for the royal-road procedure checklist panel
and chart markers (Phase I follow-up).

Each preview build is expensive (~60–90 s), so HTMLs are shared
across tests via a module-scoped fixture.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_smoke():
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import generate_visual_audit_smoke as g
    return g


def _build(name: str, tmp_path: Path) -> str:
    g = _import_smoke()
    builders = {
        "double_bottom_integrated_buy_demo":
            g._build_double_bottom_integrated_buy_demo_mobile,
        "double_top_integrated_sell_demo":
            g._build_double_top_integrated_sell_demo_mobile,
        "wait_breakout_demo":   g._build_wait_breakout_demo_mobile,
        "wait_retest_demo":     g._build_wait_retest_demo_mobile,
        "wait_event_clear_demo": g._build_wait_event_clear_demo_mobile,
    }
    out = tmp_path / f"{name}.html"
    builders[name](out_path=out)
    return out.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def built_htmls(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("visual_royal_road_procedure_cache")
    return {
        "double_bottom_integrated_buy_demo":
            _build("double_bottom_integrated_buy_demo", tmp),
        "double_top_integrated_sell_demo":
            _build("double_top_integrated_sell_demo", tmp),
        "wait_breakout_demo":     _build("wait_breakout_demo", tmp),
        "wait_retest_demo":       _build("wait_retest_demo", tmp),
        "wait_event_clear_demo":  _build("wait_event_clear_demo", tmp),
    }


# ─────────────────────────────────────────────────────────────────
# Section presence (READY / WAIT / EVENT)
# ─────────────────────────────────────────────────────────────────


def test_procedure_panel_section_heading_present(built_htmls):
    html = built_htmls["double_top_integrated_sell_demo"]
    assert "7. 王道手順チェック" in html
    assert "g5-royal-road-procedure-panel" in html
    assert "royal_road_procedure_checklist_v1" in html


def test_panel_renumbering_with_structural_lines(built_htmls):
    """After the structural-lines panel was added at row 8, the
    candidate panel and manual-line panel both shift down by one."""
    html = built_htmls["double_top_integrated_sell_demo"]
    assert "8. 構造ライン" in html
    assert "9. エントリー候補" in html
    assert "10. 手動線操作" in html


def test_procedure_panel_lists_all_14_step_labels(built_htmls):
    """All 14 royal-road step labels must appear in the panel for the
    READY page so the user can verify the procedure end-to-end."""
    html = built_htmls["double_top_integrated_sell_demo"]
    for label in (
        "環境認識",
        "ダウ理論",
        "重要水平線",
        "トレンドライン",
        "波形認識",
        "Wライン",
        "ブレイク確認",
        "リターンムーブ確認",
        "ローソク足確認",
        "ENTRY 価格",
        "STOP 価格",
        "TP 価格",
        "RR (リスクリワード)",
        "イベント確認",
    ):
        assert label in html, f"missing royal-road step label: {label!r}"


def test_ready_page_summary_says_ready(built_htmls):
    html = built_htmls["double_top_integrated_sell_demo"]
    assert "王道手順のP0条件が揃っているためREADY" in html


def test_visual_audit_confirmation_candle_is_shown_as_required_step(
    built_htmls,
):
    """The confirmation-candle step is P0-required and must surface
    that fact in the rendered procedure panel. Scoped to the panel
    block so we don't accidentally match a "P0" / "ローソク足確認"
    appearing elsewhere in the HTML."""
    html = built_htmls["double_top_integrated_sell_demo"]

    panel_match = re.search(
        r"<div class='g5-row g5-royal-road-procedure-panel'"
        r"[\s\S]*?</table>",
        html,
    )
    assert panel_match, "royal-road procedure panel not found"
    panel = panel_match.group(0)

    # Step row is rendered with data-step-key + data-step-importance.
    cc_row = re.search(
        r"data-step-key='confirmation_candle' "
        r"data-step-status='([A-Z_]+)' "
        r"data-step-importance='([A-Z0-9]+)'",
        panel,
    )
    assert cc_row, "confirmation_candle row not found in panel"
    assert cc_row.group(2) == "P0", (
        f"confirmation_candle must be P0 but rendered as "
        f"{cc_row.group(2)!r}"
    )
    # 9. ローソク足確認 row is visible to the user.
    assert "ローソク足確認" in panel
    # P0 column header ("P") + at least one P0 cell are present.
    assert "P0" in panel


def test_wait_breakout_page_surfaces_wnl_not_broken(built_htmls):
    html = built_htmls["wait_breakout_demo"]
    assert "7. 王道手順チェック" in html
    assert "wnl_not_broken" in html
    assert "ブレイク待ち" in html or "ブレイク" in html


def test_wait_retest_page_surfaces_awaiting_retest(built_htmls):
    html = built_htmls["wait_retest_demo"]
    assert "7. 王道手順チェック" in html
    assert "awaiting_retest_confirmation" in html
    assert "リターンムーブ" in html


def test_wait_event_clear_keeps_technical_setup_visible(built_htmls):
    """WAIT_EVENT_CLEAR must NOT wipe the technical setup. The
    procedure panel should show event_clear=BLOCK while wave / W-line
    / breakout / retest / confirmation / entry / stop / target / rr
    remain visible."""
    html = built_htmls["wait_event_clear_demo"]
    assert "7. 王道手順チェック" in html
    assert "イベント" in html
    # Technical step labels are still present in the panel
    assert "波形認識" in html
    assert "Wライン" in html
    assert "ブレイク確認" in html


# ─────────────────────────────────────────────────────────────────
# Entry-candidates panel surfaces the royal-road P0 verdict
# ─────────────────────────────────────────────────────────────────


def test_entry_candidates_panel_shows_royal_road_p0_verdict_ready(
    built_htmls,
):
    html = built_htmls["double_top_integrated_sell_demo"]
    # data-attribute always present
    assert "data-royal-road-p0-pass" in html
    # the visible badge text
    assert "王道P0" in html
    # for a READY page the badge must say PASS
    assert re.search(
        r"王道P0:</b>\s*(?:<[^>]+>\s*)?PASS", html
    ), "READY page does not show 王道P0: PASS in the candidate panel"


def test_entry_candidates_panel_shows_royal_road_p0_verdict_wait(
    built_htmls,
):
    html = built_htmls["wait_retest_demo"]
    assert "王道P0" in html
    # WAIT_RETEST has retest_confirmed WAIT (P0), so P0 cannot pass.
    # NOT PASS message is rendered.
    assert "NOT PASS" in html


# ─────────────────────────────────────────────────────────────────
# Chart markers — BREAK / RETEST / CONFIRM
# ─────────────────────────────────────────────────────────────────


def _extract_main_svg(html: str) -> str:
    pat = (
        r"<svg[^>]*data-testid=['\"]main-candle-chart['\"]"
        r"[\s\S]*?</svg>"
    )
    m = re.search(pat, html)
    return m.group(0) if m else ""


def test_ready_chart_has_break_and_retest_and_confirm_markers(
    built_htmls,
):
    """The double-top READY page has breakout_confirmed=True,
    retest_confirmed=True, and a confirmation_candle, so all three
    royal-road markers must render in the main SVG."""
    html = built_htmls["double_top_integrated_sell_demo"]
    svg = _extract_main_svg(html)
    assert svg, "main SVG missing"
    assert "royal-breakout-marker" in svg
    assert ">BREAK<" in svg
    assert "royal-retest-marker" in svg
    assert ">RETEST<" in svg
    assert "royal-confirmation-marker" in svg
    assert ">CONFIRM<" in svg


def test_wait_breakout_chart_does_not_show_break_marker(built_htmls):
    """wait_breakout has breakout_confirmed=False so the BREAK marker
    must NOT render."""
    html = built_htmls["wait_breakout_demo"]
    svg = _extract_main_svg(html)
    assert svg
    # BREAK text should not appear inside the main chart SVG. (The
    # wait-breakout token "WAIT BREAKOUT" is rendered as a separate
    # entry-status badge — that's fine.)
    assert "royal-breakout-marker" not in svg


def test_wait_retest_chart_has_break_marker_but_no_retest_or_confirm(
    built_htmls,
):
    """wait_retest has breakout_confirmed=True, retest_confirmed=False,
    confirmation_candle empty → only the BREAK marker is present."""
    html = built_htmls["wait_retest_demo"]
    svg = _extract_main_svg(html)
    assert svg
    assert "royal-breakout-marker" in svg
    assert "royal-retest-marker" not in svg
    assert "royal-confirmation-marker" not in svg


def test_marker_data_attributes_present(built_htmls):
    """The markers expose data-source / data-trigger-line-id /
    data-confirmation-candle so downstream consumers can correlate
    them with the entry_plan."""
    html = built_htmls["double_top_integrated_sell_demo"]
    svg = _extract_main_svg(html)
    assert "data-source='entry_plan'" in svg
    assert "data-trigger-line-id=" in svg
    assert "data-confirmation-candle=" in svg
