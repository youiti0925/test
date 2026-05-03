"""Phase F/G preview — SVG-scoped wave-shape visibility regression
tests.

This is the contract for "the user must be able to see the wave shape
on the chart". All assertions extract the chart SVG by data-testid
(`main-candle-chart` / `wave-only-chart`) and grep INSIDE that SVG —
never the full HTML body — so a passing test cannot be faked by
text leaking into the side panel, top card, or audit JSON dump.
"""
from __future__ import annotations

import re
from pathlib import Path


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
        "wait_breakout_demo":  g._build_wait_breakout_demo_mobile,
        "wait_retest_demo":    g._build_wait_retest_demo_mobile,
        "wait_event_clear_demo": g._build_wait_event_clear_demo_mobile,
    }
    out = tmp_path / f"{name}.html"
    builders[name](out_path=out)
    return out.read_text(encoding="utf-8")


def extract_svg_by_testid(html: str, testid: str) -> str:
    """Return the FIRST <svg ... data-testid="<testid>" ...>...</svg>
    block from `html`. Returns "" if not found."""
    pattern = (
        rf'<svg[^>]*data-testid=["\']{re.escape(testid)}["\']'
        rf'[\s\S]*?</svg>'
    )
    m = re.search(pattern, html)
    return m.group(0) if m else ""


# ─────────────────────────────────────────────────────────────────────
# Required tokens — main candle chart
# ─────────────────────────────────────────────────────────────────────


def test_main_svg_carries_data_testid_double_bottom(tmp_path):
    html = _build("double_bottom_integrated_buy_demo", tmp_path)
    assert "data-testid='main-candle-chart'" in html or \
        'data-testid="main-candle-chart"' in html, (
        "main SVG missing data-testid='main-candle-chart'"
    )


def test_double_bottom_main_svg_has_all_required_tokens(tmp_path):
    html = _build("double_bottom_integrated_buy_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    assert svg, "main SVG block not extractable"
    for label in ("DB1", "B1", "B2", "NL", "BR",
                  "WNL", "WSL", "WTP",
                  "ENTRY", "STOP", "TP"):
        assert label in svg, (
            f"BUY-READY main SVG missing required token {label!r}"
        )


def test_double_top_main_svg_has_all_required_tokens(tmp_path):
    html = _build("double_top_integrated_sell_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    assert svg, "main SVG block not extractable"
    for label in ("DT1", "P1", "P2", "NL", "BR",
                  "WNL", "WSL", "WTP",
                  "ENTRY", "STOP", "TP"):
        assert label in svg, (
            f"SELL-READY main SVG missing required token {label!r}"
        )


def test_wait_breakout_main_svg_has_wnl_and_state_text(tmp_path):
    html = _build("wait_breakout_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    assert svg
    assert "WNL" in svg, "WAIT_BREAKOUT main SVG missing WNL"
    assert "WAIT BREAKOUT" in svg, (
        "WAIT_BREAKOUT main SVG missing literal 'WAIT BREAKOUT' token"
    )


def test_wait_retest_main_svg_has_wnl_and_state_text(tmp_path):
    html = _build("wait_retest_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    assert svg
    assert "WNL" in svg, "WAIT_RETEST main SVG missing WNL"
    assert "WAIT RETEST" in svg, (
        "WAIT_RETEST main SVG missing literal 'WAIT RETEST' token"
    )


def test_wait_event_clear_main_svg_has_event_and_state_text(tmp_path):
    html = _build("wait_event_clear_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    assert svg
    assert "EVENT" in svg, "WAIT_EVENT_CLEAR main SVG missing 'EVENT' token"
    assert "WAIT EVENT" in svg, (
        "WAIT_EVENT_CLEAR main SVG missing literal 'WAIT EVENT' token"
    )


# ─────────────────────────────────────────────────────────────────────
# Required tokens — wave-only chart
# ─────────────────────────────────────────────────────────────────────


def test_wave_only_svg_present_for_double_bottom(tmp_path):
    html = _build("double_bottom_integrated_buy_demo", tmp_path)
    svg = extract_svg_by_testid(html, "wave-only-chart")
    assert svg, "wave-only SVG missing"
    for label in ("DB1", "B1", "B2", "NL", "BR",
                  "WNL", "WSL", "WTP"):
        assert label in svg, (
            f"BUY-READY wave-only SVG missing required token {label!r}"
        )


def test_wave_only_svg_present_for_double_top(tmp_path):
    html = _build("double_top_integrated_sell_demo", tmp_path)
    svg = extract_svg_by_testid(html, "wave-only-chart")
    assert svg, "wave-only SVG missing"
    for label in ("DT1", "P1", "P2", "NL", "BR",
                  "WNL", "WSL", "WTP"):
        assert label in svg, (
            f"SELL-READY wave-only SVG missing required token {label!r}"
        )


def test_wave_only_svg_section_wrapper_present(tmp_path):
    """Each wave-only chart must sit inside a <section
    class='wave-only-section'> with an h3 heading so the user sees a
    titled chart, not a bare SVG dropped into the page."""
    html = _build("double_bottom_integrated_buy_demo", tmp_path)
    assert "wave-only-section" in html, (
        "wave-only chart section wrapper missing"
    )
    assert "波形専用チャート" in html, (
        "wave-only chart heading 波形専用チャート missing"
    )


# ─────────────────────────────────────────────────────────────────────
# Right-panel wave-parts inventory
# ─────────────────────────────────────────────────────────────────────


def test_right_panel_wave_parts_section_db_buy(tmp_path):
    html = _build("double_bottom_integrated_buy_demo", tmp_path)
    assert "g5-pattern-parts-panel" in html, (
        "side-panel class g5-pattern-parts-panel missing"
    )
    assert "波形部位" in html, "side-panel heading 波形部位 missing"
    for header in ("<th>B1</th>", "<th>B2</th>", "<th>NL</th>"):
        assert header in html, (
            f"side-panel wave-parts table missing row {header!r}"
        )
    assert "WNL / ENTRY" in html
    assert "WSL / STOP"  in html
    assert "WTP / TP"    in html


def test_right_panel_wave_parts_section_dt_sell(tmp_path):
    html = _build("double_top_integrated_sell_demo", tmp_path)
    assert "g5-pattern-parts-panel" in html
    assert "波形部位" in html
    for header in ("<th>P1</th>", "<th>P2</th>", "<th>NL</th>"):
        assert header in html, (
            f"side-panel wave-parts table missing row {header!r}"
        )


# ─────────────────────────────────────────────────────────────────────
# Chart title — must NOT be WFIB-led
# ─────────────────────────────────────────────────────────────────────


def test_main_svg_title_is_not_wfib_led(tmp_path):
    """Chart title in main SVG must lead with 判断 / 波形候補, NOT
    WFIB. The first 'token' after 判断: should not be a WFIB id."""
    for name in ("double_bottom_integrated_buy_demo",
                 "double_top_integrated_sell_demo"):
        html = _build(name, tmp_path)
        svg = extract_svg_by_testid(html, "main-candle-chart")
        m = re.search(r"<text[^>]*y='18'[^>]*>([^<]+)</text>", svg)
        assert m, f"{name}: chart title not found"
        title = m.group(1)
        assert "波形候補" in title, (
            f"{name}: title missing 波形候補: {title!r}"
        )
        # The 参考線 list must start with one of the W-pattern ids,
        # not WFIB. (Order is enforced by _build_user_facing_title.)
        ref_match = re.search(r"参考線:\s*([^/]+)", title)
        if ref_match:
            first_id = ref_match.group(1).split(",")[0].strip()
            assert not first_id.startswith("WFIB"), (
                f"{name}: 参考線 list still WFIB-led: {first_id!r}"
            )
