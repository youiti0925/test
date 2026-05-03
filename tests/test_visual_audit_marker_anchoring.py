"""Royal-road BREAK / RETEST / CONFIRM marker anchoring tests.

These pin the contract that the procedure markers are anchored to
real chart structure (BR / post-BR / last bar) rather than fixed
to the chart's left edge as before, AND that WAIT_BREAKOUT cases
do NOT show a confirmed BREAK marker (the BR pivot is rendered as
"BR?" / projected instead).
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
        "double_top_integrated_sell_demo":
            g._build_double_top_integrated_sell_demo_mobile,
        "wait_breakout_demo": g._build_wait_breakout_demo_mobile,
    }
    out = tmp_path / f"{name}.html"
    builders[name](out_path=out)
    return out.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def built_htmls(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("visual_marker_anchoring_cache")
    return {
        "double_top_integrated_sell_demo":
            _build("double_top_integrated_sell_demo", tmp),
        "wait_breakout_demo":
            _build("wait_breakout_demo", tmp),
    }


def _extract_svg(html: str) -> str:
    m = re.search(
        r"<svg[^>]*data-testid=['\"]main-candle-chart['\"][\s\S]*?</svg>",
        html,
    )
    assert m, "main-candle-chart SVG not found"
    return m.group(0)


def test_break_retest_confirm_are_not_left_fixed_badges(built_htmls):
    html = built_htmls["double_top_integrated_sell_demo"]
    svg = _extract_svg(html)

    for cls in (
        "royal-breakout-marker",
        "royal-retest-marker",
        "royal-confirmation-marker",
    ):
        assert cls in svg, f"{cls!r} missing from main SVG"

    # Anchor metadata pins the new layout — every marker must carry
    # data-anchor-quality + data-anchor-part + data-source.
    assert "data-anchor-quality=" in svg
    assert "data-anchor-part=" in svg
    assert (
        "data-source=\"entry_plan\"" in svg
        or "data-source='entry_plan'" in svg
    )


def test_marker_anchor_quality_values_are_canonical(built_htmls):
    html = built_htmls["double_top_integrated_sell_demo"]
    svg = _extract_svg(html)

    quality_values = set(re.findall(
        r"data-anchor-quality='([A-Z]+)'", svg
    ))
    assert quality_values, "no data-anchor-quality emitted"
    # Allowed values are EXACT and APPROX.
    assert quality_values <= {"EXACT", "APPROX"}


def test_wait_breakout_uses_projected_br_not_confirmed_br(built_htmls):
    html = built_htmls["wait_breakout_demo"]
    svg = _extract_svg(html)

    # The status badge says WAIT BREAKOUT...
    assert "WAIT BREAKOUT" in html
    # ...and there must be NO confirmed BREAK marker.
    assert "royal-breakout-marker" not in svg
    # ...and BR must be rendered as projected (BR?) with the
    # data-projected attribute carrying the reason.
    assert (
        "BR?" in svg
        or "data-projected=\"true\"" in svg
        or "data-projected='true'" in svg
    )
    assert (
        "data-reason='breakout_not_confirmed'" in svg
        or "data-reason=\"breakout_not_confirmed\"" in svg
    )


def test_confirmation_marker_anchored_to_last_bar(built_htmls):
    """The CONFIRM marker is the simplest to anchor (it's at the
    last visible bar) — pin that data-anchor-part='last_bar' is set
    when present."""
    html = built_htmls["double_top_integrated_sell_demo"]
    svg = _extract_svg(html)
    if "royal-confirmation-marker" in svg:
        assert (
            "data-anchor-part='last_bar'" in svg
            or "data-anchor-part=\"last_bar\"" in svg
        )
