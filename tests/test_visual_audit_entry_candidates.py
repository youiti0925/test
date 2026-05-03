"""Phase I-1/I-2 — visual_audit entry-candidate panel pin tests.

These verify the "7. エントリー候補" right-panel section appears in
the rendered mobile preview HTML for the integrated demos, and that
it surfaces both the selected candidate and the candidate inventory.

Each preview build is expensive (~60–90 s), so we share built HTML
across tests via a module-scoped fixture.
"""
from __future__ import annotations

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
        "wait_retest_demo":    g._build_wait_retest_demo_mobile,
        "wait_event_clear_demo": g._build_wait_event_clear_demo_mobile,
    }
    out = tmp_path / f"{name}.html"
    builders[name](out_path=out)
    return out.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def built_htmls(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("visual_entry_candidates_cache")
    return {
        "double_bottom_integrated_buy_demo":
            _build("double_bottom_integrated_buy_demo", tmp),
        "wait_retest_demo":
            _build("wait_retest_demo", tmp),
        "wait_event_clear_demo":
            _build("wait_event_clear_demo", tmp),
    }


# ─────────────────────────────────────────────────────────────────
# Section presence
# ─────────────────────────────────────────────────────────────────

def test_visual_audit_shows_entry_candidates_panel(built_htmls):
    html = built_htmls["double_bottom_integrated_buy_demo"]

    assert "エントリー候補" in html, "section heading missing"
    assert "採用候補" in html, "selected-candidate sub-heading missing"
    assert "neckline_retest" in html, "candidate entry_type missing"
    # Selected-candidate or panel CSS class must appear so a hide-list
    # / styling consumer can target it.
    assert (
        "selected_entry_candidate" in html
        or "entry-candidate-selected" in html
    )


def test_visual_audit_panel_appears_for_wait_retest(built_htmls):
    html = built_htmls["wait_retest_demo"]
    assert "エントリー候補" in html
    # WAIT_RETEST status should be visible somewhere in the HTML —
    # either in the selected-candidate badge or the candidate row.
    assert "WAIT_RETEST" in html


def test_visual_audit_panel_appears_for_wait_event_clear(built_htmls):
    html = built_htmls["wait_event_clear_demo"]
    assert "エントリー候補" in html
    # The risk-gate early-exit must still surface the candidate panel
    # so users see the would-have-been entry plan even when the gate
    # forces a HOLD.
    assert "WAIT_EVENT_CLEAR" in html or "WAIT EVENT" in html


# ─────────────────────────────────────────────────────────────────
# Schema markers in payload dump (panel + payload JSON dump)
# ─────────────────────────────────────────────────────────────────

def test_payload_keys_appear_in_html(built_htmls):
    """The schema marker for the new Phase I payload keys must appear
    in the HTML so downstream consumers / audits can grep them. The
    mobile single-file build does not include a raw payload JSON
    dump, but the panel root carries data-attributes that uniquely
    identify the entry-candidate observation layer."""
    html = built_htmls["double_bottom_integrated_buy_demo"]
    assert "entry_candidate_v1" in html, (
        "entry_candidate schema_version not surfaced in HTML"
    )
    # Either the data-attr (mobile) or the payload-dump key name
    # (full report) must be present; both indicate the candidate
    # layer wired through to rendering.
    assert (
        "data-selected-entry-candidate-type" in html
        or "selected_entry_candidate" in html
    )
    assert (
        "data-entry-candidates-count" in html
        or "entry_candidates" in html
    )


def test_panel_inventory_table_row_has_score(built_htmls):
    """The candidate inventory table must show the score column so the
    user can audit which candidate ranked highest."""
    html = built_htmls["double_bottom_integrated_buy_demo"]
    # Either the panel inventory table header row or the payload dump
    # must include the score / final_score column.
    assert "score" in html or "final_score" in html


def test_panel_renumbering_after_phase_i_followup(built_htmls):
    """Phase I follow-up adds 7. 王道手順チェック; the order is now
    7. 王道手順チェック / 8. エントリー候補 / 9. 手動線操作."""
    html = built_htmls["double_bottom_integrated_buy_demo"]
    assert "7. 王道手順チェック" in html
    assert "8. エントリー候補" in html
    assert "9. 手動線操作" in html
