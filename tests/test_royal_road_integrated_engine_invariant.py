"""Invariant tests for the opt-in royal_road_decision_v2_integrated profile.

These tests pin down what MUST NOT regress when the integrated profile
is added in Phase A. They are deliberately strict so any future change
that would make the integrated profile leak into the default /
live / OANDA / paper paths is caught immediately.

What this file guarantees:
  - The default decision profile is still "current_runtime".
  - The integrated profile name is registered in
    SUPPORTED_DECISION_PROFILES at the END (does not displace any
    existing profile).
  - The existing royal_road_decision_v2 entrypoint is unaffected.
  - Live (OANDA) / paper / broker source files do NOT import the
    integrated module — there is no path for it to influence live
    trading.
  - The Phase A scaffold returns Decision(HOLD) with a clear reason,
    so accidentally selecting the integrated profile cannot silently
    place trades.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.fx.decision_engine import Decision
from src.fx.risk_gate import RiskState
from src.fx.royal_road_decision import (
    SUPPORTED_DECISION_PROFILES,
    validate_decision_profile,
)
from src.fx.royal_road_decision_v2 import (
    PROFILE_NAME_V2,
    decide_royal_road_v2,
)
from src.fx.royal_road_integrated_decision import (
    DEFAULT_INTEGRATED_MODE,
    INTEGRATED_MODE_BALANCED,
    INTEGRATED_MODE_STRICT,
    PROFILE_NAME_V2_INTEGRATED,
    SUPPORTED_INTEGRATED_MODES,
    IntegratedEvidenceAxis,
    RoyalRoadIntegratedDecision,
    decide_royal_road_v2_integrated,
    empty_integrated_decision,
    validate_integrated_mode,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_FX = REPO_ROOT / "src" / "fx"


# ── Profile registration invariants ───────────────────────────────
def test_default_profile_is_current_runtime():
    """First entry of SUPPORTED_DECISION_PROFILES must remain
    "current_runtime" — otherwise CLI defaults / argparse choices
    fall out of sync."""
    assert SUPPORTED_DECISION_PROFILES[0] == "current_runtime"


def test_supported_profiles_keep_existing_three():
    """v1 / v2 / current_runtime must all still be present."""
    for name in (
        "current_runtime",
        "royal_road_decision_v1",
        "royal_road_decision_v2",
    ):
        assert name in SUPPORTED_DECISION_PROFILES, (
            f"existing profile {name!r} disappeared"
        )


def test_integrated_profile_is_registered_at_end():
    """Integrated profile MUST be appended (not inserted before
    existing entries) so the default list order is preserved."""
    assert SUPPORTED_DECISION_PROFILES[-1] == PROFILE_NAME_V2_INTEGRATED
    assert PROFILE_NAME_V2_INTEGRATED == "royal_road_decision_v2_integrated"


def test_validate_decision_profile_accepts_integrated():
    assert (
        validate_decision_profile(PROFILE_NAME_V2_INTEGRATED)
        == PROFILE_NAME_V2_INTEGRATED
    )


def test_validate_decision_profile_rejects_typo():
    with pytest.raises(ValueError):
        validate_decision_profile("royal_road_decision_v2_integrate")  # typo


# ── Mode invariants ───────────────────────────────────────────────
def test_default_integrated_mode_is_balanced():
    assert DEFAULT_INTEGRATED_MODE == INTEGRATED_MODE_BALANCED


def test_supported_integrated_modes_are_two():
    assert set(SUPPORTED_INTEGRATED_MODES) == {
        INTEGRATED_MODE_BALANCED,
        INTEGRATED_MODE_STRICT,
    }


def test_validate_integrated_mode_canonical():
    assert validate_integrated_mode("integrated_balanced") == "integrated_balanced"
    assert validate_integrated_mode("integrated_strict") == "integrated_strict"


def test_validate_integrated_mode_rejects_unknown():
    for bad in ("balanced", "strict", "integrated", "", "INTEGRATED_BALANCED"):
        with pytest.raises(ValueError):
            validate_integrated_mode(bad)


# ── Existing v2 must not regress ──────────────────────────────────
def test_existing_v2_profile_constant_unchanged():
    assert PROFILE_NAME_V2 == "royal_road_decision_v2"


def test_existing_v2_decide_callable_unchanged_signature():
    """decide_royal_road_v2 must still accept the v2 keyword set
    exactly as before. We don't run a full bar — we just import and
    inspect the parameter list."""
    import inspect
    sig = inspect.signature(decide_royal_road_v2)
    expected = {
        "df_window", "technical_confluence", "pattern",
        "higher_timeframe_trend", "risk_reward", "risk_state",
        "atr_value", "last_close", "symbol", "macro_context",
        "df_lower_tf", "lower_tf_interval", "stop_mode",
        "stop_atr_mult", "tp_atr_mult", "base_bar_close_ts",
        "mode", "min_risk_reward",
    }
    assert set(sig.parameters.keys()) == expected, (
        "decide_royal_road_v2 signature changed — Phase A must not "
        "modify the v2 entrypoint."
    )


# ── Live / OANDA / paper isolation ────────────────────────────────
LIVE_PATH_FILES = (
    SRC_FX / "broker.py",
    SRC_FX / "oanda.py",
)


@pytest.mark.parametrize("path", LIVE_PATH_FILES, ids=lambda p: p.name)
def test_live_paths_do_not_reference_integrated_module(path: Path):
    """broker / oanda source files must not contain
    "royal_road_integrated_decision" anywhere — neither as an import
    nor as a string. This guarantees opt-in cannot leak to live."""
    text = path.read_text(encoding="utf-8")
    assert "royal_road_integrated_decision" not in text, (
        f"{path.name} mentions the integrated decision module — this "
        "must remain backtest-only."
    )
    assert "PROFILE_NAME_V2_INTEGRATED" not in text


def test_cli_default_decision_profile_is_current_runtime():
    """CLI argparse default for --decision-profile must remain
    "current_runtime" so callers who don't opt in are unaffected."""
    cli_text = (SRC_FX / "cli.py").read_text(encoding="utf-8")
    # Find the --decision-profile add_argument block and check default=
    m = re.search(
        r'add_argument\(\s*"--decision-profile"[^)]*default="([^"]+)"',
        cli_text,
        re.DOTALL,
    )
    assert m is not None, "could not find --decision-profile in cli.py"
    assert m.group(1) == "current_runtime"


# ── Phase A scaffold contract ─────────────────────────────────────
def _ts() -> pd.Timestamp:
    return pd.Timestamp("2025-01-01 00:00:00", tz="UTC")


def _df() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": [1.0] * 10, "high": [1.1] * 10, "low": [0.9] * 10,
        "close": [1.0] * 10, "volume": [1000] * 10,
    }, index=idx)


def test_phase_a_scaffold_returns_hold():
    decision = decide_royal_road_v2_integrated(
        df_window=_df(),
        technical_confluence={},
        pattern=None,
        higher_timeframe_trend=None,
        risk_reward=None,
        risk_state=RiskState(),
        atr_value=0.01,
        last_close=1.0,
        symbol="USDJPY",
        macro_context=None,
        df_lower_tf=None,
        lower_tf_interval=None,
        base_bar_close_ts=_ts(),
        mode="integrated_balanced",
    )
    assert isinstance(decision, Decision)
    assert decision.action == "HOLD"
    assert decision.advisory.get("profile") == PROFILE_NAME_V2_INTEGRATED
    assert decision.advisory.get("mode") == "integrated_balanced"
    assert decision.advisory.get("scaffold") is True
    integrated = decision.advisory.get("integrated_decision") or {}
    assert integrated.get("action") == "HOLD"
    assert integrated.get("label") == "HOLD_PHASE_A_SCAFFOLD"
    assert integrated.get("schema_version") == "royal_road_integrated_decision_v1"


def test_phase_a_scaffold_rejects_legacy_modes():
    """Passing a legacy v2 mode (e.g. "balanced") must raise — the
    integrated profile only accepts integrated_* modes."""
    with pytest.raises(ValueError):
        decide_royal_road_v2_integrated(
            df_window=_df(),
            technical_confluence={},
            pattern=None,
            higher_timeframe_trend=None,
            risk_reward=None,
            risk_state=RiskState(),
            atr_value=0.01,
            last_close=1.0,
            symbol="USDJPY",
            macro_context=None,
            df_lower_tf=None,
            lower_tf_interval=None,
            base_bar_close_ts=_ts(),
            mode="balanced",  # ← legacy v2 mode, must reject
        )


# ── Dataclass contract ────────────────────────────────────────────
def test_integrated_evidence_axis_validates_side():
    with pytest.raises(ValueError):
        IntegratedEvidenceAxis(
            axis="x", side="LONG", status="PASS",
            strength=0.5, confidence=0.5,
            used_in_decision=True, required=False,
            reason_ja="", source="test",
        )


def test_integrated_evidence_axis_validates_status():
    with pytest.raises(ValueError):
        IntegratedEvidenceAxis(
            axis="x", side="BUY", status="OK",
            strength=0.5, confidence=0.5,
            used_in_decision=True, required=False,
            reason_ja="", source="test",
        )


def test_integrated_evidence_axis_strength_range():
    with pytest.raises(ValueError):
        IntegratedEvidenceAxis(
            axis="x", side="BUY", status="PASS",
            strength=1.5, confidence=0.5,
            used_in_decision=True, required=False,
            reason_ja="", source="test",
        )


def test_integrated_evidence_axis_to_dict_round_trip():
    ax = IntegratedEvidenceAxis(
        axis="wave", side="BUY", status="PASS",
        strength=0.7, confidence=0.8,
        used_in_decision=True, required=True,
        reason_ja="ダブルボトム + WNL 突破済み", source="wave_shape_review",
    )
    d = ax.to_dict()
    assert d["axis"] == "wave"
    assert d["side"] == "BUY"
    assert d["status"] == "PASS"
    assert d["strength"] == 0.7
    assert d["confidence"] == 0.8
    assert d["used_in_decision"] is True
    assert d["required"] is True
    assert d["source"] == "wave_shape_review"


def test_empty_integrated_decision_is_hold_neutral():
    d = empty_integrated_decision(
        mode="integrated_balanced",
        reason_ja="testing",
    )
    assert isinstance(d, RoyalRoadIntegratedDecision)
    assert d.action == "HOLD"
    assert d.side_bias == "NEUTRAL"
    assert d.confidence == 0.0
    assert d.axes == []
    assert d.used_modules == []
    out = d.to_dict()
    assert out["schema_version"] == "royal_road_integrated_decision_v1"
    assert out["action"] == "HOLD"
    assert out["axes"] == []
