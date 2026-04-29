"""PR #19 tests: literature-based parameter baseline catalog.

The catalog is **metadata-only**. The single most important invariant
is `test_parameter_profile_metadata_only_trade_results_identical`:
attaching `--parameter-profile literature_baseline_v1` (and therefore a
populated `run_metadata.parameters`) must NOT change the trade list,
metrics, hold_reasons, or equity_curve. Any divergence here is a
design failure.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest
from src.fx.cli import _build_parameters_metadata
from src.fx.parameter_defaults import (
    KNOWN_PARAMETER_PROFILES,
    PARAMETER_BASELINE_ID,
    PARAMETER_BASELINE_V1,
    PARAMETER_BASELINE_VERSION,
    PARAMETER_SWEEP_SPACE_V1,
    baseline_payload_hash,
    literature_event_window_hours,
)


_REQUIRED_BASELINE_SECTIONS = (
    "rsi", "macd", "bollinger", "atr", "risk", "sma",
    "vix", "dxy", "events", "waveform", "signal_voting",
)


# ─── Test 1 — schema integrity ──────────────────────────────────────────────


def test_parameter_baseline_v1_has_required_sections():
    """Every catalog section enumerated in the design must be present.

    A future refactor that drops or renames a section would silently
    break run_metadata.parameters consumers; this pin makes the rename
    a hard test failure that flags the metadata-schema change.
    """
    assert PARAMETER_BASELINE_V1["baseline_id"] == PARAMETER_BASELINE_ID
    assert PARAMETER_BASELINE_V1["baseline_version"] == PARAMETER_BASELINE_VERSION
    for section in _REQUIRED_BASELINE_SECTIONS:
        assert section in PARAMETER_BASELINE_V1, (
            f"baseline missing required section: {section!r}"
        )
    # Every section must declare its evidence_level.
    for section in _REQUIRED_BASELINE_SECTIONS:
        assert "evidence_level" in PARAMETER_BASELINE_V1[section], (
            f"section {section!r} missing evidence_level"
        )


# ─── Test 2 — canonical literature-standard values ──────────────────────────


def test_parameter_baseline_v1_canonical_indicator_values():
    """Literature-standard / practitioner-convention defaults are
    explicit values, not derived. Pinning them protects PR #19 from
    later drift away from the documented baseline."""
    assert PARAMETER_BASELINE_V1["rsi"]["period"] == 14
    assert PARAMETER_BASELINE_V1["rsi"]["overbought"] == 70
    assert PARAMETER_BASELINE_V1["rsi"]["oversold"] == 30

    assert PARAMETER_BASELINE_V1["macd"]["fast"] == 12
    assert PARAMETER_BASELINE_V1["macd"]["slow"] == 26
    assert PARAMETER_BASELINE_V1["macd"]["signal"] == 9

    assert PARAMETER_BASELINE_V1["bollinger"]["period"] == 20
    assert PARAMETER_BASELINE_V1["bollinger"]["std"] == 2.0

    assert PARAMETER_BASELINE_V1["atr"]["period"] == 14
    assert PARAMETER_BASELINE_V1["atr"]["stop_mult"] == 1.5
    assert PARAMETER_BASELINE_V1["atr"]["tp_mult"] == 3.0

    assert PARAMETER_BASELINE_V1["sma"]["fast"] == 50
    assert PARAMETER_BASELINE_V1["sma"]["slow"] == 200

    assert PARAMETER_BASELINE_V1["risk"]["risk_per_trade_pct"] == 1.0
    assert PARAMETER_BASELINE_V1["risk"]["risk_reward_ratio"] == 2.0


# ─── Test 3 — literature_v1 event windows ───────────────────────────────────


def test_parameter_baseline_v1_event_windows_literature_values():
    """The catalog's event windows are the literature_v1 numbers from
    Deep Research. They are intentionally DIFFERENT from PR #18's
    runtime spec (FOMC=6/ECB=4/BOE=6/BOJ=6) — the baseline records
    what should be sweep-tested, not what is currently active."""
    events = PARAMETER_BASELINE_V1["events"]
    assert events["NFP"] == 2
    assert events["CPI"] == 2
    assert events["FOMC"] == 3
    assert events["ECB"] == 3
    assert events["BOE"] == 2
    assert events["BOJ"] == 4
    assert events["INTERVENTION"] == 6


# ─── Test 4 — sweep_space schema ────────────────────────────────────────────


def test_parameter_sweep_space_v1_has_required_sections():
    """sweep_space carries named bundles that must be evaluated atomically
    (event windows are correlated across kinds). The bundle catalog must
    expose at least design_v1_current and literature_v1 so PR #20+ can
    A/B them."""
    assert PARAMETER_SWEEP_SPACE_V1["sweep_space_id"] == "literature_sweep_v1"
    assert "events" in PARAMETER_SWEEP_SPACE_V1
    profiles = PARAMETER_SWEEP_SPACE_V1["events"]["window_profiles"]
    assert "design_v1_current" in profiles
    assert "literature_v1" in profiles


# ─── Test 5 — baseline ↔ sweep_profile literature_v1 same numbers ──────────


def test_literature_baseline_event_windows_match_sweep_profile():
    """literature_v1 numbers appear in two places (baseline and
    sweep_space) — chosen for readability over indirection. The two
    must stay literally identical; this pin is the integrity test."""
    baseline_windows = literature_event_window_hours()
    sweep_lit = PARAMETER_SWEEP_SPACE_V1["events"]["window_profiles"][
        "literature_v1"
    ]
    assert baseline_windows == sweep_lit, (
        f"literature_v1 drift: baseline={baseline_windows} vs "
        f"sweep={sweep_lit}"
    )


def test_event_window_profiles_are_distinct():
    """design_v1_current (PR #18 active runtime) and literature_v1
    (PR #19 baseline candidate) are intentionally different — that's
    the entire premise of the PR #20+ event window sweep. If they
    became identical the sweep would degenerate."""
    profiles = PARAMETER_SWEEP_SPACE_V1["events"]["window_profiles"]
    assert profiles["design_v1_current"] != profiles["literature_v1"]


# ─── Test 6 — payload hash stability ────────────────────────────────────────


def test_parameter_baseline_payload_hash_stable():
    """baseline_payload_hash is dict-insertion-order independent
    (sort_keys=True). Two recomputes from the same payload yield the
    same hex. A copy with the same values yields the same hash; an
    edit changes it."""
    h1 = baseline_payload_hash()
    h2 = baseline_payload_hash()
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 64

    # Same content → same hash (insertion order independent).
    same_copy = json.loads(json.dumps(PARAMETER_BASELINE_V1, sort_keys=True))
    assert baseline_payload_hash(same_copy) == h1

    # Changed content → different hash.
    edited = json.loads(json.dumps(PARAMETER_BASELINE_V1))
    edited["rsi"]["period"] = 99
    assert baseline_payload_hash(edited) != h1


# ─── Test 7 — metadata emission without profile ─────────────────────────────


def test_run_metadata_parameters_emitted_without_profile():
    """No --parameter-profile: the catalog is still recorded, but
    `parameter_profile` is None and `applied_to_runtime` is False."""
    block = _build_parameters_metadata(None)
    assert block["parameter_profile"] is None
    assert block["runtime_profile"] == "current_runtime"
    assert block["baseline_id"] == PARAMETER_BASELINE_ID
    assert block["baseline_version"] == PARAMETER_BASELINE_VERSION
    assert block["baseline_payload_hash"] == baseline_payload_hash()
    assert block["baseline_reference"] == PARAMETER_BASELINE_ID
    assert block["baseline_values"]["rsi"]["period"] == 14
    assert block["sweep_space_reference"] == "literature_sweep_v1"
    assert block["applied_to_runtime"] is False


# ─── Test 8 — metadata emission with literature profile ─────────────────────


def test_run_metadata_parameters_emitted_with_literature_profile():
    """--parameter-profile literature_baseline_v1: the profile name is
    recorded verbatim; runtime is still untouched
    (`applied_to_runtime` is False)."""
    block = _build_parameters_metadata(PARAMETER_BASELINE_ID)
    assert block["parameter_profile"] == PARAMETER_BASELINE_ID
    assert block["baseline_id"] == PARAMETER_BASELINE_ID
    assert block["baseline_payload_hash"] == baseline_payload_hash()
    assert block["applied_to_runtime"] is False


# ─── Test 9 — unknown profile is rejected ───────────────────────────────────


def test_unknown_parameter_profile_fails():
    """An unrecognised profile name must raise — the CLI propagates
    the failure as exit code 2 (handled in cmd_backtest_engine)."""
    assert PARAMETER_BASELINE_ID in KNOWN_PARAMETER_PROFILES
    with pytest.raises(ValueError, match="unknown --parameter-profile"):
        _build_parameters_metadata("nonexistent_profile_v999")


# ─── Test 10 — CRITICAL: metadata-only invariant ────────────────────────────


def _ohlcv(n: int, *, start: str = "2025-06-01", seed: int = 42,
           freq: str = "1h") -> pd.DataFrame:
    """Synthetic OHLCV identical in shape to test_backtest_engine_context."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close, "high": close + 0.3, "low": close - 0.3,
            "close": close, "volume": [1000] * n,
        },
        index=idx,
    )


def _trade_summary(res):
    return [
        (t.entry_ts, t.exit_ts, t.side, t.exit_reason,
         round(t.pnl, 8), round(t.return_pct, 8), t.bars_held)
        for t in res.trades
    ]


def test_parameter_profile_metadata_only_trade_results_identical():
    """CRITICAL INVARIANT (test #10): attaching --parameter-profile
    literature_baseline_v1 must NOT change the backtest. Trade list,
    hold_reasons, metrics, AND equity_curve must be byte-identical.

    If this test fails, the catalog has been wired into the runtime
    somewhere — that violates PR #19 scope and is a design failure.
    """
    df = _ohlcv(400, start="2025-06-01", seed=7)

    res_no_profile = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        parameters=_build_parameters_metadata(None),
    )
    res_with_profile = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        parameters=_build_parameters_metadata(PARAMETER_BASELINE_ID),
    )

    assert _trade_summary(res_no_profile) == _trade_summary(res_with_profile), (
        "Trade list diverged when --parameter-profile was attached — "
        "PR #19 metadata-only invariant violated. The catalog must NOT "
        "flow into decide_action / risk_gate."
    )
    assert res_no_profile.hold_reasons == res_with_profile.hold_reasons
    m_no = res_no_profile.metrics()
    m_yes = res_with_profile.metrics()
    assert m_no["n_trades"] == m_yes["n_trades"]
    assert m_no["win_rate"] == m_yes["win_rate"]
    assert m_no["total_return_pct"] == m_yes["total_return_pct"]
    # Equity curve must be identical bar-by-bar.
    assert len(res_no_profile.equity_curve) == len(res_with_profile.equity_curve)
    for (t1, e1), (t2, e2) in zip(
        res_no_profile.equity_curve, res_with_profile.equity_curve,
    ):
        assert t1 == t2 and e1 == e2

    # And of course the parameter block IS recorded — that's the whole
    # point. Profile name differs; baseline payload + applied_to_runtime
    # are stable across both runs.
    p_no = res_no_profile.run_metadata.to_dict()["parameters"]
    p_yes = res_with_profile.run_metadata.to_dict()["parameters"]
    assert p_no["parameter_profile"] is None
    assert p_yes["parameter_profile"] == PARAMETER_BASELINE_ID
    assert p_no["baseline_payload_hash"] == p_yes["baseline_payload_hash"]
    assert p_no["applied_to_runtime"] is False
    assert p_yes["applied_to_runtime"] is False


# ─── Bonus: parameters defaults to None when omitted ────────────────────────


def test_run_metadata_parameters_defaults_to_none_when_omitted():
    """Engine call without `parameters=` keeps `run_metadata.parameters`
    as None — preserves backwards compatibility for callers (live cmd_trade
    constructs RunMetadata directly without the kwarg)."""
    df = _ohlcv(200, start="2025-09-01", seed=11)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    assert res.run_metadata.to_dict()["parameters"] is None
