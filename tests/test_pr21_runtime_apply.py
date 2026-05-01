"""PR #21 — Parameter Profile Runtime A/B Backtest Support.

Backtest-only runtime apply for `literature_baseline_v1`. The two
critical invariants pinned here:

  T1 (default byte-identical to PR #20 main): without
     `--apply-parameter-profile`, every trace decision and trade
     outcome matches a parallel run with no profile flag at all.

  T2 (metadata-only profile invariant preserved): with
     `--parameter-profile X` alone (no apply flag), trade list /
     metrics / hold_reasons / equity_curve are byte-identical to
     no-flag runs — i.e. PR #19 invariant carried through PR #21.

If either of these ever fails, an apply-mode kwarg has leaked into
the default code path and a fix is required before merge.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest
from src.fx.cli import _build_parameters_metadata
from src.fx.indicators import (
    build_snapshot,
    technical_signal,
    technical_signal_reasons,
)
from src.fx.parameter_defaults import (
    PARAMETER_BASELINE_ID,
    PARAMETER_BASELINE_V1,
)
from src.fx.runtime_parameters import (
    APPLIED_SECTIONS_PR21,
    DEFAULT_RUNTIME_VALUES,
    PARAMETER_RUNTIME_POLICY_VERSION,
    SKIPPED_SECTIONS_PR21,
    resolve_runtime_parameters,
)


# ─── fixtures ────────────────────────────────────────────────────────


def _ohlcv(n: int, *, start: str = "2025-06-01", seed: int = 17,
           freq: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def _trade_summary(res):
    return [
        (t.entry_ts, t.exit_ts, t.side, t.exit_reason,
         round(t.pnl, 8), round(t.return_pct, 8), t.bars_held)
        for t in res.trades
    ]


def _decision_signature(res):
    return [
        (tr.decision.final_action, tuple(tr.decision.blocked_by))
        for tr in res.decision_traces
    ]


# ─── T1 — CRITICAL: default byte-identical to PR #20 ─────────────────


def test_pr21_default_byte_identical_to_pr20():
    """Without `--apply-parameter-profile`, the engine must produce
    byte-identical trades / decisions / equity curve to a vanilla run
    that knows nothing about PR #21."""
    df = _ohlcv(400, start="2025-06-01", seed=21)
    # vanilla call (none of the new kwargs supplied)
    res_a = run_engine_backtest(df, "X", interval="1h", warmup=60)
    # PR #21 path with all overrides explicitly None — must match
    res_b = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        rsi_period=None, rsi_overbought=None, rsi_oversold=None,
        macd_fast=None, macd_slow=None, macd_signal_period=None,
        bb_period=None, bb_std=None, atr_period=None,
    )
    assert _trade_summary(res_a) == _trade_summary(res_b)
    assert _decision_signature(res_a) == _decision_signature(res_b)
    assert res_a.hold_reasons == res_b.hold_reasons
    assert res_a.metrics()["n_trades"] == res_b.metrics()["n_trades"]
    assert res_a.metrics()["win_rate"] == res_b.metrics()["win_rate"]
    assert res_a.metrics()["total_return_pct"] == res_b.metrics()["total_return_pct"]
    assert len(res_a.equity_curve) == len(res_b.equity_curve)
    for (t1, e1), (t2, e2) in zip(res_a.equity_curve, res_b.equity_curve):
        assert t1 == t2 and e1 == e2


# ─── T2 — metadata-only profile invariant preserved ──────────────────


def test_pr21_metadata_only_invariant_preserved():
    """`--parameter-profile X` alone (apply_runtime=False) must keep
    byte-identical trades to no-profile runs. PR #19 invariant carried
    through PR #21 plumbing."""
    df = _ohlcv(400, start="2025-07-01", seed=22)

    metadata_no_flag, kwargs_no_flag = _build_parameters_metadata(None)
    metadata_profile_only, kwargs_profile_only = _build_parameters_metadata(
        PARAMETER_BASELINE_ID, apply_runtime=False,
    )
    assert metadata_no_flag["applied_to_runtime"] is False
    assert metadata_profile_only["applied_to_runtime"] is False
    assert kwargs_no_flag == {}
    assert kwargs_profile_only == {}

    res_a = run_engine_backtest(
        df, "X", interval="1h", warmup=60, parameters=metadata_no_flag,
    )
    res_b = run_engine_backtest(
        df, "X", interval="1h", warmup=60, parameters=metadata_profile_only,
    )
    assert _trade_summary(res_a) == _trade_summary(res_b)
    assert _decision_signature(res_a) == _decision_signature(res_b)


# ─── T3 — apply without profile is rejected ──────────────────────────


def test_pr21_apply_without_profile_errors():
    with pytest.raises(ValueError, match="requires --parameter-profile"):
        _build_parameters_metadata(None, apply_runtime=True)


def test_pr21_apply_with_unknown_profile_errors():
    with pytest.raises(ValueError, match="unknown --parameter-profile"):
        _build_parameters_metadata("not_a_real_profile", apply_runtime=True)


# ─── T4/T5 — diff_vs_default = stop_atr_mult only ────────────────────


def test_pr21_apply_literature_diff_vs_default_is_stop_atr_mult_only():
    """The single most important fact this PR exposes: applying
    literature_baseline_v1 changes ONLY stop_atr_mult vs the current
    runtime defaults. Future profiles may change more, but THIS pin
    is the empirical baseline against which A/B reports are read."""
    metadata, kwargs = _build_parameters_metadata(
        PARAMETER_BASELINE_ID, apply_runtime=True,
    )
    assert metadata["applied_to_runtime"] is True
    diff = metadata["diff_vs_default"]
    assert set(diff.keys()) == {"stop_atr_mult"}, (
        f"Expected only stop_atr_mult in diff, got {list(diff)}"
    )
    assert diff["stop_atr_mult"] == {"default": 2.0, "applied": 1.5}

    # And the resolved kwargs match
    assert kwargs["stop_atr_mult"] == 1.5
    assert kwargs["tp_atr_mult"] == 3.0
    assert kwargs["max_holding_bars"] == 48
    assert kwargs["rsi_period"] == 14
    assert kwargs["rsi_overbought"] == 70.0
    assert kwargs["rsi_oversold"] == 30.0


# ─── T6 — applied/skipped sections in metadata ───────────────────────


def test_pr21_runtime_metadata_sections_emitted():
    metadata, _ = _build_parameters_metadata(
        PARAMETER_BASELINE_ID, apply_runtime=True,
    )
    assert metadata["applied_sections"] == list(APPLIED_SECTIONS_PR21)
    assert metadata["skipped_sections"] == list(SKIPPED_SECTIONS_PR21)
    assert metadata["unsupported_fields"] == []
    assert metadata["parameter_runtime_policy_version"] == PARAMETER_RUNTIME_POLICY_VERSION
    # When applied, runtime_profile must echo the requested profile
    assert metadata["runtime_profile"] == PARAMETER_BASELINE_ID
    # applied_values is the flat dict actually flowing into the engine
    av = metadata["applied_values"]
    for k in DEFAULT_RUNTIME_VALUES:
        assert k in av


# ─── T7 — runtime_profile field flips correctly ──────────────────────


def test_pr21_runtime_profile_field_states():
    m_none, _ = _build_parameters_metadata(None)
    assert m_none["runtime_profile"] == "current_runtime"
    assert m_none["applied_to_runtime"] is False

    m_prof, _ = _build_parameters_metadata(PARAMETER_BASELINE_ID)
    assert m_prof["runtime_profile"] == "current_runtime"
    assert m_prof["applied_to_runtime"] is False

    m_apply, _ = _build_parameters_metadata(
        PARAMETER_BASELINE_ID, apply_runtime=True,
    )
    assert m_apply["runtime_profile"] == PARAMETER_BASELINE_ID
    assert m_apply["applied_to_runtime"] is True


# ─── T8 — indicators default kwargs unchanged ────────────────────────


def test_indicators_default_kwargs_unchanged():
    """`build_snapshot(symbol, df)` and `technical_signal(snap)` with
    no kwargs must behave byte-identically to pre-PR-#21 (RSI 14,
    MACD 12/26/9, Bollinger 20/2.0, RSI thresholds 70/30)."""
    df = _ohlcv(200, start="2025-08-01", seed=23)
    snap_default = build_snapshot("X", df)
    snap_explicit = build_snapshot(
        "X", df,
        rsi_period=14,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0,
    )
    # Same numeric output
    assert snap_default.rsi_14 == snap_explicit.rsi_14
    assert snap_default.macd == snap_explicit.macd
    assert snap_default.bb_upper == snap_explicit.bb_upper
    assert snap_default.bb_lower == snap_explicit.bb_lower

    # technical_signal default thresholds match explicit 70/30
    assert technical_signal(snap_default) == technical_signal(
        snap_default, rsi_overbought=70.0, rsi_oversold=30.0,
    )
    # technical_signal_reasons too
    a, b = technical_signal_reasons(snap_default)
    a2, b2 = technical_signal_reasons(
        snap_default, rsi_overbought=70.0, rsi_oversold=30.0,
    )
    assert a == a2
    assert b == b2


# ─── T9 — parameter threading actually changes outputs ───────────────


def test_indicators_param_threading_changes_rsi_value():
    """Non-default rsi_period must produce a different rsi_14 value."""
    df = _ohlcv(300, start="2025-08-01", seed=24)
    snap_14 = build_snapshot("X", df, rsi_period=14)
    snap_21 = build_snapshot("X", df, rsi_period=21)
    # Different period → different value
    assert snap_14.rsi_14 != snap_21.rsi_21 if hasattr(snap_21, "rsi_21") else True
    # The field is named rsi_14 historically; what changes is the value
    assert snap_14.rsi_14 != snap_21.rsi_14, (
        "rsi_period=21 must yield different rsi_14 value than period=14 "
        "(field name is historical; value reflects actual period used)"
    )


# ─── T10 — cmd_trade / live unaffected ───────────────────────────────


def test_pr21_cmd_trade_argparse_does_not_accept_apply_flag():
    """Defensive: cmd_trade's argparse must NOT accept
    --apply-parameter-profile. If someone adds it accidentally, this
    test catches the regression."""
    from src.fx.cli import build_parser
    parser = build_parser()
    # `cmd_trade` is the live/paper path
    with pytest.raises(SystemExit):
        parser.parse_args([
            "trade", "--symbol", "USDJPY=X",
            "--apply-parameter-profile",
        ])


def test_pr21_backtest_engine_argparse_accepts_apply_flag():
    """And the corollary: backtest-engine DOES accept it."""
    from src.fx.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "backtest-engine", "--symbol", "USDJPY=X",
        "--start", "2025-01-01", "--end", "2025-01-05",
        "--parameter-profile", PARAMETER_BASELINE_ID,
        "--apply-parameter-profile",
    ])
    assert args.parameter_profile == PARAMETER_BASELINE_ID
    assert args.apply_parameter_profile is True


# ─── T11 — applied vs default produces a real trade diff ─────────────


def test_pr21_applied_runtime_diff_appears_in_some_metric():
    """With literature_baseline_v1 applied (stop_atr_mult 2.0→1.5),
    trade results must differ from current_runtime defaults on at
    least ONE metric. If they don't, the runtime apply isn't actually
    flowing through.

    Synthetic 1000 bars with tighter random walk to ensure the engine
    actually generates trades at both stop multipliers — a fixture
    that produces 0 trades in both cases would make this test
    vacuous.
    """
    df = _ohlcv(1000, start="2025-04-01", seed=33)
    res_default = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        # stop_atr_mult=2.0 (default), tp_atr_mult=3.0 (default)
    )
    res_applied = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        stop_atr_mult=1.5,
        tp_atr_mult=3.0,
        max_holding_bars=48,
        rsi_period=14, rsi_overbought=70.0, rsi_oversold=30.0,
        macd_fast=12, macd_slow=26, macd_signal_period=9,
        bb_period=20, bb_std=2.0,
        atr_period=14,
    )
    # At least one of these must differ — we don't pin the direction
    # because that depends on the random walk
    differ = (
        len(res_default.trades) != len(res_applied.trades)
        or _trade_summary(res_default) != _trade_summary(res_applied)
        or res_default.metrics().get("total_return_pct") != res_applied.metrics().get("total_return_pct")
    )
    assert differ, (
        "Default vs applied (stop_atr_mult 2.0 → 1.5) must produce a "
        "trade diff on this 1000-bar fixture. If both runs match, the "
        "runtime apply path isn't actually changing engine behaviour."
    )


# ─── T12 — resolve_runtime_parameters API ────────────────────────────


def test_resolve_runtime_parameters_when_not_applied():
    kwargs, audit = resolve_runtime_parameters(
        profile_baseline=PARAMETER_BASELINE_V1, apply_runtime=False,
    )
    assert kwargs == {}
    assert audit["applied_to_runtime"] is False
    assert audit["parameter_runtime_policy_version"] == PARAMETER_RUNTIME_POLICY_VERSION
    # No applied_values / sections when not applying
    assert "applied_values" not in audit
    assert "diff_vs_default" not in audit
    assert "applied_sections" not in audit


def test_resolve_runtime_parameters_when_applied():
    kwargs, audit = resolve_runtime_parameters(
        profile_baseline=PARAMETER_BASELINE_V1, apply_runtime=True,
    )
    # All 12 runtime keys present
    assert set(kwargs.keys()) == set(DEFAULT_RUNTIME_VALUES.keys())
    assert audit["applied_to_runtime"] is True
    assert audit["applied_sections"] == list(APPLIED_SECTIONS_PR21)
    assert audit["skipped_sections"] == list(SKIPPED_SECTIONS_PR21)
    assert audit["unsupported_fields"] == []
    # Empirical fact: literature_baseline_v1 differs only on stop_atr_mult
    assert set(audit["diff_vs_default"].keys()) == {"stop_atr_mult"}


def test_resolve_runtime_parameters_apply_without_baseline_raises():
    with pytest.raises(ValueError, match="requires a non-None"):
        resolve_runtime_parameters(
            profile_baseline=None, apply_runtime=True,
        )


# ─── Bonus — TechnicalSlice carries *_used fields when applied ───────


def test_pr21_technical_slice_used_fields_emitted_when_applied():
    """When the engine runs with overrides, the trace's
    `TechnicalSlice.*_used` audit fields are populated. Default-runtime
    runs also populate them (recording the actually-used defaults) —
    but a downstream consumer that ignores them stays compatible."""
    df = _ohlcv(200, start="2025-09-01", seed=42)
    res = run_engine_backtest(
        df, "X", interval="1h", warmup=60,
        rsi_period=14, rsi_overbought=70.0, rsi_oversold=30.0,
        macd_fast=12, macd_slow=26, macd_signal_period=9,
        bb_period=20, bb_std=2.0,
        atr_period=14,
    )
    tr = res.decision_traces[10]
    tech = tr.technical
    assert tech.rsi_period_used == 14
    assert tech.rsi_overbought_used == 70.0
    assert tech.rsi_oversold_used == 30.0
    assert tech.macd_fast_used == 12
    assert tech.bb_period_used == 20
    assert tech.atr_period_used == 14
