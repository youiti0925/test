"""Runtime parameter resolution for backtest A/B (PR #21).

Translates a `PARAMETER_BASELINE_V1`-shaped catalog into
`run_engine_backtest` kwargs, and produces the audit metadata that
goes into `run_metadata.parameters` when a profile is applied at
runtime.

This module is BACKTEST-ONLY by design — `cmd_trade` (live / OANDA /
paper) and `cmd_analyze` never call into it. Connecting a profile to
those paths would be a separate PR.

Connected sections (PR #21):
  rsi (period / overbought / oversold)
  macd (fast / slow / signal)
  bollinger (period / std)
  atr (period / stop_mult / tp_mult)
  risk (max_hold_bars)

Not yet connected (skipped sections):
  events            — event window literature_v1 NOT wired into risk_gate
  waveform          — waveform parameters NOT wired into decide_action
  sma               — SMA observation, no decision use
  vix / dxy         — observation only, no rule
  signal_voting     — vote_threshold mode not yet implemented
  long_term         — observation only

Default-runtime values (used to compute `diff_vs_default`):
  rsi_period            = 14
  rsi_overbought        = 70
  rsi_oversold          = 30
  macd_fast             = 12
  macd_slow             = 26
  macd_signal           = 9
  bb_period             = 20
  bb_std                = 2.0
  atr_period            = 14
  stop_atr_mult         = 2.0
  tp_atr_mult           = 3.0
  max_holding_bars      = 48

These mirror `indicators.py` defaults and `run_engine_backtest`
defaults. They are NOT in `PARAMETER_BASELINE_V1` (whose values
constitute literature_baseline_v1, which intentionally differ on
`stop_atr_mult` 1.5).
"""
from __future__ import annotations

from typing import Any, Final


PARAMETER_RUNTIME_POLICY_VERSION: Final[str] = "parameter_runtime_v1"


# Current runtime defaults (single source of truth for diff_vs_default).
# Must match `indicators.py` defaults and `run_engine_backtest` defaults.
DEFAULT_RUNTIME_VALUES: Final[dict[str, Any]] = {
    "rsi_period":         14,
    "rsi_overbought":     70.0,
    "rsi_oversold":       30.0,
    "macd_fast":          12,
    "macd_slow":          26,
    "macd_signal":        9,
    "bb_period":          20,
    "bb_std":             2.0,
    "atr_period":         14,
    "stop_atr_mult":      2.0,
    "tp_atr_mult":        3.0,
    "max_holding_bars":   48,
}


# Sections in PARAMETER_BASELINE_V1 connected at runtime by PR #21.
# Order is canonical for `applied_sections` reporting.
APPLIED_SECTIONS_PR21: Final[tuple[str, ...]] = (
    "rsi", "macd", "bollinger", "atr", "risk",
)

# Sections deliberately NOT connected at runtime by PR #21.
SKIPPED_SECTIONS_PR21: Final[tuple[str, ...]] = (
    "events", "waveform", "sma", "vix", "dxy",
    "signal_voting", "long_term",
)


def _flatten_baseline(baseline: dict[str, Any]) -> dict[str, Any]:
    """Resolve `PARAMETER_BASELINE_V1` into a flat dict of runtime kwargs.

    Returns the values that would land in `run_engine_backtest` and
    indicator builders. Keys match `DEFAULT_RUNTIME_VALUES` so callers
    can compute the diff symmetrically.
    """
    out: dict[str, Any] = {}
    rsi = baseline.get("rsi", {}) or {}
    out["rsi_period"]     = int(rsi.get("period", DEFAULT_RUNTIME_VALUES["rsi_period"]))
    out["rsi_overbought"] = float(rsi.get("overbought", DEFAULT_RUNTIME_VALUES["rsi_overbought"]))
    out["rsi_oversold"]   = float(rsi.get("oversold", DEFAULT_RUNTIME_VALUES["rsi_oversold"]))

    macd = baseline.get("macd", {}) or {}
    out["macd_fast"]      = int(macd.get("fast", DEFAULT_RUNTIME_VALUES["macd_fast"]))
    out["macd_slow"]      = int(macd.get("slow", DEFAULT_RUNTIME_VALUES["macd_slow"]))
    out["macd_signal"]    = int(macd.get("signal", DEFAULT_RUNTIME_VALUES["macd_signal"]))

    bb = baseline.get("bollinger", {}) or {}
    out["bb_period"]      = int(bb.get("period", DEFAULT_RUNTIME_VALUES["bb_period"]))
    out["bb_std"]         = float(bb.get("std", DEFAULT_RUNTIME_VALUES["bb_std"]))

    atr = baseline.get("atr", {}) or {}
    out["atr_period"]     = int(atr.get("period", DEFAULT_RUNTIME_VALUES["atr_period"]))
    out["stop_atr_mult"]  = float(atr.get("stop_mult", DEFAULT_RUNTIME_VALUES["stop_atr_mult"]))
    out["tp_atr_mult"]    = float(atr.get("tp_mult", DEFAULT_RUNTIME_VALUES["tp_atr_mult"]))

    risk = baseline.get("risk", {}) or {}
    out["max_holding_bars"] = int(risk.get(
        "max_hold_bars", DEFAULT_RUNTIME_VALUES["max_holding_bars"]
    ))
    return out


def _diff_vs_default(applied: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-key {default, applied} entries for keys whose applied value
    differs from `DEFAULT_RUNTIME_VALUES`. Keys identical to default
    are omitted (so the dict size = "how many parameters actually
    changed").
    """
    diff: dict[str, dict[str, Any]] = {}
    for k, default_v in DEFAULT_RUNTIME_VALUES.items():
        applied_v = applied.get(k)
        if applied_v is None:
            continue
        if applied_v != default_v:
            diff[k] = {"default": default_v, "applied": applied_v}
    return diff


def resolve_runtime_parameters(
    *,
    profile_baseline: dict[str, Any] | None,
    apply_runtime: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a baseline catalog into runtime kwargs + audit metadata.

    Parameters
    ----------
    profile_baseline:
        Either a `PARAMETER_BASELINE_V1`-shaped dict (when
        `--parameter-profile X` was supplied), or None.
    apply_runtime:
        True when `--apply-parameter-profile` was supplied. False
        keeps the engine on its existing defaults (PR #19 invariant
        preserved — metadata-only).

    Returns
    -------
    (runtime_kwargs, runtime_audit)
      runtime_kwargs is the dict to splat into `run_engine_backtest`.
        Empty dict when `apply_runtime` is False — the engine then
        uses its own kwarg defaults, identical to PR #20 main.
      runtime_audit is the dict folded into
        `run_metadata.parameters` at metadata build time:
          applied_to_runtime, applied_sections, skipped_sections,
          unsupported_fields, applied_values, diff_vs_default,
          parameter_runtime_policy_version.
        When not applied, only audit-relevant flags are emitted
        (applied_to_runtime=False).
    """
    if not apply_runtime:
        return ({}, {
            "applied_to_runtime": False,
            "parameter_runtime_policy_version": PARAMETER_RUNTIME_POLICY_VERSION,
        })

    if profile_baseline is None:
        # CLI must reject this combination; defensive guard.
        raise ValueError(
            "resolve_runtime_parameters: apply_runtime=True requires "
            "a non-None profile_baseline (use --parameter-profile X "
            "alongside --apply-parameter-profile)"
        )

    applied = _flatten_baseline(profile_baseline)
    diff = _diff_vs_default(applied)
    audit = {
        "applied_to_runtime": True,
        "applied_sections": list(APPLIED_SECTIONS_PR21),
        "skipped_sections": list(SKIPPED_SECTIONS_PR21),
        "unsupported_fields": [],  # No unsupported fields at PR #21 level.
        "applied_values": dict(applied),
        "diff_vs_default": dict(diff),
        "parameter_runtime_policy_version": PARAMETER_RUNTIME_POLICY_VERSION,
    }
    return (applied, audit)
