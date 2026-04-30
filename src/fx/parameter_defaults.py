"""Literature-based parameter baseline catalog (PR #19).

This module is intentionally **metadata-only**. None of these constants
flow into `decision_engine.decide_action`, `risk_gate.RiskGate`, or any
indicator computation. The catalog exists to:

1. Make every numerical parameter in the research pipeline traceable to
   one of {literature_standard, practitioner_convention,
   project_initial_value, experimental_hypothesis, not_recommended}.
2. Emit `run_metadata.parameters` so an audit can reconcile a stored
   backtest with the catalog version that was current at run time.
3. Provide the sweep space for PR #20+ rule evaluation. Each entry's
   `sweep_candidates` is the canonical list to walk in optimisation;
   `PARAMETER_SWEEP_SPACE_V1.events.window_profiles` lists named bundles
   that must be evaluated together (e.g. design_v1_current vs
   literature_v1) rather than per-event.

`literature_baseline_v1` is **not** an optimised strategy. It is the
literature/practitioner starting point against which sweeps are scored.

Live (`cmd_trade --broker oanda`) does not consume this catalog. Only
`cmd_backtest_engine` writes a parameters block to RunMetadata, and
`--parameter-profile` toggles only what is recorded — never how trades
are taken. Pinned by
`tests/test_parameter_defaults.py::test_parameter_profile_metadata_only_trade_results_identical`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final


PARAMETER_BASELINE_ID: Final[str] = "literature_baseline_v1"
PARAMETER_BASELINE_VERSION: Final[str] = "2026-04-30"


# ---------------------------------------------------------------------------
# PARAMETER_BASELINE_V1 — literature/practitioner-standard initial values
# ---------------------------------------------------------------------------
#
# Each section carries an `evidence_level` (str OR dict[field -> level]):
#   - literature_standard       — broadly published default
#   - practitioner_convention   — common in practice; less rigorous proof
#   - project_initial_value     — picked for this project, not yet validated
#   - experimental_hypothesis   — operational guess; sweep mandatory
#   - not_recommended           — known weak; record but do not adopt
#
# The shape is deliberately loose `dict[str, Any]` so the JSON dumped
# into run_metadata is human-readable and PR #20+ sweep code can consume
# it without dataclass migration.

PARAMETER_BASELINE_V1: Final[dict[str, Any]] = {
    "baseline_id": PARAMETER_BASELINE_ID,
    "baseline_version": PARAMETER_BASELINE_VERSION,

    "rsi": {
        "period": 14,
        "overbought": 70,
        "oversold": 30,
        "evidence_level": "literature_standard",
        "sweep_candidates": {
            "period": [8, 14, 21],
            "thresholds": [[20, 80], [30, 70], [40, 60]],
        },
    },

    "macd": {
        "fast": 12,
        "slow": 26,
        "signal": 9,
        "evidence_level": "literature_standard",
        "sweep_candidates": [
            {"fast": 8, "slow": 17, "signal": 9},
            {"fast": 12, "slow": 26, "signal": 9},
            {"fast": 5, "slow": 35, "signal": 5},
        ],
    },

    "bollinger": {
        "period": 20,
        "std": 2.0,
        "pctb_low": 0.2,
        "pctb_high": 0.8,
        "evidence_level": "literature_standard",
        "sweep_candidates": [
            {"period": 14, "std": 2.0},
            {"period": 20, "std": 2.0},
            {"period": 20, "std": 2.5},
        ],
    },

    "atr": {
        "period": 14,
        "stop_mult": 1.5,
        "tp_mult": 3.0,
        "evidence_level": "practitioner_convention",
        "sweep_candidates": {
            "period": [10, 14, 20],
            "stop_mult": [1.0, 1.5, 2.0],
            "tp_mult": [2.0, 3.0, 4.0],
        },
    },

    "risk": {
        "risk_per_trade_pct": 1.0,
        "risk_reward_ratio": 2.0,
        "max_hold_bars": 48,
        "evidence_level": {
            "risk_per_trade_pct": "practitioner_convention",
            "risk_reward_ratio": "practitioner_convention",
            "max_hold_bars": "experimental_hypothesis",
        },
        "sweep_candidates": {
            "risk_per_trade_pct": [0.25, 0.5, 1.0, 1.5],
            "risk_reward_ratio": [1.5, 2.0, 2.5],
            "max_hold_bars": [24, 36, 48, 72],
        },
    },

    "sma": {
        "fast": 50,
        "mid": 90,
        "slow": 200,
        "evidence_level": {
            "fast": "practitioner_convention",
            "mid": "project_initial_value",
            "slow": "practitioner_convention",
        },
        "sweep_candidates": [
            {"fast": 30, "mid": 90, "slow": 200},
            {"fast": 50, "mid": 90, "slow": 200},
            {"fast": 50, "mid": 100, "slow": 200},
        ],
    },

    "vix": {
        "buckets": [12, 20, 30],
        "evidence_level": "practitioner_convention",
        "sweep_candidates": [
            [12, 20, 30],
            [15, 20, 25, 30],
        ],
    },

    "dxy": {
        "return_windows_bars": [5, 20],
        "zscore_lookback_bars": 60,
        "evidence_level": "project_initial_value",
        "sweep_candidates": {
            "return_windows_bars": [[3, 12], [5, 20], [8, 24]],
            "zscore_lookback_bars": [20, 40, 60, 120],
        },
    },

    # NOTE: literature_v1 windows. PR #18 active spec
    # (FOMC=6 / ECB=4 / BOE=6 / BOJ=6) is intentionally different and
    # remains the runtime gate. This block is metadata-only; it is
    # NOT wired into risk_gate. PR #20+ event window sweep compares
    # the two profiles head-to-head.
    "events": {
        "NFP": 2,
        "CPI": 2,
        "FOMC": 3,
        "ECB": 3,
        "BOE": 2,
        "BOJ": 4,
        "INTERVENTION": 6,
        "evidence_level": "practitioner_convention",
        "sweep_candidates": {
            "NFP": [1, 2, 3],
            "CPI": [1, 2, 3],
            "FOMC": [2, 3, 4, 6],
            "ECB": [2, 3, 4],
            "BOE": [2, 3, 4],
            "BOJ": [3, 4, 6],
            "INTERVENTION": [3, 6],
        },
    },

    # absolute_similarity_threshold is recorded as `not_recommended`
    # because library/normalization/window-length make a fixed 0–1 cut
    # non-portable. The recommended baseline is distance percentile,
    # but it is NOT wired into the live waveform gate in this PR.
    "waveform": {
        "window_bars": 60,
        "horizon_bars": 24,
        "top_k": 30,
        "similarity_mode": "distance_percentile",
        "high_similarity_pct": 0.10,
        "medium_similarity_pct": 0.20,
        "evidence_level": {
            "dtw_method": "supported_but_task_specific",
            "absolute_similarity_threshold": "not_recommended",
            "distance_percentile": "recommended_baseline",
        },
        "sweep_candidates": {
            "window_bars": [36, 48, 60, 72],
            "horizon_bars": [6, 12, 24],
            "top_k": [10, 20, 30, 50],
            "high_similarity_pct": [0.05, 0.10, 0.15],
            "medium_similarity_pct": [0.15, 0.20, 0.30],
        },
    },

    "signal_voting": {
        "vote_threshold_mode": "fraction_plus_floor",
        "min_vote_fraction": 0.60,
        "min_votes_floor": 3,
        "evidence_level": "project_initial_value",
        "sweep_candidates": {
            "min_vote_fraction": [0.50, 0.60, 0.67],
            "min_votes_floor": [2, 3, 4],
        },
    },
}


# ---------------------------------------------------------------------------
# PARAMETER_SWEEP_SPACE_V1 — head-to-head profiles for PR #20+ sweeps
# ---------------------------------------------------------------------------
#
# Per-section sweep_candidates already live inside PARAMETER_BASELINE_V1.
# This catalog adds NAMED BUNDLES that must be evaluated atomically
# (e.g. event windows are correlated across kinds — a sweep that picks
# NFP=2h × FOMC=6h × BOE=2h is a real combination, not a per-event
# Cartesian explosion). Single source of truth for the literature_v1
# event profile is PARAMETER_BASELINE_V1.events; this catalog mirrors
# the same numbers and is pinned identical by
# `test_literature_baseline_event_windows_match_sweep_profile`.

PARAMETER_SWEEP_SPACE_V1: Final[dict[str, Any]] = {
    "sweep_space_id": "literature_sweep_v1",
    "sweep_space_version": PARAMETER_BASELINE_VERSION,

    "events": {
        # Bundle 1: PR #18 active runtime spec. The matcher in
        # risk_gate.HIGH_IMPACT_WINDOWS_HOURS uses these numbers today.
        "window_profiles": {
            "design_v1_current": {
                "NFP": 2,
                "CPI": 2,
                "FOMC": 6,
                "ECB": 4,
                "BOE": 6,
                "BOJ": 6,
            },
            # Bundle 2: literature/practitioner starting point (PR #19
            # baseline). Mirrors PARAMETER_BASELINE_V1.events.
            "literature_v1": {
                "NFP": 2,
                "CPI": 2,
                "FOMC": 3,
                "ECB": 3,
                "BOE": 2,
                "BOJ": 4,
                "INTERVENTION": 6,
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def baseline_payload_hash(baseline: dict[str, Any] | None = None) -> str:
    """Stable sha256 of the canonical JSON encoding of a baseline dict.

    Two backtest runs that record the same `baseline_payload_hash` were
    audited against the literally same baseline catalog. Sort keys to
    keep the hash invariant under dict insertion order.
    """
    payload = baseline if baseline is not None else PARAMETER_BASELINE_V1
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def literature_event_window_hours() -> dict[str, int]:
    """Return the literature_v1 event window dict, stripped of catalog
    sidecar keys (`evidence_level` / `sweep_candidates`).

    Used by both run_metadata emission and the
    `test_literature_baseline_event_windows_match_sweep_profile`
    invariant.
    """
    section = PARAMETER_BASELINE_V1["events"]
    return {
        k: v for k, v in section.items()
        if k not in ("evidence_level", "sweep_candidates")
    }


KNOWN_PARAMETER_PROFILES: Final[tuple[str, ...]] = (
    PARAMETER_BASELINE_ID,
)
