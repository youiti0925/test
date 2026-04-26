"""Prediction evaluation: closing the loop between signal and outcome.

Each TradeSignal commits Claude to a falsifiable claim:
  - direction the market will move
  - magnitude (% change) it will reach within the horizon
  - price level that, if hit, falsifies the setup

This module compares those predictions against actual price action
afterwards and writes the verdict back to storage.

Outcome categories:
  CORRECT      — direction matches AND magnitude ≥ ½ of expected
  PARTIAL      — direction matches but magnitude < ½ of expected
  WRONG        — opposite direction OR invalidation_price was hit
  INCONCLUSIVE — flat outcome that doesn't fit any of the above

We evaluate against *intra-window* extremes too (max_favorable_pct,
max_adverse_pct) so a price that hit our target then reversed is still
classified honestly: the post-mortem can ask "did we hold long enough?"
versus "were we directionally wrong?".
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Verdict:
    status: str  # CORRECT | PARTIAL | WRONG | INCONCLUSIVE | INSUFFICIENT_DATA
    actual_direction: str  # UP | DOWN | FLAT
    actual_magnitude_pct: float  # signed: positive = up
    max_favorable_pct: float  # in the predicted direction
    max_adverse_pct: float  # against the predicted direction
    invalidation_hit: bool
    note: str

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "actual_direction": self.actual_direction,
            "actual_magnitude_pct": round(self.actual_magnitude_pct, 4),
            "max_favorable_pct": round(self.max_favorable_pct, 4),
            "max_adverse_pct": round(self.max_adverse_pct, 4),
            "invalidation_hit": self.invalidation_hit,
            "note": self.note,
        }


def evaluate_prediction(
    prediction: dict,
    bars_after_entry: pd.DataFrame,
    partial_threshold: float = 0.5,
) -> Verdict:
    """Score a single prediction against the bars that came after it.

    `bars_after_entry` must be the OHLCV slice strictly AFTER the entry
    bar, and must contain at least `horizon_bars` rows for a clean
    verdict; if fewer rows are available, returns INSUFFICIENT_DATA.
    """
    horizon = int(prediction["horizon_bars"])
    if len(bars_after_entry) < horizon:
        return Verdict(
            status="INSUFFICIENT_DATA",
            actual_direction="FLAT",
            actual_magnitude_pct=0.0,
            max_favorable_pct=0.0,
            max_adverse_pct=0.0,
            invalidation_hit=False,
            note=f"Need {horizon} bars after entry, have {len(bars_after_entry)}",
        )

    window = bars_after_entry.iloc[:horizon]
    entry = float(prediction["entry_price"])
    expected_dir = prediction["expected_direction"]
    expected_mag = float(prediction["expected_magnitude_pct"])
    invalidation = prediction.get("invalidation_price")

    end_price = float(window["close"].iloc[-1])
    actual_pct = 100.0 * (end_price - entry) / entry  # signed

    high_max = float(window["high"].max())
    low_min = float(window["low"].min())
    up_pct = 100.0 * (high_max - entry) / entry
    down_pct = 100.0 * (low_min - entry) / entry  # negative

    # Has the invalidation level been touched at any point in the window?
    invalidation_hit = False
    if invalidation is not None:
        invalidation = float(invalidation)
        if expected_dir == "UP" and low_min <= invalidation:
            invalidation_hit = True
        elif expected_dir == "DOWN" and high_max >= invalidation:
            invalidation_hit = True
        elif expected_dir == "FLAT":
            # for FLAT, invalidation is a "should not exceed" magnitude
            move_distance_pct = max(up_pct, abs(down_pct))
            if move_distance_pct > expected_mag:
                invalidation_hit = True

    # Actual direction classification (use 0.05% deadband as flat)
    if actual_pct > 0.05:
        actual_dir = "UP"
    elif actual_pct < -0.05:
        actual_dir = "DOWN"
    else:
        actual_dir = "FLAT"

    # max_favorable / max_adverse depend on the predicted direction
    if expected_dir == "UP":
        max_favorable = up_pct
        max_adverse = down_pct
    elif expected_dir == "DOWN":
        max_favorable = -down_pct  # flip sign so favorable is positive
        max_adverse = -up_pct
    else:  # FLAT
        max_favorable = -max(up_pct, abs(down_pct))  # smaller move = better
        max_adverse = max(up_pct, abs(down_pct))

    # Verdict
    if invalidation_hit:
        return Verdict(
            status="WRONG",
            actual_direction=actual_dir,
            actual_magnitude_pct=actual_pct,
            max_favorable_pct=max_favorable,
            max_adverse_pct=max_adverse,
            invalidation_hit=True,
            note="Invalidation level was touched within the horizon.",
        )

    if expected_dir == "FLAT":
        max_excursion = max(up_pct, abs(down_pct))
        if max_excursion <= expected_mag:
            status = "CORRECT"
            note = f"Stayed within ±{expected_mag}% (max excursion {max_excursion:.2f}%)."
        else:
            status = "WRONG"
            note = (
                f"Expected ≤±{expected_mag}% range but excursion was "
                f"{max_excursion:.2f}%."
            )
        return Verdict(
            status=status,
            actual_direction=actual_dir,
            actual_magnitude_pct=actual_pct,
            max_favorable_pct=max_favorable,
            max_adverse_pct=max_adverse,
            invalidation_hit=False,
            note=note,
        )

    # Directional case (UP / DOWN)
    expected_sign = 1 if expected_dir == "UP" else -1
    realized_in_dir_pct = expected_sign * actual_pct  # positive means right way

    if realized_in_dir_pct >= expected_mag:
        status = "CORRECT"
        note = f"Hit target: realized {realized_in_dir_pct:.2f}% vs expected {expected_mag}%."
    elif realized_in_dir_pct >= expected_mag * partial_threshold:
        status = "PARTIAL"
        note = (
            f"Right direction but underpowered: "
            f"realized {realized_in_dir_pct:.2f}% vs expected {expected_mag}%."
        )
    elif realized_in_dir_pct > 0:
        status = "PARTIAL"
        note = (
            f"Direction correct but moved only {realized_in_dir_pct:.2f}% "
            f"of the expected {expected_mag}%."
        )
    elif realized_in_dir_pct == 0:
        status = "INCONCLUSIVE"
        note = "Closed flat at entry."
    else:
        status = "WRONG"
        note = (
            f"Wrong direction: expected {expected_dir} but realized "
            f"{realized_in_dir_pct:.2f}% (negative = against)."
        )

    return Verdict(
        status=status,
        actual_direction=actual_dir,
        actual_magnitude_pct=actual_pct,
        max_favorable_pct=max_favorable,
        max_adverse_pct=max_adverse,
        invalidation_hit=False,
        note=note,
    )


def slice_bars_after(df: pd.DataFrame, after_ts) -> pd.DataFrame:
    """Return the bars strictly after the given timestamp.

    Accepts naive or tz-aware datetimes and ISO strings.
    """
    ts = pd.to_datetime(after_ts)
    # Match the dataframe's tz convention
    if df.index.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    elif df.index.tz is None and ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return df.loc[df.index > ts]
