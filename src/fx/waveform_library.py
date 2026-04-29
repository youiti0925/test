"""Waveform sample library (spec §7.2).

Slides a fixed window over historical OHLCV and stores one sample per
position: the normalised vector + structure codes + the realised
forward returns at fixed horizons. The forward returns are the LABEL
the matcher uses later to predict what a similar shape might do next.

Storage format
--------------
JSON Lines (`.jsonl`) — one sample per line. Easy to append to, easy
to grep, no schema lock-in. Vectors are stored as float lists; for big
libraries (10k+ samples) you'd swap this for parquet, but JSONL is
fine for the research scale we operate at.

Critical rule: NO future leak in the *signature*
------------------------------------------------
The window's signature uses bars [start, end). The forward-return
labels use bars [end, end + horizon). The window's vector and the
labels never overlap, so when a future caller asks "find shapes
similar to TODAY's window", the labels they retrieve are always strictly
post-window data — exactly what we want for prediction research.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .waveform_matcher import (
    NormalizeMethod,
    WaveformSignature,
    compute_signature,
)


@dataclass(frozen=True)
class WaveformSample:
    """One past window with its signature and realised forward returns.

    `forward_returns_pct[h]` is the percent return of the close at
    `end + h` versus the close at `end`. Horizons are bar counts, not
    hours — the timeframe is recorded so callers know what "4 bars"
    means.

    `forward_return_end_ts[h]` is the **wall-clock timestamp of the
    target bar** that produced `forward_returns_pct[h]`. Bar count is
    not the same as wall-clock time when bars contain weekend / holiday
    gaps (FX is closed Sat-Sun): "24 bars after Friday 23:00" lands on
    Monday-or-later, NOT on Saturday 23:00. Look-ahead-safe filters in
    backtest_engine MUST use `forward_return_end_ts[h]` rather than
    `end_ts + h * bar_interval` to decide whether a sample's label was
    realised by `bar_ts`. Older library files (pre-PR-#15) lack this
    field — consumers should treat such samples as ineligible for
    look-ahead-sensitive lookups (safe-side default).

    `max_favorable_pct` / `max_adverse_pct` use the longest horizon
    measured (typically `max(forward_horizons)`).
    """

    symbol: str
    timeframe: str
    start_ts: datetime
    end_ts: datetime
    signature: WaveformSignature
    forward_returns_pct: dict[int, float | None]
    max_favorable_pct: float | None
    max_adverse_pct: float | None
    # PR #15: per-horizon target-bar timestamp. Optional for backward
    # compat with older JSONL files; missing values mean "lookup at
    # this horizon is not safe-to-evaluate" downstream.
    forward_return_end_ts: dict[int, datetime | None] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_ts": self.start_ts.isoformat(),
            "end_ts": self.end_ts.isoformat(),
            "signature": self.signature.to_dict(),
            # JSON keys must be strings; restore via int() on load.
            "forward_returns_pct": {
                str(k): v for k, v in self.forward_returns_pct.items()
            },
            "forward_return_end_ts": {
                str(k): (v.isoformat() if v is not None else None)
                for k, v in self.forward_return_end_ts.items()
            },
            "max_favorable_pct": self.max_favorable_pct,
            "max_adverse_pct": self.max_adverse_pct,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WaveformSample":
        end_ts_map_raw = d.get("forward_return_end_ts") or {}
        end_ts_map: dict[int, datetime | None] = {}
        for k, v in end_ts_map_raw.items():
            if v is None:
                end_ts_map[int(k)] = None
            else:
                end_ts_map[int(k)] = _parse_ts(v)
        return cls(
            symbol=d["symbol"],
            timeframe=d["timeframe"],
            start_ts=_parse_ts(d["start_ts"]),
            end_ts=_parse_ts(d["end_ts"]),
            signature=WaveformSignature.from_dict(d["signature"]),
            forward_returns_pct={
                int(k): v for k, v in (d.get("forward_returns_pct") or {}).items()
            },
            max_favorable_pct=d.get("max_favorable_pct"),
            max_adverse_pct=d.get("max_adverse_pct"),
            forward_return_end_ts=end_ts_map,
        )


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_library(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    window_bars: int = 60,
    step_bars: int = 5,
    forward_horizons: Iterable[int] = (4, 12, 24),
    normalize: NormalizeMethod = "z_score",
) -> list[WaveformSample]:
    """Slide `window_bars` over `df` and emit one sample per `step_bars`.

    Parameters
    ----------
    window_bars:
        Length of each sample's signature. 60 is a sensible default for
        hourly data — about 2.5 days of context.
    step_bars:
        How far to advance between samples. 1 = every bar (heaviest);
        5 = every 5 bars (recommended for long histories).
    forward_horizons:
        Bar offsets at which to record the realised future return.
        Each horizon must fit inside `df`; samples too close to the
        right edge are still emitted but their too-far labels are None.
    normalize:
        Forwarded to `compute_signature`. z_score keeps cross-instrument
        compares level-invariant.

    Returns
    -------
    A list of WaveformSample. Caller decides whether to write them via
    `write_library()` or process in memory.
    """
    if window_bars < 10:
        raise ValueError("window_bars must be ≥ 10 to fit a meaningful pattern")
    if step_bars < 1:
        raise ValueError("step_bars must be ≥ 1")

    horizons = sorted(set(int(h) for h in forward_horizons if int(h) > 0))
    if not horizons:
        raise ValueError("forward_horizons must contain at least one positive int")
    max_h = max(horizons)

    samples: list[WaveformSample] = []
    n = len(df)
    if n < window_bars + 1:
        return samples

    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    idx = df.index

    # Slide so that the window ENDS at position `end` (exclusive). The
    # last allowable end is `n` (the bar after the final close), but
    # then forward returns wouldn't exist. We still emit those and let
    # the labels be None — useful for "what would TODAY's window match
    # against?" calls without polluting downstream stats.
    for end in range(window_bars, n + 1, step_bars):
        start = end - window_bars
        window_df = df.iloc[start:end]
        sig = compute_signature(window_df, method=normalize)

        anchor_close = closes[end - 1]
        forward: dict[int, float | None] = {}
        # PR #15: also record the wall-clock timestamp of each horizon's
        # target bar. `target = end - 1 + h` is a bar-count offset into
        # the SAME df (which already excludes weekend gaps for FX), so
        # idx[target] is the actual close timestamp the label was
        # measured at — NOT `end_ts + h * bar_interval`, which double-
        # counts gaps. Look-ahead-safe filters in backtest must use
        # this timestamp.
        forward_end_ts: dict[int, datetime | None] = {}
        for h in horizons:
            target = end - 1 + h
            if target >= n or anchor_close == 0 or not math.isfinite(anchor_close):
                forward[h] = None
                forward_end_ts[h] = None
            else:
                forward[h] = float(
                    100.0 * (closes[target] - anchor_close) / anchor_close
                )
                forward_end_ts[h] = _to_dt(idx[target])

        max_fav: float | None = None
        max_adv: float | None = None
        if anchor_close > 0:
            far_end = min(end - 1 + max_h, n - 1)
            if far_end > end - 1:
                seg_high = highs[end:far_end + 1]
                seg_low = lows[end:far_end + 1]
                if seg_high.size:
                    max_fav = float(100.0 * (seg_high.max() - anchor_close) / anchor_close)
                if seg_low.size:
                    max_adv = float(100.0 * (seg_low.min() - anchor_close) / anchor_close)

        samples.append(WaveformSample(
            symbol=symbol,
            timeframe=timeframe,
            start_ts=_to_dt(idx[start]),
            end_ts=_to_dt(idx[end - 1]),
            signature=sig,
            forward_returns_pct=forward,
            max_favorable_pct=max_fav,
            max_adverse_pct=max_adv,
            forward_return_end_ts=forward_end_ts,
        ))

    return samples


def _to_dt(ts) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(timezone.utc)
    return t.to_pydatetime()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def write_library(path: Path, samples: Iterable[WaveformSample], *, append: bool = False) -> int:
    """Write samples to JSONL. Returns the number of records written.

    `append=True` is intended for incremental builds — refresh the
    library nightly without rewriting older samples.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    n = 0
    with p.open(mode, encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), default=str))
            f.write("\n")
            n += 1
    return n


def read_library(path: Path) -> list[WaveformSample]:
    """Read every line of a JSONL library file."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[WaveformSample] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(WaveformSample.from_dict(json.loads(line)))
    return out


__all__ = [
    "WaveformSample",
    "build_library",
    "write_library",
    "read_library",
]
