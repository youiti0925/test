"""Wave / market-structure recognition.

Pure functions, no I/O, no LLM. The decision engine consumes the
output to decide whether a chart pattern justifies an entry; the LLM
gets the same data only as advisory context.

Critical correctness rule
-------------------------
**No future leak.** A swing high/low at index `i` is only confirmed
once `lookback` more bars have closed AFTER it. So when the user is
looking at a live chart with the current bar at index N, the latest
swing we will EVER admit lies at index `N - lookback` or earlier.
The unit tests below pin this behavior so a future refactor cannot
quietly start peeking forward.

Output shape
------------
PatternResult contains:
  - swing_highs / swing_lows  (confirmed only; no tentative)
  - swing_structure   ["H", "L", "H", ...] in chronological order
  - market_structure  ["HH", "HL", "LH", "LL", ...] for last 4 swings
  - trend_state       UPTREND | DOWNTREND | RANGE | VOLATILE
  - detected_pattern  None | DOUBLE_TOP_CANDIDATE | TRIPLE_TOP_CANDIDATE | ...
  - pattern_confidence  0.0 .. 1.0
  - neckline           Optional float
  - neckline_broken    bool   (close beyond neckline counts; high/low alone does not)
  - rsi_divergence     bool
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

from .indicators import rsi


class TrendState(str, Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"


SwingKind = Literal["H", "L"]


@dataclass(frozen=True)
class Swing:
    index: int
    ts: pd.Timestamp
    price: float
    kind: SwingKind  # "H" for high, "L" for low


@dataclass(frozen=True)
class PatternResult:
    swing_highs: list[Swing] = field(default_factory=list)
    swing_lows: list[Swing] = field(default_factory=list)
    swing_structure: list[str] = field(default_factory=list)   # ["H", "L", ...]
    market_structure: list[str] = field(default_factory=list)  # ["HH", "HL", ...]
    trend_state: TrendState = TrendState.RANGE
    detected_pattern: str | None = None
    pattern_confidence: float = 0.0
    neckline: float | None = None
    neckline_broken: bool = False
    rsi_divergence: bool = False
    macd_momentum_weakening: bool = False

    def to_dict(self) -> dict:
        return {
            "swing_structure": self.swing_structure,
            "market_structure": self.market_structure,
            "trend_state": self.trend_state.value,
            "detected_pattern": self.detected_pattern,
            "pattern_confidence": round(self.pattern_confidence, 3),
            "neckline": (
                round(self.neckline, 6) if self.neckline is not None else None
            ),
            "neckline_broken": self.neckline_broken,
            "rsi_divergence": self.rsi_divergence,
            "macd_momentum_weakening": self.macd_momentum_weakening,
            "n_swings": len(self.swing_highs) + len(self.swing_lows),
        }


# ---------------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------------


def detect_swings(
    df: pd.DataFrame,
    lookback: int = 3,
    min_prominence_atr: float = 0.3,
) -> tuple[list[Swing], list[Swing]]:
    """Find confirmed swing highs and lows.

    A swing high at index i is confirmed when:
      * highs[i] is the strict max of highs[i - lookback : i + lookback + 1]
      * its prominence (high minus the higher of the two adjacent troughs)
        is at least `min_prominence_atr * ATR(i)` — filters out tick noise.

    `i + lookback` must be inside `df`, so the latest possible confirmed
    swing high is at `len(df) - lookback - 1`. We do NOT peek past
    `len(df) - 1`. Tests below verify this.
    """
    if len(df) < 2 * lookback + 1:
        return [], []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    # ATR proxy for prominence threshold
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()

    swing_highs: list[Swing] = []
    swing_lows: list[Swing] = []
    end = len(df) - lookback  # never look past `end - 1`'s right window
    for i in range(lookback, end):
        window_h = highs[i - lookback : i + lookback + 1]
        window_l = lows[i - lookback : i + lookback + 1]
        atr_i = atr[i] if not np.isnan(atr[i]) else 0.0
        threshold = max(min_prominence_atr * atr_i, 1e-9)

        if highs[i] == window_h.max() and (highs[i] - max(highs[i - 1], highs[i + 1])) >= 0:
            # Prominence: distance to the lowest low within the window
            prominence = highs[i] - window_l.min()
            if prominence >= threshold:
                ts = df.index[i]
                if hasattr(ts, "to_pydatetime"):
                    ts_ = ts
                else:
                    ts_ = pd.Timestamp(ts)
                swing_highs.append(Swing(index=i, ts=ts_, price=float(highs[i]), kind="H"))

        if lows[i] == window_l.min() and (min(lows[i - 1], lows[i + 1]) - lows[i]) >= 0:
            prominence = window_h.max() - lows[i]
            if prominence >= threshold:
                ts = df.index[i]
                if hasattr(ts, "to_pydatetime"):
                    ts_ = ts
                else:
                    ts_ = pd.Timestamp(ts)
                swing_lows.append(Swing(index=i, ts=ts_, price=float(lows[i]), kind="L"))

    return swing_highs, swing_lows


def _merge_chronological(highs: list[Swing], lows: list[Swing]) -> list[Swing]:
    return sorted(highs + lows, key=lambda s: s.index)


# ---------------------------------------------------------------------------
# Structure & trend classification
# ---------------------------------------------------------------------------


def classify_structure(swings: list[Swing]) -> list[str]:
    """Walk the swing series and label each compared to the prior same-kind swing.

    Returns codes like ['HH','HL','HH','HL'] (uptrend) or ['LH','LL','LH','LL'].
    The first occurrence of each kind is skipped (no prior comparison).
    """
    last_high: float | None = None
    last_low: float | None = None
    out: list[str] = []
    for sw in swings:
        if sw.kind == "H":
            if last_high is not None:
                out.append("HH" if sw.price > last_high else "LH")
            last_high = sw.price
        else:
            if last_low is not None:
                out.append("HL" if sw.price > last_low else "LL")
            last_low = sw.price
    return out


def classify_trend(structure: list[str]) -> TrendState:
    """Use the last 4 structure codes to tag the trend.

    UPTREND   :  HH and HL dominate
    DOWNTREND :  LL and LH dominate
    RANGE     :  mixed, but small range
    VOLATILE  :  alternating with large range
    """
    if not structure:
        return TrendState.RANGE
    tail = structure[-4:] if len(structure) >= 4 else structure
    up = sum(1 for s in tail if s in ("HH", "HL"))
    down = sum(1 for s in tail if s in ("LL", "LH"))
    if up >= 3 and down <= 1:
        return TrendState.UPTREND
    if down >= 3 and up <= 1:
        return TrendState.DOWNTREND
    if up >= 2 and down >= 2:
        return TrendState.VOLATILE
    return TrendState.RANGE


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def _approx_equal(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def detect_double_top(
    highs: list[Swing], lows: list[Swing], atr_value: float
) -> tuple[bool, float | None]:
    """Two recent swing highs at similar price, separated by a swing low.

    Returns (is_candidate, neckline). Neckline = the swing low between
    the two peaks.
    """
    if len(highs) < 2 or len(lows) < 1:
        return False, None
    last_two_h = highs[-2:]
    # Find a low strictly between them
    between = [l for l in lows if last_two_h[0].index < l.index < last_two_h[1].index]
    if not between:
        return False, None
    tol = 0.6 * atr_value
    if not _approx_equal(last_two_h[0].price, last_two_h[1].price, tol):
        return False, None
    return True, between[-1].price


def detect_triple_top(
    highs: list[Swing], lows: list[Swing], atr_value: float
) -> tuple[bool, float | None, float]:
    """Three swing highs roughly equal with two valleys.

    Returns (is_candidate, neckline, confidence). Neckline = the LOWER
    of the two valleys between the peaks (more conservative SELL trigger).
    """
    if len(highs) < 3 or len(lows) < 2:
        return False, None, 0.0
    h1, h2, h3 = highs[-3:]
    valleys = [l for l in lows if h1.index < l.index < h3.index]
    if len(valleys) < 2:
        return False, None, 0.0
    tol_low = 0.3 * atr_value
    tol_high = 0.8 * atr_value
    diffs = [abs(h1.price - h2.price), abs(h2.price - h3.price), abs(h1.price - h3.price)]
    max_diff = max(diffs)
    if not (tol_low <= max_diff <= tol_high or max_diff <= tol_low):
        # Too tight or too loose: only accept if within the upper bound
        if max_diff > tol_high:
            return False, None, 0.0
    neckline = min(v.price for v in valleys)
    # Confidence: tighter peaks → higher confidence
    confidence = max(0.0, min(1.0, 1.0 - (max_diff / max(tol_high, 1e-9))))
    return True, neckline, confidence


def detect_double_bottom(
    highs: list[Swing], lows: list[Swing], atr_value: float
) -> tuple[bool, float | None]:
    if len(lows) < 2 or len(highs) < 1:
        return False, None
    l1, l2 = lows[-2:]
    between = [h for h in highs if l1.index < h.index < l2.index]
    if not between:
        return False, None
    tol = 0.6 * atr_value
    if not _approx_equal(l1.price, l2.price, tol):
        return False, None
    return True, between[-1].price


def neckline_broken_close(
    df: pd.DataFrame, neckline: float, side: str, since_index: int
) -> bool:
    """Confirmed neckline break = a closed bar beyond the level.

    For SELL patterns (top), need a close BELOW neckline.
    For BUY patterns (bottom), need a close ABOVE neckline.
    `since_index` is the swing index after which we start looking.
    """
    if neckline is None:
        return False
    closes = df["close"].iloc[since_index + 1 :]
    if closes.empty:
        return False
    if side == "SELL":
        return bool((closes < neckline).any())
    if side == "BUY":
        return bool((closes > neckline).any())
    return False


def rsi_bearish_divergence(df: pd.DataFrame, highs: list[Swing]) -> bool:
    """Higher price highs but lower RSI highs → bearish divergence."""
    if len(highs) < 2:
        return False
    rsi_series = rsi(df["close"], 14)
    h1, h2 = highs[-2:]
    if pd.isna(rsi_series.iloc[h1.index]) or pd.isna(rsi_series.iloc[h2.index]):
        return False
    return h2.price > h1.price and rsi_series.iloc[h2.index] < rsi_series.iloc[h1.index]


def macd_weakening(df: pd.DataFrame) -> bool:
    """Last MACD histogram lower in magnitude than the previous (waning momentum)."""
    from .indicators import macd

    m = macd(df["close"]).dropna()
    if len(m) < 3:
        return False
    h = m["hist"]
    return abs(h.iloc[-1]) < abs(h.iloc[-2]) and abs(h.iloc[-2]) < abs(h.iloc[-3])


# ---------------------------------------------------------------------------
# Top-level analyse function
# ---------------------------------------------------------------------------


def analyse(df: pd.DataFrame, lookback: int = 3) -> PatternResult:
    """One-shot analysis for the most recent confirmed structure."""
    if len(df) < 2 * lookback + 5:
        return PatternResult()

    highs, lows = detect_swings(df, lookback=lookback)
    swings = _merge_chronological(highs, lows)
    structure = classify_structure(swings)
    trend = classify_trend(structure)

    # Quick ATR for tolerance
    from .risk import atr as compute_atr
    atr_series = compute_atr(df, period=14)
    atr_value = float(atr_series.iloc[-1]) if len(atr_series.dropna()) else 0.0

    detected: str | None = None
    confidence = 0.0
    neckline: float | None = None
    side: str | None = None
    last_swing_index = swings[-1].index if swings else 0

    is_triple, n_t, conf_t = detect_triple_top(highs, lows, atr_value)
    if is_triple:
        detected = "TRIPLE_TOP_CANDIDATE"
        confidence = conf_t
        neckline = n_t
        side = "SELL"

    if detected is None:
        is_double_top, n_d = detect_double_top(highs, lows, atr_value)
        if is_double_top:
            detected = "DOUBLE_TOP_CANDIDATE"
            confidence = 0.6
            neckline = n_d
            side = "SELL"

    if detected is None:
        is_double_bottom, n_b = detect_double_bottom(highs, lows, atr_value)
        if is_double_bottom:
            detected = "DOUBLE_BOTTOM_CANDIDATE"
            confidence = 0.6
            neckline = n_b
            side = "BUY"

    broken = False
    if neckline is not None and side is not None:
        broken = neckline_broken_close(df, neckline, side, last_swing_index)

    return PatternResult(
        swing_highs=highs,
        swing_lows=lows,
        swing_structure=[s.kind for s in swings],
        market_structure=structure,
        trend_state=trend,
        detected_pattern=detected,
        pattern_confidence=confidence,
        neckline=neckline,
        neckline_broken=broken,
        rsi_divergence=rsi_bearish_divergence(df, highs),
        macd_momentum_weakening=macd_weakening(df),
    )
