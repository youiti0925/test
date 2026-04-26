"""Waveform similarity primitives.

Goal (spec §7.1): given a price window, decide how similar it is to
another price window — independent of absolute price level. A USDJPY
double-top at 130 yen and one at 150 yen should compare as "very
similar" if the *shapes* match.

What's in here:
  * normalize_window: shape-only representation (start-price / z-score / ATR)
  * cosine_similarity / correlation_similarity: fast, length-equal compares
  * dtw_distance / dtw_similarity: tolerant to small time-axis stretches
  * structure_similarity: HH/HL/LH/LL string overlap
  * compute_signature: package the above into a comparable WaveformSignature
  * similarity(): top-level scorer that blends shape + structure

What's NOT in here:
  * Library storage / sliding-window labelling — see waveform_library.py
  * Aggregation of forward returns over similar matches — see waveform_backtest.py
  * Trade decisions — Decision Engine consumes waveform as advisory only.

Algorithmic notes
-----------------
* DTW uses an O(N×M) implementation with an optional Sakoe-Chiba band
  (`window_ratio`). For windows of 60-200 bars this runs in microseconds;
  larger libraries scan via cosine first and DTW only on the top-K.
* All similarities return a value in [0, 1] where 1 means identical.
* Cosine on z-scored vectors is invariant to BOTH price level AND scale,
  which is exactly what we want for cross-market shape compares.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

from .patterns import analyse as analyse_patterns


NormalizeMethod = Literal["start_price", "z_score", "atr", "min_max"]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_window(
    closes: np.ndarray,
    method: NormalizeMethod = "z_score",
    *,
    atr_value: float | None = None,
) -> np.ndarray:
    """Return a shape-only representation of `closes`.

    `start_price`:  closes / closes[0] - 1  (cumulative percent change)
    `z_score`:      (closes - mean) / std    (level- and scale-invariant)
    `atr`:          (closes - closes[0]) / atr_value (price displacement
                    in ATR units; preserves direction magnitude)
    `min_max`:      (closes - min) / (max - min) → [0, 1]
    """
    if closes.ndim != 1 or len(closes) == 0:
        raise ValueError("closes must be a 1-D non-empty array")

    if method == "start_price":
        base = closes[0]
        if base == 0:
            return np.zeros_like(closes)
        return closes / base - 1.0
    if method == "z_score":
        std = closes.std()
        if std == 0:
            return np.zeros_like(closes)
        return (closes - closes.mean()) / std
    if method == "atr":
        if atr_value is None or atr_value <= 0:
            raise ValueError("atr method requires a positive atr_value")
        return (closes - closes[0]) / atr_value
    if method == "min_max":
        lo, hi = closes.min(), closes.max()
        if hi == lo:
            return np.zeros_like(closes)
        return (closes - lo) / (hi - lo)
    raise ValueError(f"Unknown normalize method: {method!r}")


# ---------------------------------------------------------------------------
# Similarity primitives
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two equal-length vectors → [-1, 1].

    Returned in [0, 1] as `(cos + 1) / 2` so it composes uniformly with
    the other "higher = more similar" similarity scores.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.5  # no information; neutral
    cos = float(np.dot(a, b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return 0.5 * (cos + 1.0)


def correlation_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation mapped to [0, 1]."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.std() == 0 or b.std() == 0:
        return 0.5
    r = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(r):
        return 0.5
    return 0.5 * (r + 1.0)


def dtw_distance(
    a: np.ndarray, b: np.ndarray,
    *, window_ratio: float | None = 0.1,
) -> float:
    """Dynamic Time Warping distance with optional Sakoe-Chiba band.

    `window_ratio = 0.1` → a cell (i, j) is reachable only if
    |i/N − j/M| ≤ 0.1; this keeps DTW from collapsing wildly different
    series onto each other and gives a ~10× speedup vs the unconstrained
    version. Pass None to disable the band.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise ValueError("DTW requires non-empty inputs")

    band = None
    if window_ratio is not None:
        band = max(1, int(window_ratio * max(n, m)))

    INF = float("inf")
    cost = np.full((n + 1, m + 1), INF)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        if band is not None:
            j_lo = max(1, i - band)
            j_hi = min(m, i + band)
        else:
            j_lo, j_hi = 1, m
        for j in range(j_lo, j_hi + 1):
            d = abs(a[i - 1] - b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j],
                                 cost[i, j - 1],
                                 cost[i - 1, j - 1])

    final = cost[n, m]
    if final == INF:
        # Shouldn't happen with band ≥ 1, but be paranoid for safety
        return INF
    # Normalise by path length (n + m) so different lengths compare fairly
    return final / (n + m)


def dtw_similarity(
    a: np.ndarray, b: np.ndarray,
    *, scale: float = 1.0, window_ratio: float | None = 0.1,
) -> float:
    """DTW distance mapped to (0, 1] via exp(-d / scale).

    `scale` controls how forgiving we are. Set to ~1.0 when feeding
    z-scored inputs — typical "similar" series land around 0.3-0.6.
    """
    d = dtw_distance(a, b, window_ratio=window_ratio)
    if d == float("inf"):
        return 0.0
    return float(np.exp(-d / max(scale, 1e-9)))


def structure_similarity(a: list[str], b: list[str]) -> float:
    """Jaccard-style overlap on aligned market-structure codes.

    Each element is one of HH / HL / LH / LL. We compare the trailing
    `min(len(a), len(b))` elements position-by-position to reward
    structural agreement at the most recent few swings.
    """
    if not a or not b:
        return 0.0
    k = min(len(a), len(b))
    aa, bb = a[-k:], b[-k:]
    matches = sum(1 for x, y in zip(aa, bb) if x == y)
    return matches / k


# ---------------------------------------------------------------------------
# Signature object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaveformSignature:
    """Comparable shape representation of a price window.

    `vector` is the normalised close series; `structure` is the
    market_structure list from `patterns.analyse`. Both are needed to
    score holistic similarity: shape match alone can fool you on
    isolated wiggles that don't actually form the same waveform.
    """

    vector: np.ndarray
    structure: list[str]
    swing_structure: list[str]
    trend_state: str
    detected_pattern: str | None
    length: int
    method: NormalizeMethod = "z_score"
    atr: float | None = None

    def to_dict(self) -> dict:
        return {
            "vector": [float(x) for x in self.vector],
            "structure": list(self.structure),
            "swing_structure": list(self.swing_structure),
            "trend_state": self.trend_state,
            "detected_pattern": self.detected_pattern,
            "length": self.length,
            "method": self.method,
            "atr": self.atr,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WaveformSignature":
        return cls(
            vector=np.asarray(d["vector"], dtype=float),
            structure=list(d.get("structure") or []),
            swing_structure=list(d.get("swing_structure") or []),
            trend_state=d.get("trend_state") or "RANGE",
            detected_pattern=d.get("detected_pattern"),
            length=int(d.get("length") or len(d["vector"])),
            method=d.get("method") or "z_score",
            atr=d.get("atr"),
        )


def compute_signature(
    df: pd.DataFrame,
    *,
    method: NormalizeMethod = "z_score",
    atr_value: float | None = None,
) -> WaveformSignature:
    """Build a WaveformSignature from an OHLCV window.

    Caller controls `method` — z_score is the default because it makes
    cross-instrument compares trivial. For attribution work you may
    prefer `atr` or `start_price` to keep magnitude information.
    """
    closes = df["close"].to_numpy(dtype=float)
    vec = normalize_window(closes, method=method, atr_value=atr_value)
    pat = analyse_patterns(df)
    return WaveformSignature(
        vector=vec,
        structure=list(pat.market_structure),
        swing_structure=list(pat.swing_structure),
        trend_state=pat.trend_state.value,
        detected_pattern=pat.detected_pattern,
        length=len(vec),
        method=method,
        atr=atr_value,
    )


# ---------------------------------------------------------------------------
# Top-level similarity
# ---------------------------------------------------------------------------


SimilarityMethod = Literal["cosine", "correlation", "dtw"]


def similarity(
    a: WaveformSignature,
    b: WaveformSignature,
    *,
    method: SimilarityMethod = "dtw",
    structure_weight: float = 0.25,
    dtw_window_ratio: float | None = 0.1,
) -> float:
    """Composite similarity in [0, 1] — higher = more similar.

    Score = (1 - structure_weight) * shape + structure_weight * structure_match.
    `shape` uses cosine, correlation, or DTW depending on `method`.
    Vectors are length-aligned to the shorter common length first; DTW
    handles small differences natively, the other two require equal
    length so we left-truncate.
    """
    if not 0.0 <= structure_weight <= 1.0:
        raise ValueError("structure_weight must be in [0, 1]")

    if method == "dtw":
        # DTW tolerates length differences directly.
        shape = dtw_similarity(
            a.vector, b.vector, window_ratio=dtw_window_ratio,
        )
    else:
        n = min(len(a.vector), len(b.vector))
        if n == 0:
            return 0.0
        va, vb = a.vector[-n:], b.vector[-n:]
        if method == "cosine":
            shape = cosine_similarity(va, vb)
        else:  # correlation
            shape = correlation_similarity(va, vb)

    struct = structure_similarity(a.structure, b.structure)
    return (1.0 - structure_weight) * shape + structure_weight * struct


__all__ = [
    "NormalizeMethod",
    "SimilarityMethod",
    "WaveformSignature",
    "normalize_window",
    "cosine_similarity",
    "correlation_similarity",
    "dtw_distance",
    "dtw_similarity",
    "structure_similarity",
    "compute_signature",
    "similarity",
]
