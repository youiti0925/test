"""Waveform-based forward statistics (spec §7.3).

Given a current window's signature and a library of past samples, find
the K most similar past windows and aggregate their realised forward
returns. Output is a `WaveformBias` the Decision Engine can consume
as ADVISORY input — never as the sole reason to enter.

Hard rules (spec §7.4)
----------------------
* Below `min_sample_count` matches → bias = HOLD, confidence = 0.
  We refuse to call a verdict from too few examples.
* `confidence` is the share of similar samples that agree on the sign
  of the chosen horizon's return, scaled by the average match score.
  It NEVER exceeds the directional share — a 60/40 split with perfect
  shape match still scores 0.6, not 1.0.
* The bias label only flips to BUY/SELL when the agreeing share
  exceeds `min_directional_share` (default 0.6). Otherwise HOLD.
* Forward labels of the input signature itself are NOT consulted (and
  the input has none anyway — `compute_signature` doesn't compute them).
  We never feed the future of the asked-about window into the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .waveform_library import WaveformSample
from .waveform_matcher import (
    SimilarityMethod,
    WaveformSignature,
    similarity,
)


@dataclass(frozen=True)
class WaveformMatch:
    """One library hit ranked by similarity to the asked-about window."""

    score: float
    sample: WaveformSample

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "symbol": self.sample.symbol,
            "timeframe": self.sample.timeframe,
            "end_ts": self.sample.end_ts.isoformat(),
            "trend_state": self.sample.signature.trend_state,
            "detected_pattern": self.sample.signature.detected_pattern,
            "forward_returns_pct": dict(self.sample.forward_returns_pct),
            "max_favorable_pct": self.sample.max_favorable_pct,
            "max_adverse_pct": self.sample.max_adverse_pct,
        }


@dataclass(frozen=True)
class WaveformBias:
    """Advisory directional bias the Decision Engine consumes.

    `action` is BUY / SELL / HOLD. `confidence` is the share of similar
    samples agreeing on the sign of the chosen horizon, attenuated by
    the average similarity score and by sample count.
    """

    action: str                # BUY | SELL | HOLD
    confidence: float          # 0.0 .. 1.0
    sample_count: int
    horizon_bars: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    avg_forward_return_pct: float | None
    avg_max_favorable_pct: float | None
    avg_max_adverse_pct: float | None
    avg_match_score: float
    method: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "sample_count": self.sample_count,
            "horizon_bars": self.horizon_bars,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "avg_forward_return_pct": (
                round(self.avg_forward_return_pct, 4)
                if self.avg_forward_return_pct is not None else None
            ),
            "avg_max_favorable_pct": (
                round(self.avg_max_favorable_pct, 4)
                if self.avg_max_favorable_pct is not None else None
            ),
            "avg_max_adverse_pct": (
                round(self.avg_max_adverse_pct, 4)
                if self.avg_max_adverse_pct is not None else None
            ),
            "avg_match_score": round(self.avg_match_score, 4),
            "method": self.method,
            "reason": self.reason,
        }


def _bias_label(
    bullish: int, bearish: int, total: int,
    *, min_directional_share: float,
) -> str:
    if total == 0:
        return "HOLD"
    if bullish / total >= min_directional_share and bullish > bearish:
        return "BUY"
    if bearish / total >= min_directional_share and bearish > bullish:
        return "SELL"
    return "HOLD"


def _empty_bias(
    *, sample_count: int, horizon: int, method: str, reason: str,
) -> WaveformBias:
    return WaveformBias(
        action="HOLD",
        confidence=0.0,
        sample_count=sample_count,
        horizon_bars=horizon,
        bullish_count=0,
        bearish_count=0,
        neutral_count=0,
        avg_forward_return_pct=None,
        avg_max_favorable_pct=None,
        avg_max_adverse_pct=None,
        avg_match_score=0.0,
        method=method,
        reason=reason,
    )


def find_similar(
    target: WaveformSignature,
    library: Iterable[WaveformSample],
    *,
    method: SimilarityMethod = "dtw",
    top_k: int = 30,
    min_score: float = 0.55,
    structure_weight: float = 0.25,
) -> list[WaveformMatch]:
    """Return the top-K samples most similar to `target`.

    `min_score` filters out weak matches before ranking — saves time on
    large libraries and prevents the "30 noisy lookalikes" problem.
    """
    matches: list[WaveformMatch] = []
    for sample in library:
        try:
            s = similarity(
                target, sample.signature,
                method=method, structure_weight=structure_weight,
            )
        except (ValueError, IndexError):
            continue
        if s < min_score:
            continue
        matches.append(WaveformMatch(score=s, sample=sample))

    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:top_k]


def aggregate_bias(
    matches: list[WaveformMatch],
    *,
    horizon_bars: int,
    method: str = "dtw",
    min_sample_count: int = 20,
    min_directional_share: float = 0.6,
    neutral_band_pct: float = 0.05,
) -> WaveformBias:
    """Turn a list of similar past matches into a directional bias.

    Parameters
    ----------
    horizon_bars:
        Which forward horizon to score. Must exist in the samples'
        `forward_returns_pct` dict (otherwise samples are skipped).
    min_sample_count:
        Below this many usable matches we refuse to label — bias=HOLD,
        confidence=0. Spec §7.3 rule.
    min_directional_share:
        Fraction of usable matches that must agree on the sign of the
        forward return for the bias to flip to BUY/SELL.
    neutral_band_pct:
        Returns within ±neutral_band_pct percent are counted neutral
        (neither bullish nor bearish). Defaults to 0.05% — tighter than
        the average bar move, so most samples vote.
    """
    if not matches:
        return _empty_bias(
            sample_count=0, horizon=horizon_bars, method=method,
            reason="no matches",
        )

    usable = [
        m for m in matches
        if m.sample.forward_returns_pct.get(horizon_bars) is not None
    ]
    if len(usable) < min_sample_count:
        return _empty_bias(
            sample_count=len(usable), horizon=horizon_bars, method=method,
            reason=(
                f"only {len(usable)} usable matches at horizon={horizon_bars} "
                f"(need ≥ {min_sample_count})"
            ),
        )

    returns = np.asarray(
        [m.sample.forward_returns_pct[horizon_bars] for m in usable],
        dtype=float,
    )
    bullish = int((returns > neutral_band_pct).sum())
    bearish = int((returns < -neutral_band_pct).sum())
    neutral = len(returns) - bullish - bearish

    label = _bias_label(
        bullish, bearish, len(returns),
        min_directional_share=min_directional_share,
    )

    avg_score = float(np.mean([m.score for m in usable]))
    # Confidence: directional agreement * average match score. Capped
    # at the agreeing share so cosmetics never bump a 60/40 split above
    # 60% confidence.
    if label == "BUY":
        agree = bullish / len(returns)
    elif label == "SELL":
        agree = bearish / len(returns)
    else:
        agree = max(bullish, bearish) / len(returns)
    confidence = min(agree, agree * avg_score) if label != "HOLD" else 0.0

    favs = [m.sample.max_favorable_pct for m in usable if m.sample.max_favorable_pct is not None]
    advs = [m.sample.max_adverse_pct for m in usable if m.sample.max_adverse_pct is not None]

    return WaveformBias(
        action=label,
        confidence=confidence,
        sample_count=len(returns),
        horizon_bars=horizon_bars,
        bullish_count=bullish,
        bearish_count=bearish,
        neutral_count=neutral,
        avg_forward_return_pct=float(returns.mean()),
        avg_max_favorable_pct=float(np.mean(favs)) if favs else None,
        avg_max_adverse_pct=float(np.mean(advs)) if advs else None,
        avg_match_score=avg_score,
        method=method,
        reason=(
            f"{bullish} up / {bearish} down / {neutral} flat "
            f"of {len(returns)} similar past windows"
        ),
    )


def waveform_lookup(
    target: WaveformSignature,
    library: Iterable[WaveformSample],
    *,
    horizon_bars: int = 24,
    method: SimilarityMethod = "dtw",
    top_k: int = 30,
    min_score: float = 0.55,
    min_sample_count: int = 20,
    min_directional_share: float = 0.6,
    structure_weight: float = 0.25,
) -> tuple[WaveformBias, list[WaveformMatch]]:
    """End-to-end: find_similar → aggregate_bias. Returns both."""
    matches = find_similar(
        target, library,
        method=method, top_k=top_k, min_score=min_score,
        structure_weight=structure_weight,
    )
    bias = aggregate_bias(
        matches,
        horizon_bars=horizon_bars,
        method=method,
        min_sample_count=min_sample_count,
        min_directional_share=min_directional_share,
    )
    return bias, matches


__all__ = [
    "WaveformBias",
    "WaveformMatch",
    "find_similar",
    "aggregate_bias",
    "waveform_lookup",
]
