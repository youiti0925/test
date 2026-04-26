"""Attribution: candidate causes for a price move (spec §8 / §17).

Important framing first
-----------------------
This module does NOT claim "the cause was X". Per spec §17, you cannot
prove causation from one price series. What it CAN do is rank a closed
taxonomy of candidate influences by the evidence available at the time
of the move. The output is always a *list of candidates with weights*,
not a single "the answer is X".

Closed taxonomy (the only labels we will ever emit)
---------------------------------------------------
  TECHNICAL_PATTERN  detected_pattern + neckline break aligned with the move
  CPI_SURPRISE       a CPI event fired in the window
  PCE_SURPRISE       a PCE event fired in the window
  FOMC_RISK          FOMC / Fed-related event in the window
  BOJ_EVENT          BOJ / 日銀 event in the window
  NFP_RISK           NFP / Employment in the window
  RATE_DECISION      generic rate decision (ECB / BoE / RBA / SNB)
  YIELD_MOVE         large concurrent move in US10Y or yield-spread
  DXY_MOVE           large concurrent move in DXY (matters for FX)
  RISK_ON_OFF        VIX / equity moves opposite the asset
  NEWS_SPIKE         news_volume spiked vs trailing baseline
  SENTIMENT_SPIKE    sentiment_score |z| or volume spike
  SPREAD_SPIKE       spread_pct out of normal regime
  UNKNOWN            no candidate cleared its threshold

Threshold defaults are defensive on purpose — a small move with a
small CPI surprise should NOT light up the candidate list. The whole
point of the module is to surface the "this looks like a CPI day"
moments, not to attribute every wiggle.

Inputs
------
You supply:
  * `move_pct`           the percent return being explained
  * `atr_at_move`        ATR at the start of the move (for sizing)
  * `events_nearby`      list of dicts (events with title/impact)
  * `pattern`            optional PatternResult (technical alignment)
  * `macro_changes_pct`  dict like {"us10y": +1.2, "dxy": -0.4, "vix": +6.0}
  * `sentiment_snapshot` optional dict from sentiment subsystem
  * `news_volume_z`      optional float — z-score of news volume vs trail
  * `spread_pct`         optional float
  * `spread_baseline_pct` optional float (typical regime)

The function is deterministic and pure — no I/O. Plug it on top of
the MarketTimeline rows for batch attribution research.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .patterns import PatternResult


# Closed taxonomy — `code` field on every AttributionCandidate must
# match one of these strings. Keep this list short and curated.
ATTRIBUTION_CODES = (
    "TECHNICAL_PATTERN",
    "CPI_SURPRISE",
    "PCE_SURPRISE",
    "FOMC_RISK",
    "BOJ_EVENT",
    "NFP_RISK",
    "RATE_DECISION",
    "YIELD_MOVE",
    "DXY_MOVE",
    "RISK_ON_OFF",
    "NEWS_SPIKE",
    "SENTIMENT_SPIKE",
    "SPREAD_SPIKE",
    "UNKNOWN",
)


@dataclass(frozen=True)
class AttributionCandidate:
    """One candidate cause with its evidence weight.

    `weight` ∈ [0, 1] is the relative confidence in *this* candidate,
    NOT a probability. Multiple candidates can have weight ≈ 1.0 when
    several factors line up — that's exactly what we want, because in
    practice macro days are confounded.
    """

    code: str
    weight: float
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "weight": round(self.weight, 4),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class AttributionResult:
    move_pct: float
    direction: str           # UP | DOWN | FLAT
    candidates: list[AttributionCandidate] = field(default_factory=list)
    notable_threshold_pct: float = 0.5
    notable: bool = False    # whether move_pct exceeded the threshold

    def top(self, k: int = 3) -> list[AttributionCandidate]:
        return sorted(self.candidates, key=lambda c: c.weight, reverse=True)[:k]

    def to_dict(self) -> dict:
        return {
            "move_pct": round(self.move_pct, 4),
            "direction": self.direction,
            "notable": self.notable,
            "notable_threshold_pct": self.notable_threshold_pct,
            "candidates": [c.to_dict() for c in self.candidates],
            "top": [c.to_dict() for c in self.top()],
        }


# ---------------------------------------------------------------------------
# Internal scoring helpers — each maps "raw evidence" to weight ∈ [0, 1]
# ---------------------------------------------------------------------------


def _direction_of(move_pct: float, threshold: float = 0.05) -> str:
    if abs(move_pct) <= threshold:
        return "FLAT"
    return "UP" if move_pct > 0 else "DOWN"


def _saturate(value: float, scale: float) -> float:
    """Map |value| into [0, 1] using saturation; |value|=scale → 0.5."""
    if scale <= 0:
        return 0.0
    x = abs(value) / scale
    return float(x / (1.0 + x))


def _events_with_keyword(
    events: Iterable[dict], keywords: tuple[str, ...]
) -> list[dict]:
    out = []
    for e in events or []:
        upper = (e.get("title") or "").upper()
        for kw in keywords:
            if kw in upper:
                out.append(e)
                break
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attribute_move(
    move_pct: float,
    *,
    atr_at_move: float | None = None,
    events_nearby: Iterable[dict] = (),
    pattern: PatternResult | None = None,
    macro_changes_pct: dict | None = None,
    sentiment_snapshot: dict | None = None,
    news_volume_z: float | None = None,
    spread_pct: float | None = None,
    spread_baseline_pct: float | None = None,
    notable_threshold_pct: float = 0.5,
) -> AttributionResult:
    """Score every candidate cause for `move_pct` and return them.

    Threshold semantics
    -------------------
    * `notable_threshold_pct`: a move smaller than this is reported but
      its `notable` flag stays False — most callers should ignore the
      candidates in that case.
    * Per-candidate score uses a saturation curve so a CPI surprise
      with macro_changes_pct.us10y = +5bp doesn't dwarf the same surprise
      with +1bp; both push the FOMC/CPI candidate above 0.5 weight, but
      the bigger move pushes harder.
    """
    direction = _direction_of(move_pct)
    notable = abs(move_pct) >= notable_threshold_pct
    candidates: list[AttributionCandidate] = []

    macro_changes_pct = dict(macro_changes_pct or {})
    events_list = list(events_nearby or [])

    # ── Event candidates ────────────────────────────────────────────
    cpi = _events_with_keyword(events_list, ("CPI",))
    if cpi:
        # Weight: 0.6 baseline; +0.2 if move > 0.5%; +0.2 if 10Y also moved.
        w = 0.6
        if abs(move_pct) >= 0.5:
            w += 0.2
        if abs(macro_changes_pct.get("us10y", 0.0)) >= 1.0:
            w += 0.2
        candidates.append(AttributionCandidate(
            "CPI_SURPRISE", min(1.0, w),
            {"events": cpi, "us10y_change_pct": macro_changes_pct.get("us10y")},
        ))

    pce = _events_with_keyword(events_list, ("PCE",))
    if pce:
        w = 0.5
        if abs(move_pct) >= 0.4:
            w += 0.2
        candidates.append(AttributionCandidate(
            "PCE_SURPRISE", min(1.0, w), {"events": pce},
        ))

    fomc = _events_with_keyword(events_list, ("FOMC", "FED CHAIR"))
    if fomc:
        w = 0.7
        if abs(move_pct) >= 0.4:
            w += 0.2
        candidates.append(AttributionCandidate(
            "FOMC_RISK", min(1.0, w), {"events": fomc},
        ))

    boj = _events_with_keyword(events_list, ("BOJ", "日銀", "UEDA"))
    if boj:
        w = 0.7
        if abs(move_pct) >= 0.4:
            w += 0.2
        candidates.append(AttributionCandidate(
            "BOJ_EVENT", min(1.0, w), {"events": boj},
        ))

    nfp = _events_with_keyword(events_list, ("NFP", "NONFARM", "EMPLOYMENT"))
    if nfp:
        w = 0.6
        if abs(move_pct) >= 0.4:
            w += 0.2
        candidates.append(AttributionCandidate(
            "NFP_RISK", min(1.0, w), {"events": nfp},
        ))

    rate = _events_with_keyword(events_list, ("RATE DECISION", "ECB", "BOE", "RBA", "SNB"))
    if rate:
        candidates.append(AttributionCandidate(
            "RATE_DECISION", 0.6, {"events": rate},
        ))

    # ── Macro-move candidates ───────────────────────────────────────
    us10 = macro_changes_pct.get("us10y")
    if us10 is not None and abs(us10) >= 0.5:
        # Direction agreement bumps weight: USD pairs typically rise on
        # 10Y up. We don't try to be cute about which direction matters
        # for which symbol — surface the candidate, let the analyst decide.
        weight = _saturate(us10, scale=2.0)  # |1bp| → ~0.33; |2bp| → 0.5
        candidates.append(AttributionCandidate(
            "YIELD_MOVE", weight,
            {"us10y_change_pct": us10,
             "us_short_change_pct": macro_changes_pct.get("us_short_yield_proxy")},
        ))

    dxy = macro_changes_pct.get("dxy")
    if dxy is not None and abs(dxy) >= 0.2:
        weight = _saturate(dxy, scale=0.6)
        candidates.append(AttributionCandidate(
            "DXY_MOVE", weight, {"dxy_change_pct": dxy},
        ))

    vix = macro_changes_pct.get("vix")
    sp500 = macro_changes_pct.get("sp500")
    risk_signal = 0.0
    if vix is not None and vix >= 5.0:
        risk_signal = max(risk_signal, _saturate(vix, scale=10.0))
    if sp500 is not None and abs(sp500) >= 1.0:
        risk_signal = max(risk_signal, _saturate(sp500, scale=2.0))
    if risk_signal > 0.0:
        candidates.append(AttributionCandidate(
            "RISK_ON_OFF", risk_signal,
            {"vix_change_pct": vix, "sp500_change_pct": sp500},
        ))

    # ── Pattern candidate ───────────────────────────────────────────
    if pattern is not None and pattern.detected_pattern:
        is_top = pattern.detected_pattern in (
            "DOUBLE_TOP_CANDIDATE", "TRIPLE_TOP_CANDIDATE", "HEAD_AND_SHOULDERS",
        )
        is_bottom = pattern.detected_pattern in (
            "DOUBLE_BOTTOM_CANDIDATE", "TRIPLE_BOTTOM_CANDIDATE",
            "INVERSE_HEAD_AND_SHOULDERS",
        )
        # The pattern is only credible as a cause if (a) a neckline broke
        # AND (b) the move sign agrees with the pattern's expected side.
        if pattern.neckline_broken and (
            (is_top and direction == "DOWN")
            or (is_bottom and direction == "UP")
        ):
            base = 0.5 + 0.4 * float(pattern.pattern_confidence or 0.0)
            candidates.append(AttributionCandidate(
                "TECHNICAL_PATTERN", min(1.0, base),
                {
                    "pattern": pattern.detected_pattern,
                    "neckline_broken": True,
                    "pattern_confidence": pattern.pattern_confidence,
                },
            ))

    # ── Sentiment / news candidates ─────────────────────────────────
    if sentiment_snapshot:
        velocity = abs(sentiment_snapshot.get("sentiment_velocity") or 0.0)
        mentions = sentiment_snapshot.get("mention_count_24h") or 0
        if velocity >= 0.5 or mentions >= 300:
            weight = max(_saturate(velocity, 0.6), _saturate(mentions, 500.0))
            candidates.append(AttributionCandidate(
                "SENTIMENT_SPIKE", weight,
                {"velocity": velocity, "mention_count_24h": mentions},
            ))

    if news_volume_z is not None and news_volume_z >= 1.5:
        candidates.append(AttributionCandidate(
            "NEWS_SPIKE", _saturate(news_volume_z, 2.0),
            {"news_volume_z": news_volume_z},
        ))

    # ── Spread candidate ───────────────────────────────────────────
    if spread_pct is not None and spread_baseline_pct is not None:
        if spread_baseline_pct > 0 and spread_pct >= 3.0 * spread_baseline_pct:
            weight = _saturate(
                spread_pct / spread_baseline_pct - 1.0, scale=4.0,
            )
            candidates.append(AttributionCandidate(
                "SPREAD_SPIKE", weight,
                {"spread_pct": spread_pct,
                 "baseline_pct": spread_baseline_pct},
            ))

    # ── ATR sanity tag (optional metadata) ─────────────────────────
    # If the move is far smaller than typical bar volatility, downgrade
    # everything except UNKNOWN — small noise rarely needs an explanation.
    if atr_at_move is not None and atr_at_move > 0:
        atr_pct = 100.0 * atr_at_move / max(1e-9, abs(move_pct) / max(abs(move_pct), 1e-9))
        # NOTE: we don't actually scale weights here — the threshold on
        # `notable` already conveys "smaller than ATR isn't worth a story".
        # This branch is kept as a hook for future ATR-aware downscaling.
        _ = atr_pct  # noqa: F841

    if not candidates:
        candidates.append(AttributionCandidate(
            "UNKNOWN", 1.0, {"note": "no candidate cleared its threshold"},
        ))

    return AttributionResult(
        move_pct=move_pct,
        direction=direction,
        candidates=candidates,
        notable_threshold_pct=notable_threshold_pct,
        notable=notable,
    )


__all__ = [
    "ATTRIBUTION_CODES",
    "AttributionCandidate",
    "AttributionResult",
    "attribute_move",
]
