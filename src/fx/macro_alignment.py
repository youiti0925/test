"""Macro alignment for royal_road_decision_v2 — symbol-aware.

Different FX pairs respond differently to DXY / yields / VIX. The
crude "DXY up = BUY" rule is rejected here. Each symbol gets its own
mapping table.

Symbol-specific rules (heuristic, NOT validated)
-----------------------------------------------

USDJPY:
  DXY up                     → BUY-leaning   (USD strength)
  US10Y yield up             → BUY-leaning   (carry favours JPY weakness)
  VIX up significantly       → SELL-leaning  (risk-off → JPY safe-haven bid)

EURUSD:
  DXY up                     → SELL-leaning  (mirror)
  US10Y yield up             → SELL-leaning
  VIX up                     → SELL-leaning  (USD safety bid)

GBPUSD:
  DXY up                     → SELL-leaning
  VIX up                     → SELL-leaning

AUDUSD:
  VIX up                     → SELL-leaning  (AUD risk-on currency)
  Risk-on equity rally       → BUY-leaning

Other / unknown symbols:
  Returned as alignment=UNKNOWN, score=0.0; never blocks trades on its
  own. The royal-road decision uses macro_alignment as a TIE-BREAKER /
  bonus, not as a hard gate, except when a strong "macro_strong_against"
  signal is present.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal


_DXY_TREND_BULL_BUCKETS: Final[set[str]] = {"UP", "STRONG_UP"}
_DXY_TREND_BEAR_BUCKETS: Final[set[str]] = {"DOWN", "STRONG_DOWN"}
_VIX_HIGH_LEVEL: Final[float] = 25.0
_VIX_VERY_HIGH_LEVEL: Final[float] = 30.0
_YIELD_DELTA_BULL_BP: Final[float] = 5.0
_YIELD_DELTA_BEAR_BP: Final[float] = -5.0


AlignmentLabel = Literal["BUY", "SELL", "NEUTRAL", "UNKNOWN"]
RegimeLabel = Literal["RISK_ON", "RISK_OFF", "MIXED", "UNKNOWN"]


@dataclass(frozen=True)
class MacroAlignmentSnapshot:
    schema_version: str
    symbol: str
    dxy_alignment: AlignmentLabel
    yield_alignment: AlignmentLabel
    vix_regime: Literal["LOW", "ELEVATED", "HIGH", "VERY_HIGH", "UNKNOWN"]
    risk_on_off: RegimeLabel
    currency_bias: AlignmentLabel
    macro_score: float                # -1.0 (strong SELL) .. +1.0 (strong BUY)
    macro_reasons: list = field(default_factory=list)
    macro_block_reasons: list = field(default_factory=list)
    macro_strong_against: AlignmentLabel = "UNKNOWN"
    unavailable_reasons: list = field(default_factory=list)
    # Hawkish / dovish tone of central bank events. v2 has NO data
    # source for this — we always emit UNKNOWN with an explicit
    # unavailable_reason so consumers don't mistake "implicit NEUTRAL"
    # for "no data". Implementing tone extraction (FOMC statement
    # parsing, ECB dovish/hawkish classifier) is future work; pinned
    # in `docs/royal_road_decision_v2.md`.
    event_tone: Literal["HAWKISH", "DOVISH", "NEUTRAL", "UNKNOWN"] = "UNKNOWN"
    event_tone_unavailable_reason: str = "tone_extraction_not_implemented"

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "dxy_alignment": self.dxy_alignment,
            "yield_alignment": self.yield_alignment,
            "vix_regime": self.vix_regime,
            "risk_on_off": self.risk_on_off,
            "currency_bias": self.currency_bias,
            "macro_score": float(self.macro_score),
            "macro_reasons": list(self.macro_reasons),
            "macro_block_reasons": list(self.macro_block_reasons),
            "macro_strong_against": self.macro_strong_against,
            "unavailable_reasons": list(self.unavailable_reasons),
            "event_tone": self.event_tone,
            "event_tone_unavailable_reason": self.event_tone_unavailable_reason,
        }


def empty_alignment(symbol: str, reason: str) -> MacroAlignmentSnapshot:
    return MacroAlignmentSnapshot(
        schema_version="macro_alignment_v2",
        symbol=symbol,
        dxy_alignment="UNKNOWN",
        yield_alignment="UNKNOWN",
        vix_regime="UNKNOWN",
        risk_on_off="UNKNOWN",
        currency_bias="UNKNOWN",
        macro_score=0.0,
        macro_reasons=[],
        macro_block_reasons=[],
        macro_strong_against="UNKNOWN",
        unavailable_reasons=[reason],
    )


def _dxy_alignment_for_symbol(symbol: str, bucket: str | None) -> AlignmentLabel:
    if bucket is None:
        return "UNKNOWN"
    if bucket in _DXY_TREND_BULL_BUCKETS:
        if symbol.startswith("USD"):       # USDJPY=X
            return "BUY"
        if symbol.endswith("USD=X"):       # EURUSD=X / GBPUSD=X / AUDUSD=X
            return "SELL"
        return "NEUTRAL"
    if bucket in _DXY_TREND_BEAR_BUCKETS:
        if symbol.startswith("USD"):
            return "SELL"
        if symbol.endswith("USD=X"):
            return "BUY"
        return "NEUTRAL"
    return "NEUTRAL"


def _yield_alignment_for_symbol(
    symbol: str, us10y_change_bp: float | None,
) -> AlignmentLabel:
    if us10y_change_bp is None:
        return "UNKNOWN"
    if us10y_change_bp >= _YIELD_DELTA_BULL_BP:
        if symbol.startswith("USD"):
            return "BUY"
        if symbol.endswith("USD=X"):
            return "SELL"
        return "NEUTRAL"
    if us10y_change_bp <= _YIELD_DELTA_BEAR_BP:
        if symbol.startswith("USD"):
            return "SELL"
        if symbol.endswith("USD=X"):
            return "BUY"
        return "NEUTRAL"
    return "NEUTRAL"


def _vix_regime_label(vix: float | None) -> str:
    if vix is None:
        return "UNKNOWN"
    if vix < 20.0:
        return "LOW"
    if vix < _VIX_HIGH_LEVEL:
        return "ELEVATED"
    if vix < _VIX_VERY_HIGH_LEVEL:
        return "HIGH"
    return "VERY_HIGH"


def _vix_alignment_for_symbol(symbol: str, vix: float | None) -> AlignmentLabel:
    if vix is None:
        return "UNKNOWN"
    if vix >= _VIX_HIGH_LEVEL:
        # Risk-off:
        # USDJPY: JPY safe-haven bid → SELL
        # AUDUSD: AUD risk-currency hit → SELL
        # GBPUSD: GBP risk-currency hit → SELL
        # EURUSD: USD safety bid → SELL
        return "SELL"
    if vix < 18.0:
        # Risk-on (low VIX):
        # USDJPY: NEUTRAL (no strong bias)
        # AUDUSD: BUY (AUD risk-on)
        # GBPUSD: NEUTRAL (slight BUY)
        # EURUSD: NEUTRAL
        if symbol.startswith("AUD"):
            return "BUY"
        return "NEUTRAL"
    return "NEUTRAL"


def _risk_on_off(vix_regime: str) -> RegimeLabel:
    if vix_regime in ("HIGH", "VERY_HIGH"):
        return "RISK_OFF"
    if vix_regime == "LOW":
        return "RISK_ON"
    if vix_regime == "ELEVATED":
        return "MIXED"
    return "UNKNOWN"


def _aggregate_score(
    dxy: AlignmentLabel, yld: AlignmentLabel, vix: AlignmentLabel,
) -> tuple[float, AlignmentLabel]:
    weights = {"BUY": 1.0, "SELL": -1.0, "NEUTRAL": 0.0, "UNKNOWN": 0.0}
    total = weights[dxy] + weights[yld] + weights[vix]
    n_known = sum(1 for x in (dxy, yld, vix) if x != "UNKNOWN")
    if n_known == 0:
        return 0.0, "UNKNOWN"
    score = total / max(n_known, 1)
    if score >= 0.5:
        bias: AlignmentLabel = "BUY"
    elif score <= -0.5:
        bias = "SELL"
    else:
        bias = "NEUTRAL"
    return score, bias


def compute_macro_alignment(
    *,
    symbol: str,
    macro_context: dict | None,
) -> MacroAlignmentSnapshot:
    """Build a macro alignment snapshot from the per-bar
    `MacroContextSlice.to_dict()` payload.

    `macro_context` may be None when the engine had no macro fetch.
    Returns UNKNOWN-shaped snapshot in that case (does not crash).
    """
    if not macro_context:
        return empty_alignment(symbol, "macro_context_unavailable")

    dxy_bucket = macro_context.get("dxy_trend_5d_bucket")
    us10y_change = macro_context.get("us10y_change_24h_bp")
    vix = macro_context.get("vix")

    dxy_align = _dxy_alignment_for_symbol(symbol, dxy_bucket)
    yld_align = _yield_alignment_for_symbol(symbol, us10y_change)
    vix_regime = _vix_regime_label(vix)
    vix_align = _vix_alignment_for_symbol(symbol, vix)
    risk_state = _risk_on_off(vix_regime)

    score, bias = _aggregate_score(dxy_align, yld_align, vix_align)

    reasons: list[str] = []
    block_reasons: list[str] = []
    unavailable: list[str] = []
    if dxy_align == "UNKNOWN":
        unavailable.append("dxy_unavailable")
    else:
        reasons.append(f"dxy:{dxy_bucket}->{dxy_align}")
    if yld_align == "UNKNOWN":
        unavailable.append("us10y_unavailable")
    else:
        reasons.append(f"us10y_change_24h_bp:{us10y_change:+.1f}->{yld_align}")
    if vix_align == "UNKNOWN":
        unavailable.append("vix_unavailable")
    else:
        reasons.append(f"vix:{vix:.1f} ({vix_regime})->{vix_align}")

    # macro_strong_against fires only on VERY_HIGH VIX (genuine risk-off).
    macro_strong_against: AlignmentLabel = "UNKNOWN"
    if vix_regime == "VERY_HIGH":
        # Risk-off + this symbol's vix_align direction tells us which
        # side macro is strongly against.
        if vix_align == "SELL":
            macro_strong_against = "BUY"
            block_reasons.append("vix_very_high_blocks_buy")
        elif vix_align == "BUY":
            macro_strong_against = "SELL"
            block_reasons.append("vix_very_high_blocks_sell")

    return MacroAlignmentSnapshot(
        schema_version="macro_alignment_v2",
        symbol=symbol,
        dxy_alignment=dxy_align,
        yield_alignment=yld_align,
        vix_regime=vix_regime,
        risk_on_off=risk_state,
        currency_bias=bias,
        macro_score=score,
        macro_reasons=reasons,
        macro_block_reasons=block_reasons,
        macro_strong_against=macro_strong_against,
        unavailable_reasons=unavailable,
    )


__all__ = [
    "MacroAlignmentSnapshot",
    "compute_macro_alignment",
    "empty_alignment",
]
