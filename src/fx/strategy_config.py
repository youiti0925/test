"""Strategy parameter container (spec §10).

A typed bundle of every knob the engine exposes, plus YAML I/O. The
calibration sweep iterates over copies of this dataclass with one or
more fields varied — keeping the search space explicit and self-documenting.

Default values mirror the historical engine defaults so loading an empty
YAML reproduces today's behaviour.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

try:
    import yaml  # type: ignore
    HAVE_YAML = True
except ImportError:  # pragma: no cover - PyYAML optional
    HAVE_YAML = False


@dataclass(frozen=True)
class SwingConfig:
    lookback: int = 3
    min_prominence_atr: float = 0.3
    triple_top_tolerance_atr: float = 0.8
    neckline_break_atr: float = 0.2


@dataclass(frozen=True)
class RiskConfig:
    stop_atr: float = 2.0
    take_profit_atr: float = 3.0
    min_risk_reward: float = 1.5
    risk_pct: float = 0.01
    max_holding_bars: int = 48


@dataclass(frozen=True)
class EventsConfig:
    high_impact_hold_hours: float = 2.0
    central_bank_hold_hours: float = 6.0
    calendar_max_age_hours: float = 24.0


@dataclass(frozen=True)
class SpreadConfig:
    max_spread_pct: float = 0.05


@dataclass(frozen=True)
class WaveformConfig:
    window_bars: int = 60
    min_similar_count: int = 20
    min_confidence: float = 0.6
    method: str = "dtw"
    horizon_bars: int = 24
    min_directional_share: float = 0.6


@dataclass(frozen=True)
class StrategyConfig:
    """Top-level container. All sub-objects default to historical engine defaults."""

    symbol: str = "USDJPY=X"
    interval: str = "1h"
    higher_timeframe: str = "1d"
    swing: SwingConfig = field(default_factory=SwingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    spread: SpreadConfig = field(default_factory=SpreadConfig)
    waveform: WaveformConfig = field(default_factory=WaveformConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyConfig":
        return cls(
            symbol=d.get("symbol", "USDJPY=X"),
            interval=d.get("interval", "1h"),
            higher_timeframe=d.get("higher_timeframe", "1d"),
            swing=SwingConfig(**(d.get("swing") or {})),
            risk=RiskConfig(**(d.get("risk") or {})),
            events=EventsConfig(**(d.get("events") or {})),
            spread=SpreadConfig(**(d.get("spread") or {})),
            waveform=WaveformConfig(**(d.get("waveform") or {})),
        )

    def write(self, path: Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in (".yaml", ".yml") and HAVE_YAML:
            with p.open("w", encoding="utf-8") as f:
                yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        else:
            with p.open("w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "StrategyConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml") and HAVE_YAML:
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text or "{}")
        return cls.from_dict(data)


__all__ = [
    "EventsConfig",
    "RiskConfig",
    "SpreadConfig",
    "StrategyConfig",
    "SwingConfig",
    "WaveformConfig",
    "HAVE_YAML",
]
