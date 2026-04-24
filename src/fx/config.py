"""Configuration loaded from env vars."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str | None
    model: str
    effort: str
    db_path: Path
    default_symbol: str
    default_interval: str

    @classmethod
    def load(cls) -> "Config":
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model=os.environ.get("FX_MODEL", "claude-opus-4-7"),
            effort=os.environ.get("FX_EFFORT", "medium"),
            db_path=Path(os.environ.get("FX_DB_PATH", "data/fx.db")),
            default_symbol=os.environ.get("FX_SYMBOL", "USDJPY=X"),
            default_interval=os.environ.get("FX_INTERVAL", "1h"),
        )
