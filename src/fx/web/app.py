"""Flask app factory for the FX dashboard."""
from __future__ import annotations

from pathlib import Path

from flask import Flask

from ..config import Config
from ..storage import Storage
from .routes import bp


EVENTS_PATH = Path("data/events.json")


def create_app(
    config: Config | None = None,
    storage: Storage | None = None,
    events_path: Path = EVENTS_PATH,
) -> Flask:
    cfg = config or Config.load()
    store = storage or Storage(cfg.db_path)

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["FX_CONFIG"] = cfg
    app.config["FX_STORAGE"] = store
    app.config["FX_EVENTS_PATH"] = events_path
    app.register_blueprint(bp)
    return app
