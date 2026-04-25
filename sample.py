"""Gunicorn entry point.

Wires the existing `Procfile` (`gunicorn sample:app`) to the new FX
dashboard. Local dev: `python sample.py` runs Flask's debug server.
"""
from src.fx.web import create_app

app = create_app()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
