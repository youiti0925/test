"""Gunicorn / Flask entry point.

Local dev: `python sample.py` runs Flask's debug server on :5000.
For production-style deploy: `gunicorn sample:app --log-file=-`
(the project no longer ships a Procfile; pick your own deploy target).
"""
from src.fx.web import create_app

app = create_app()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
