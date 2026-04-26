"""One-click launcher for the FX dashboard.

Designed to be invoked by a double-click on the platform-specific
shims (start.command on macOS, start.bat on Windows, start.sh on Linux).

What it does:
  1. Make sure the required Python deps are installed (auto-runs
     `pip install -r requirements.txt` if anything is missing).
  2. Pick a free port (default 5000, fall back to 5001..5010).
  3. Start the Flask dashboard.
  4. Open the user's default browser at the dashboard URL.

It does NOT:
  - require a virtual environment (uses whichever python you launched it with)
  - assume ANTHROPIC_API_KEY is set (the app degrades gracefully to
    technical-only signals when the key is absent)
"""
from __future__ import annotations

import importlib
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED_MODULES = ("flask", "pandas", "numpy", "yfinance", "anthropic")
DEFAULT_PORT = 5000
PORT_SCAN_RANGE = range(DEFAULT_PORT, DEFAULT_PORT + 11)


def missing_deps() -> list[str]:
    """Return the names of required modules that are not yet importable."""
    missing: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


def install_deps() -> None:
    print("Installing dependencies (first run only)...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")]
    )


def find_free_port(candidates=PORT_SCAN_RANGE) -> int:
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No free port in {min(candidates)}..{max(candidates)}; "
        "close other servers and retry."
    )


def open_browser_when_ready(url: str, delay_s: float = 1.5) -> None:
    time.sleep(delay_s)
    try:
        webbrowser.open(url)
    except Exception as e:  # noqa: BLE001
        print(f"(could not auto-open browser: {e}; visit {url} manually)")


def main() -> int:
    os.chdir(ROOT)

    if missing := missing_deps():
        print(f"Missing modules: {', '.join(missing)}")
        try:
            install_deps()
        except subprocess.CalledProcessError as e:
            print(f"\n[error] dependency install failed: {e}")
            print("Try manually: pip install -r requirements.txt")
            input("Press Enter to exit.")
            return 1

    port = find_free_port()
    url = f"http://localhost:{port}"

    print("=" * 56)
    print(f"  fx/bot dashboard  →  {url}")
    print("  Press Ctrl+C in this window to stop.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [note] ANTHROPIC_API_KEY not set — technical-only mode.")
    print("=" * 56)

    threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()

    # Imported lazily so dep-install above can populate sys.path first.
    from src.fx.web import create_app

    app = create_app()
    try:
        app.run(host="127.0.0.1", port=port, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
