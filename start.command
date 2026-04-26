#!/usr/bin/env bash
# macOS double-click launcher.
# First-time setup: in Terminal run `chmod +x start.command` once.
set -e
cd "$(dirname "$0")"

# Pick whichever python3 is on PATH; fall back to py launcher.
if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY=python
fi

"$PY" launch.py
