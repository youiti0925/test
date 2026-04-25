#!/usr/bin/env bash
# Linux double-click launcher.
# First-time setup: `chmod +x start.sh` once. Most file managers will
# then offer "Run" or "Run in Terminal" on right-click.
set -e
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY=python
fi

"$PY" launch.py
