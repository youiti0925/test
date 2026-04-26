@echo off
REM Windows double-click launcher.
cd /d "%~dp0"

REM Try the py launcher (recommended on Windows), then fall back to python.
where py >nul 2>nul
if %errorlevel%==0 (
    py launch.py
) else (
    python launch.py
)

REM Keep the console open so the user can read errors if it crashed.
pause
