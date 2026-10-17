@echo off
setlocal
cd /d "%~dp0"
title People Counter Status
if not exist ".venv\Scripts\python.exe" (
    echo People Counter is not installed yet.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" status.py
echo.
pause
