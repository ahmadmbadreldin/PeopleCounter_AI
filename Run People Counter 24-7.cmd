@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo Run "Start People Counter.cmd" once first.
    pause
    exit /b 1
)
if not exist "branch_config.json" (
    echo Configure and save the branch first using "Start People Counter.cmd".
    pause
    exit /b 1
)

start "" /min ".venv\Scripts\pythonw.exe" watchdog.py
exit /b 0
