@echo off
setlocal
cd /d "%~dp0"
title People Counter - Monitoring

where python >nul 2>&1
if errorlevel 1 goto :python_missing
python bootstrap_environment.py
if errorlevel 1 goto :error

if not exist "branch_config.json" (
    echo No active branch configuration was found.
    echo Run "1 - Configure Branch.cmd" first and save the branch.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "from branch_config import load_branch_config, validate_config; validate_config(load_branch_config())"
if errorlevel 1 (
    echo.
    echo The saved branch configuration is incomplete or invalid.
    echo Run "1 - Configure Branch.cmd" and save it again.
    pause
    exit /b 1
)

start "" /min ".venv\Scripts\pythonw.exe" watchdog.py
exit /b 0

:python_missing
echo.
echo Python was not found. Install 64-bit Python 3.11 and enable "Add Python to PATH".
pause
exit /b 1

:error
echo.
echo Monitoring could not start. Take a photo of this window and contact support.
pause
exit /b 1
