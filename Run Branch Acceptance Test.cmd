@echo off
setlocal
cd /d "%~dp0"
title People Counter - Branch Acceptance Test

if not exist ".venv\Scripts\python.exe" (
    echo Run "Start People Counter.cmd" once first.
    pause
    exit /b 1
)
if not exist "branch_config.json" (
    echo Configure and save this branch first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" system_check.py
if errorlevel 1 (
    echo.
    echo BRANCH CHECK FAILED. Do not finish installation yet.
    pause
    exit /b 1
)
echo.
echo TECHNICAL BRANCH CHECK PASSED. Live camera test was skipped.
echo To test the camera later, run: .venv\Scripts\python.exe system_check.py --camera
echo Continue with the manual counting trials in PRODUCTION_TEST_PLAN.md.
pause
