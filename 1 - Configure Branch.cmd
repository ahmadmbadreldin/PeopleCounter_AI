@echo off
setlocal
cd /d "%~dp0"
title People Counter - Branch Configuration

where python >nul 2>&1
if errorlevel 1 goto :python_missing
python bootstrap_environment.py
if errorlevel 1 goto :error

".venv\Scripts\python.exe" setup_branch.py --configure-only
if errorlevel 1 goto :error
exit /b 0

:python_missing
echo.
echo Python was not found. Install 64-bit Python 3.11 and enable "Add Python to PATH".
pause
exit /b 1

:error
echo.
echo Branch configuration could not start. Take a photo of this window and contact support.
pause
exit /b 1
