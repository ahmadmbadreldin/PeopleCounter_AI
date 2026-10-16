@echo off
setlocal
cd /d "%~dp0"
title People Counter

if not exist ".venv\Scripts\python.exe" (
    echo Creating the local Python environment...
    python -m venv .venv
    if errorlevel 1 goto :error
)

".venv\Scripts\python.exe" -c "import importlib.util, cv2, numpy, scipy, filterpy, openpyxl; assert importlib.util.find_spec('ultralytics'); assert int(cv2.__version__.split('.')[0]) >= 5" >nul 2>&1
if errorlevel 1 (
    echo Installing project requirements. This can take several minutes...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
)

".venv\Scripts\python.exe" setup_branch.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo People Counter could not start. Take a photo of this window and contact support.
pause
exit /b 1
