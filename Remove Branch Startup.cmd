@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0remove_branch_windows_setup.ps1"
if errorlevel 1 (
    echo Removal failed. Take a photo of this window and contact support.
)
pause
