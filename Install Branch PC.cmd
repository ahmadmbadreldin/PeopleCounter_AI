@echo off
setlocal
cd /d "%~dp0"
title Install People Counter on Branch PC

echo This changes THIS branch computer only:
echo - Adds People Counter startup after Windows sign-in
echo - Disables Sleep and Hibernate on AC and battery power
echo - Does NOT prevent normal nightly shutdown
echo.
choice /C YN /M "Continue on the branch computer"
if errorlevel 2 exit /b 0

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0branch_windows_setup.ps1"
if errorlevel 1 goto :error
echo.
echo Setup finished. Restart Windows once to verify automatic startup.
pause
exit /b 0

:error
echo.
echo Setup failed. Take a photo of this window and contact support.
pause
exit /b 1
