@echo off
setlocal
cd /d "%~dp0"
title Stop Old People Counter

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_people_counter.ps1"
if errorlevel 1 goto :error

echo.
echo Now run "Start People Counter.cmd", redraw the doorway, then click Save ^& Start.
pause
exit /b 0

:error
echo.
echo Could not stop every old People Counter process.
echo Right-click this file and choose "Run as administrator", then try again.
pause
exit /b 1
