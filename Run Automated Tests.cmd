@echo off
setlocal
cd /d "%~dp0"
title People Counter - Automated Tests

if not exist ".venv\Scripts\python.exe" (
    echo Run "Start People Counter.cmd" once first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m py_compile branch_config.py people_counter_hikvision.py runtime_storage.py runtime_logging.py staff_face.py staff_uniform.py staff_enrollment.py setup_branch.py watchdog.py status.py system_check.py
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m unittest discover -v
if errorlevel 1 goto :failed
echo.
echo ALL AUTOMATED TESTS PASSED.
pause
exit /b 0

:failed
echo.
echo TESTS FAILED. Do not deploy this version to a branch.
pause
exit /b 1
