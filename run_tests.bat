@echo off
SETLOCAL EnableDelayedExpansion

echo 🧪 Starting AI Knowledge Based Test Suite...
echo 🛡️  Activating C-Drive Zero-Footprint Mode (Test Context)...

:: 0. Redirect C-Drive Noise to D-Drive
if not exist "D:\AppTemp" mkdir "D:\AppTemp"
if not exist "D:\AppLogs" mkdir "D:\AppLogs"
SET TEMP=D:\AppTemp
SET TMP=D:\AppTemp
SET HF_HOME=D:\AI_Models_Cache
SET LOG_FILE_PATH=D:\AppLogs\app_test.log

:: 1. Set PYTHONPATH and Python Path
SET PYTHONPATH=%CD%
SET PYTHON_EXE=D:\DevEnvironments\Python311\python.exe

echo.
echo 🔍 Running all tests in ./tests directory...
"%PYTHON_EXE%" -m pytest tests/ --loglevel=info

echo.
echo ✅ Test run completed.
echo 💡 Check D:\AppLogs\app_test.log for detailed execution traces.
echo.
pause
