@echo off
SETLOCAL EnableDelayedExpansion

echo 🚀 Starting AI Knowledge Based System Orchestrator...
echo 🛡️  Activating C-Drive Zero-Footprint Mode...

:: 0. Redirect C-Drive Noise to D-Drive
if not exist "D:\AppTemp" mkdir "D:\AppTemp"
if not exist "D:\AppLogs" mkdir "D:\AppLogs"
SET TEMP=D:\AppTemp
SET TMP=D:\AppTemp
SET HF_HOME=D:\AI_Models_Cache
SET LOG_FILE_PATH=D:\AppLogs\app.log

echo.

:: 1. Check for .env file
if not exist ".env" (
    echo ❌ ERROR: .env file not found. Please create one based on .env.example.
    pause
    exit /b 1
)

:: 2. Set PYTHONPATH and Python Path
SET PYTHONPATH=%CD%
SET PYTHON_APP_EXE=D:\DevEnvironments\Python311\python.exe
SET CELERY_APP_EXE=D:\DevEnvironments\Python311\Scripts\celery.exe

:: 3. Start Portable Redis (Zero Footprint on D: Drive)
echo 🔍 Starting Portable Redis on D: Drive (Port 6380)...
SET REDIS_APP_EXE=D:\Redis\redis-server.exe
if exist "%REDIS_APP_EXE%" (
    echo ✅ Redis binaries found. Launching on Port 6380...
    start "Redis Server" "%REDIS_APP_EXE%" --port 6380
) else (
    echo ❌ ERROR: Redis binary not found at %REDIS_APP_EXE%.
    echo Please check the D:\Redis folder.
)

:: 4. Start Celery Worker
echo ⚙️  Starting Celery Worker (Windows Optimized)...
start "Celery Worker" cmd /k ""%CELERY_APP_EXE%" -A app.services.tasks worker --loglevel=info -P solo"

:: 5. Start FastAPI Main API
echo 🌐 Starting FastAPI Web Server...
start "FastAPI Server" cmd /k ""%PYTHON_APP_EXE%" -m app.api.main"

:: 6. Start MCP SSE Server
echo 🔌 Starting MCP SSE Server...
start "MCP Server" cmd /k ""%PYTHON_APP_EXE%" -m app.mcp.mcp_server --transport sse --port 9382"

echo.
echo ✅ All services triggered.
echo 💡 Monitor the new windows for logs.
echo.
pause
