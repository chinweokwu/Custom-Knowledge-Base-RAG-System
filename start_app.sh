#!/bin/bash

# --- Neural Brain Self-Healing Startup ---
# This script ensures the environment is healthy before launching services.

mkdir -p logs
set -e # Exit on error

echo "🧠 Initializing Neural Brain Health Check..."

# 1. Activate Virtual Environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual Environment Activated."
else
    echo "⚠️  WARNING: venv not found. Running with system python."
fi

# 2. Run the Doctor (Self-Healing Mode)
python3 scripts/doctor.py --fix

echo "🚀 Starting AI Knowledge Based System..."

# 3. Environment Config
export PYTHONPATH=$PYTHONPATH:.
export HF_HOME=./models_cache
export LOG_FILE_PATH=./app.log

# 4. Process Cleanup (Optional: uncomment to force clean start)
# echo "🧹 Cleaning up stale processes..."
# kill $(cat logs/*.pid 2>/dev/null) 2>/dev/null || true

# 5. Start Redis (Safe Start)
if ! pgrep -x "redis-server" > /dev/null; then
    echo "📡 Starting Redis..."
    redis-server --daemonize yes || sudo service redis-server start
fi

# 6. Start Celery Worker (The Muscles)
echo "⚙️  Starting Celery Worker..."
python3 -m celery -A app.services.tasks worker --loglevel=info --concurrency=4 > logs/worker.log 2>&1 &
echo $! > logs/worker.pid

# 7. Start FastAPI (The Face)
echo "🌐 Starting FastAPI Web Server..."
python3 -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload > logs/api.log 2>&1 &
echo $! > logs/api.pid

# 8. Start MCP SSE (The Bridge)
echo "🔌 Starting MCP SSE Server..."
python3 -m app.mcp.mcp_server --transport sse --port 9382 > logs/mcp.log 2>&1 &
echo $! > logs/mcp.pid

echo ""
echo "✨ SYSTEM ONLINE ✨"
echo "------------------------------------------------"
echo "🌐 Dashboard: http://localhost:8000/static/index.html"
echo "🌐 API Docs:  http://localhost:8000/docs"
echo "📝 Logs:       tail -f logs/api.log"
echo "------------------------------------------------"
echo "💡 Run 'kill \$(cat logs/*.pid)' to stop all services."
