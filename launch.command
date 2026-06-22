#!/bin/bash

# ── SQL Agent Launcher ─────────────────────────────────────────────────────
# Double-click this file to start both the Football and NBA apps.

BASE="$(cd "$(dirname "$0")" && pwd)"

# Load API key from .env
if [ -f "$BASE/.env" ]; then
    export $(grep -v '^#' "$BASE/.env" | xargs)
else
    echo "ERROR: .env file not found at $BASE/.env"
    echo "Create it with: DEEPSEEK_API_KEY=your-key-here"
    exit 1
fi

# Use the isolated virtualenv (pinned deps, no global-env pollution).
if [ -f "$BASE/.venv/bin/activate" ]; then
    source "$BASE/.venv/bin/activate"
else
    echo "ERROR: .venv not found. Create it with:"
    echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo "========================================"
echo "  ⚽ 🏀  SQL Agent Launcher"
echo "========================================"
echo ""

# Kill anything already on these ports
lsof -ti :8501 | xargs kill -9 2>/dev/null
lsof -ti :8502 | xargs kill -9 2>/dev/null

# Disable Streamlit's file watcher — its module rescan imports a broken optional
# transformers/torchvision path and crashes the app on any edit while running.
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

echo "Starting Football Agent on http://localhost:8501 ..."
cd "$BASE/Football Agent"
LANGSMITH_PROJECT="Football SQL Agent" streamlit run app.py --server.port 8501 --server.headless true &
FOOTBALL_PID=$!

echo "Starting NBA Agent on http://localhost:8502 ..."
cd "$BASE/Nba Agent 2"
LANGSMITH_PROJECT="NBA Stats Explorer" streamlit run app.py --server.port 8502 --server.headless true &
NBA_PID=$!

# Give servers a moment to start, then open both in browser
sleep 3
open http://localhost:8501
open http://localhost:8502

echo ""
echo "========================================"
echo "  Both apps are running!"
echo "  Football → http://localhost:8501"
echo "  NBA      → http://localhost:8502"
echo ""
echo "  This terminal is now free — you can"
echo "  launch other apps normally."
echo "  To stop Statline: kill $FOOTBALL_PID $NBA_PID"
echo "========================================"

# Do NOT wait — let the launcher exit so this terminal stays free for other portfolio apps.
# The Streamlit processes are already backgrounded and will keep running independently.
