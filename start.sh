#!/bin/bash
# Start the Ollama RAG Chatbot web server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -f ".venv/bin/python" ]; then
    echo "❌ Virtual environment not found. Run 'python3 -m venv .venv' first."
    exit 1
fi

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 8000 is already in use."
    pid=$(lsof -Pi :8000 -sTCP:LISTEN -t)
    echo "   Running process: PID $pid"
    echo "   Kill with: kill $pid"
    exit 1
fi

# Start the server
echo "🚀 Starting Ollama RAG Chatbot..."
.venv/bin/python main.py web &
SERVER_PID=$!

# Give it a moment to start
sleep 2

# Check if it started successfully
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "✅ Server started (PID: $SERVER_PID)"
    echo "   Access at: http://127.0.0.1:8000"
    echo "   Chat: http://127.0.0.1:8000"
    echo "   Manage: http://127.0.0.1:8000/manage"
    echo ""
    echo "To stop the server: ./stop.sh"
else
    echo "❌ Failed to start server. Check logs above."
    exit 1
fi
