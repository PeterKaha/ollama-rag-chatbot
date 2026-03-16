#!/bin/bash
# Stop the Ollama RAG Chatbot web server

set -e

echo "🛑 Stopping Ollama RAG Chatbot..."

# Find and kill the process
pids=$(pgrep -f "python main.py web" || true)

if [ -z "$pids" ]; then
    echo "ℹ️  No running server found."
    exit 0
fi

# Kill the process(es)
killed=0
for pid in $pids; do
    echo "   Killing process: PID $pid"
    kill $pid || true 2>/dev/null
    killed=$((killed + 1))
done

# Wait a moment and check
sleep 1

# Double-check that it's really gone
remaining=$(pgrep -f "python main.py web" || true)
if [ -n "$remaining" ]; then
    echo "⚠️  Process still running, forcing kill..."
    for pid in $remaining; do
        kill -9 $pid || true 2>/dev/null
    done
    sleep 1
fi

# Final check
if pgrep -f "python main.py web" >/dev/null 2>&1; then
    echo "❌ Failed to stop server."
    exit 1
else
    echo "✅ Server stopped."
fi
