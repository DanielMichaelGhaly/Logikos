#!/bin/bash

echo "🚀 Starting Logikos with Qwen AI..."

# Set environment variables for Qwen
export MATHVIZ_AI_PROVIDER=ollama
export MATHVIZ_AI_MODEL='lly/qwen2.5-32b-instruct-iq3_m:latest'
export OLLAMA_HOST='http://127.0.0.1:11434'

# Activate virtual environment
source .venv/bin/activate

# Start backend in background
echo "🌐 Starting backend server..."
cd mathviz-backend
python main.py &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID)"

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Start frontend
echo "💻 Starting frontend..."
cd ../mathviz-frontend
echo ""
echo "🎉 Logikos is starting!"
echo "📊 Backend API: http://localhost:8000"
echo "🌐 Frontend UI: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Start frontend (this will block)
npm start

# Cleanup when frontend stops
echo "🛑 Shutting down backend..."
kill $BACKEND_PID 2>/dev/null