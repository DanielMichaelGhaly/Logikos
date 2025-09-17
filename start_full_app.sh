#!/bin/bash

echo "🚀 Starting Full Logikos App (Backend + React Frontend)"

# Kill any existing processes
echo "🧹 Cleaning up any existing servers..."
pkill -f "simple_qwen_server.py" 2>/dev/null
pkill -f "npm start" 2>/dev/null
pkill -f "react-scripts" 2>/dev/null

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Start backend in background
echo "🌐 Starting Qwen backend server..."
python simple_qwen_server.py &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID)"

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 3

# Test backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend is responding"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start React frontend
echo "💻 Starting React frontend..."
cd mathviz-frontend

# Quick npm install check
if [ ! -d "node_modules" ]; then
    echo "📦 Installing React dependencies..."
    npm install
fi

echo "🚀 Starting React app..."
echo ""
echo "🎉 Logikos Full App Starting!"
echo "📊 Backend API: http://localhost:8000"
echo "🌐 React Frontend: http://localhost:3000"
echo "🤖 Powered by Qwen 2.5 32B"
echo ""
echo "Press Ctrl+C to stop both servers"

# Start frontend (this blocks)
npm start

# Cleanup when frontend stops
echo "🛑 Shutting down backend..."
kill $BACKEND_PID 2>/dev/null
echo "✅ All servers stopped"