#!/bin/bash

echo "🚀 Starting Logikos with Qwen AI Model..."

# Set environment variables for Qwen
export MATHVIZ_AI_PROVIDER=ollama
export MATHVIZ_AI_MODEL='lly/qwen2.5-32b-instruct-iq3_m:latest'
export OLLAMA_HOST='http://127.0.0.1:11434'

echo "✅ Environment variables set:"
echo "   MATHVIZ_AI_PROVIDER=$MATHVIZ_AI_PROVIDER"
echo "   MATHVIZ_AI_MODEL=$MATHVIZ_AI_MODEL"
echo "   OLLAMA_HOST=$OLLAMA_HOST"

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if Ollama is running
echo "🤖 Checking Ollama server..."
if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not running. Please start it with 'ollama serve'"
    exit 1
fi

# Start the backend in background
echo "🌐 Starting MathViz backend server in background..."
cd mathviz-backend
python main.py &
BACKEND_PID=$!
echo "✅ Backend started with PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Start the frontend
echo "💻 Starting React frontend..."
cd ../mathviz-frontend
echo "🚀 Frontend will be available at: http://localhost:3000"
npm start

# Cleanup function
cleanup() {
    echo "\n🛑 Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM
