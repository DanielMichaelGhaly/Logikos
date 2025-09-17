#!/bin/bash

echo "🚀 Starting SIMPLE Qwen Math Server..."

# Kill any existing servers
pkill -f "simple_qwen_server.py" 2>/dev/null

# Activate virtual environment
source .venv/bin/activate

# Start the simple server
echo "🤖 Using Qwen model: lly/qwen2.5-32b-instruct-iq3_m:latest"
echo "🌐 Starting server at http://localhost:8000"
echo ""
echo "✨ This is a simple, direct connection to your Qwen model!"
echo "📝 Just type math problems and get AI solutions!"
echo ""

python simple_qwen_server.py