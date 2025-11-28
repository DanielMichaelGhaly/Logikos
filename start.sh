#!/bin/bash

echo "🚀 Starting Logikos Mathematical Problem Solver"
echo "🎨 Now with Gnuplot Contour Mapping!"

# Start Ollama if not running
echo "🤖 Checking Ollama service..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "🚀 Starting Ollama service..."
    ollama serve &
    sleep 3
    echo "✅ Ollama started"
else
    echo "✅ Ollama is already running"
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if we want to run the simple workflow or full app
if [ "$1" = "--full-app" ]; then
    echo "🌐 Starting full web application..."

    # Check if backend exists
    if [ -d "mathviz-backend" ]; then
        # Start backend in background
        echo "🔗 Starting backend server..."
        cd mathviz-backend
        python main.py &
        BACKEND_PID=$!
        echo "✅ Backend started (PID: $BACKEND_PID)"

        # Wait for backend to start
        sleep 3

        # Start frontend
        echo "💻 Starting React frontend..."
        cd ../mathviz-frontend
        echo "🚀 Frontend will be available at: http://localhost:3000"
        echo "🎯 Try these contour examples: 'x^2 + y^2', 'sin(x)*cos(y)'"
        npm start

        # Cleanup when frontend stops
        echo "🛑 Shutting down backend..."
        kill $BACKEND_PID 2>/dev/null
    else
        echo "❌ Backend directory not found. Full app mode not available."
        exit 1
    fi
else
    echo "📝 Command-line mathematical problem solver ready!"
    echo ""
    echo "✨ Usage examples:"
    echo "   python run_workflow.py \"solve 2x+5=0\""
    echo "   python run_workflow.py \"derivative of x^2+3x\""
    echo "   python run_workflow.py \"find roots of x^2-4\""
    echo ""
    echo "🤖 AI features enabled with Nemotron model"
    echo "📊 Use --verbose for detailed output"
    echo "💾 Use --save-html or --save-json to save results"
    echo "🚫 Use --no-ai flag to disable AI features"
    echo ""
fi