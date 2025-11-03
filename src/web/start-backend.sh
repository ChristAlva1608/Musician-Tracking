#!/bin/bash

# Musician Tracking Backend Startup Script

echo "🚀 Starting Musician Tracking Backend..."
echo ""

cd "$(dirname "$0")/backend"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 No virtual environment found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.dependencies_installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.dependencies_installed
    echo "✅ Dependencies installed"
fi

# Start the backend
echo ""
echo "✅ Starting FastAPI server on http://localhost:8000"
echo "📚 API Docs will be available at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py
