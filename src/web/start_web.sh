#!/bin/bash

echo "🚀 Starting Musician Tracking Web Interface"
echo "=========================================="

# Check if we're in the correct directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: Please run this script from the src/web directory"
    exit 1
fi

# Check if FastAPI is installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "⚠️  FastAPI not installed. Installing requirements..."
    pip install -r backend/requirements.txt
fi

# Start the backend server
echo ""
echo "📡 Starting FastAPI backend server..."
echo "API will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo ""

cd backend && python3 main.py