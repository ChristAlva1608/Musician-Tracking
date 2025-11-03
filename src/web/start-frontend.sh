#!/bin/bash

# Musician Tracking Frontend Startup Script

echo "🚀 Starting Musician Tracking Frontend..."
echo ""

cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Start the frontend
echo ""
echo "✅ Starting React development server on http://localhost:3000"
echo "🌐 Your browser will open automatically"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

npm start
