#!/usr/bin/env python3
"""
FastAPI Backend for Musician Tracking Web Interface
"""

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
import sys
import asyncio
import json
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import websockets
from datetime import datetime

# Add project src directory to path
# From backend/main.py -> backend -> web -> src
backend_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.dirname(backend_dir)
src_root = os.path.dirname(web_dir)
sys.path.insert(0, src_root)

from detect_v2 import DetectorV2
from integrated_video_processor import IntegratedVideoProcessor
from routers import processing, config, files, database
from websocket_manager import manager

app = FastAPI(
    title="Musician Tracking API",
    description="Web interface for musician posture and gesture tracking system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(processing.router, prefix="/api/processing", tags=["processing"])
app.include_router(config.router, prefix="/api/config", tags=["config"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(database.router, prefix="/api/database", tags=["database"])

# Global state for WebSocket connections and processing status
class AppState:
    def __init__(self):
        self.websocket_connections: List[WebSocket] = []
        self.processing_status: Dict[str, Any] = {}
        self.current_jobs: Dict[str, Any] = {}

app_state = AppState()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    app_state.websocket_connections.append(websocket)

    try:
        # Send current status
        await websocket.send_json({
            "type": "status_update",
            "data": app_state.processing_status
        })

        # Keep connection alive
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in app_state.websocket_connections:
            app_state.websocket_connections.remove(websocket)

async def broadcast_update(message: Dict[str, Any]):
    """Broadcast update to all connected WebSocket clients"""
    if app_state.websocket_connections:
        disconnected = []
        for websocket in app_state.websocket_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)

        # Remove disconnected clients
        for ws in disconnected:
            app_state.websocket_connections.remove(ws)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Musician Tracking API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/status")
async def get_status():
    """Get current processing status"""
    return {
        "processing_status": app_state.processing_status,
        "active_jobs": len(app_state.current_jobs),
        "connected_clients": len(app_state.websocket_connections)
    }

# Serve React build files (when built)
if os.path.exists("../frontend/build"):
    app.mount("/", StaticFiles(directory="../frontend/build", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )