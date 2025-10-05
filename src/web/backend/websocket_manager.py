"""
WebSocket Manager for broadcasting updates
"""
from typing import List, Dict, Any
from fastapi import WebSocket
import json

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        print(f"Broadcasting to {len(self.active_connections)} clients: {message.get('type', 'unknown')}")
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
                print(f"✓ Sent to client")
            except Exception as e:
                print(f"✗ Failed to send to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

# Global instance
manager = WebSocketManager()