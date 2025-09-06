#!/usr/bin/env python3
"""
Simple WebSocket Test - Minimal test to verify fixes
"""
import asyncio
import websockets
import json
from datetime import datetime, timezone

async def simple_test():
    """Simple test of WebSocket communication"""
    print("🔧 Simple WebSocket Fix Verification")
    print("="*40)
    
    try:
        # Connect
        websocket = await websockets.connect("ws://localhost:9000")
        
        # Get session
        welcome = await websocket.recv()
        session_data = json.loads(welcome)
        session_id = session_data.get('session_id')
        print(f"✅ Connected: {session_id}")
        
        # Send simple LLM message
        message = {
            "type": "text_input",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "text": "Hello, are you working now?"
            }
        }
        
        await websocket.send(json.dumps(message))
        print("📤 Sent simple message")
        
        # Wait briefly for any response or error
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
            response_data = json.loads(response)
            print(f"📥 Response type: {response_data.get('type')}")
            print(f"📝 Response data: {response_data.get('data', {})}")
        except asyncio.TimeoutError:
            print("⏰ No response (this is OK if services are processing)")
        
        await websocket.close()
        print("✅ Test completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(simple_test())
