#!/usr/bin/env python3
"""
Direct Health Check Test
Tests the health_check message handling directly
"""

import asyncio
import json
import websockets
import time

async def test_health_check():
    """Test health check directly"""
    print("🔍 Direct Health Check Test")
    print("=============================")
    
    try:
        # Connect to orchestrator
        websocket = await websockets.connect("ws://localhost:9000")
        session_id = f"health_test_{int(time.time())}"
        print(f"✅ Connected - Session: {session_id}")
        
        # Send health check
        health_check = {
            "type": "health_check",
            "session_id": session_id,
            "timestamp": time.time(),
            "data": {}
        }
        print(f"📤 Sending health check: {health_check}")
        await websocket.send(json.dumps(health_check))
        
        # Wait for response
        print("📥 Waiting for response...")
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            response_data = json.loads(response)
            print(f"📨 Response received:")
            print(f"🎯 Type: {response_data.get('type')}")
            print(f"📄 Full Response: {json.dumps(response_data, indent=2)}")
            
        except asyncio.TimeoutError:
            print("⏰ Health check response timeout")
        except Exception as e:
            print(f"❌ Error receiving response: {e}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    
    print("🔍 Health check test complete!")

if __name__ == "__main__":
    asyncio.run(test_health_check())
