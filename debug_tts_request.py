#!/usr/bin/env python3
"""
Simple TTS WebSocket Debug Test
Sends a basic TTS request and shows exactly what happens
"""

import asyncio
import json
import websockets
import time

async def debug_tts_request():
    """Debug TTS request processing"""
    print("🔍 TTS WebSocket Debug Test")
    print("===============================")
    
    try:
        # Connect to orchestrator
        websocket = await websockets.connect("ws://localhost:9000")
        session_id = f"debug_tts_{int(time.time())}"
        print(f"✅ Connected - Session: {session_id}")
        
        # Send a very simple TTS request
        simple_tts = {
            "type": "tts_request",
            "session_id": session_id,
            "timestamp": time.time(),
            "data": {
                "text": "Hello world",
                "voice": "neutral"
            }
        }
        
        print("📤 Sending simple TTS request...")
        print(f"📋 Message: {json.dumps(simple_tts, indent=2)}")
        
        await websocket.send(json.dumps(simple_tts))
        
        # Wait for any response
        print("📥 Waiting for response...")
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            response_data = json.loads(response)
            
            print(f"\n📨 Received Response:")
            print(f"🎯 Type: {response_data.get('type')}")
            print(f"📄 Full Response: {json.dumps(response_data, indent=2)}")
            
        except asyncio.TimeoutError:
            print("⏰ No response received within timeout")
        
        await websocket.close()
        
    except Exception as e:
        print(f"❌ Debug error: {e}")
    
    print("\n🔍 Debug test complete!")

if __name__ == "__main__":
    asyncio.run(debug_tts_request())
