#!/usr/bin/env python3
"""Quick WebSocket test script"""

import asyncio
import websockets
import json
import time

async def test_websocket_connection():
    """Test basic WebSocket connection to orchestrator"""
    print("🔌 Testing WebSocket connection...")
    
    try:
        # Connect to client port
        uri = "ws://localhost:9000/client"
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to orchestrator")
            
            # Send session start
            session_start = {
                "type": "session_start",
                "session_id": "test-session-123", 
                "timestamp": time.time(),
                "data": {
                    "client_type": "headset",
                    "audio_format": {
                        "sample_rate": 16000,
                        "channels": 1,
                        "bit_depth": 16
                    }
                },
                "metadata": {
                    "client_version": "1.0.0",
                    "test_mode": True
                }
            }
            
            await websocket.send(json.dumps(session_start))
            print("✅ Session start message sent")
            
            # Send test text input
            text_input = {
                "type": "text_input",
                "session_id": "test-session-123",
                "timestamp": time.time(),
                "data": {
                    "text": "Hello WebSocket world!"
                },
                "metadata": {
                    "input_method": "keyboard"
                }
            }
            
            await websocket.send(json.dumps(text_input))
            print("✅ Text input message sent")
            
            # Wait for potential responses
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                print(f"📨 Received: {response}")
            except asyncio.TimeoutError:
                print("⏰ No response received (timeout)")
            
            print("✅ Test completed successfully")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_websocket_connection())
    if success:
        print("🎉 WebSocket connection test PASSED")
    else:
        print("💥 WebSocket connection test FAILED")
