#!/usr/bin/env python3
"""
Direct WebSocket Routing Test - Test if orchestrator routes messages to services
"""
import asyncio
import websockets
import json
from datetime import datetime, timezone

async def test_direct_routing():
    """Test direct message routing to services"""
    print("🔀 Direct WebSocket Routing Test")
    print("="*50)
    
    print("📡 Connecting to orchestrator client port (9000)...")
    
    try:
        # Connect as client
        websocket = await websockets.connect("ws://localhost:9000")
        
        # Get welcome message
        welcome = await websocket.recv()
        welcome_data = json.loads(welcome)
        session_id = welcome_data.get('session_id')
        print(f"✅ Connected - Session: {session_id}")
        
        # Test 1: Send simple text message to LLM
        print("\n1️⃣ Testing LLM Routing...")
        llm_message = {
            "type": "text_input",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "text": "Hello LLM service, can you hear me?",
                "user_id": "test_user"
            }
        }
        
        await websocket.send(json.dumps(llm_message))
        print("   📤 Sent LLM message")
        
        # Wait for response
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"   📥 LLM Response: {json.loads(response).get('type', 'unknown')}")
        except asyncio.TimeoutError:
            print("   ⏰ LLM timeout - checking if message reached service")
        
        # Test 2: Send audio chunk to STT
        print("\n2️⃣ Testing STT Routing...")
        stt_message = {
            "type": "audio_chunk", 
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "audio_data": "fake_audio_base64",
                "chunk_index": 0,
                "is_final": True
            }
        }
        
        await websocket.send(json.dumps(stt_message))
        print("   📤 Sent STT message")
        
        # Wait for response
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"   📥 STT Response: {json.loads(response).get('type', 'unknown')}")
        except asyncio.TimeoutError:
            print("   ⏰ STT timeout - checking if message reached service")
        
        # Test 3: Send heartbeat
        print("\n3️⃣ Testing Heartbeat...")
        heartbeat = {
            "type": "heartbeat",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await websocket.send(json.dumps(heartbeat))
        print("   💓 Sent heartbeat")
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
            print(f"   📥 Heartbeat Response: {json.loads(response).get('type', 'unknown')}")
        except asyncio.TimeoutError:
            print("   ⏰ Heartbeat timeout")
        
        await websocket.close()
        print("\n✅ Routing test completed")
        
    except Exception as e:
        print(f"❌ Routing test error: {e}")

if __name__ == "__main__":
    asyncio.run(test_direct_routing())
