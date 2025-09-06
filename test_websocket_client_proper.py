#!/usr/bin/env python3
"""
Proper WebSocket Client Test - Demonstrates correct client registration and protocol
"""
import asyncio
import websockets
import json
import time
from datetime import datetime, timezone

async def test_proper_websocket_client():
    """Test WebSocket client with proper protocol"""
    print("🔌 WebSocket Client Protocol Test")
    print("="*50)
    
    print("📡 Connecting to WebSocket Orchestrator (Client Port 9000)...")
    
    try:
        # Step 1: Connect to CLIENT port (9000) - NOT service port (9001)
        websocket = await websockets.connect("ws://localhost:9000")
        print("✅ Connected to orchestrator")
        
        # Step 2: Receive welcome message (orchestrator sends this automatically)
        welcome = await websocket.recv()
        welcome_data = json.loads(welcome)
        print(f"📥 Welcome message: {welcome_data}")
        
        session_id = welcome_data.get('session_id')
        print(f"🆔 Assigned session ID: {session_id}")
        
        # Step 3: Send heartbeat (optional but good practice)
        heartbeat = {
            "type": "heartbeat",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await websocket.send(json.dumps(heartbeat))
        print("💓 Sent heartbeat")
        
        # Step 4: Receive heartbeat response
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            heartbeat_response = json.loads(response)
            print(f"📥 Heartbeat response: {heartbeat_response.get('type')}")
        except asyncio.TimeoutError:
            print("⏰ Heartbeat response timeout (this is OK)")
        
        # Step 5: Send text input (conversation message)
        text_message = {
            "type": "text_input",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "text": "Hello! Can you hear me?",
                "user_id": "test_user"
            },
            "metadata": {
                "client_type": "headset",
                "test_mode": True
            }
        }
        
        await websocket.send(json.dumps(text_message))
        print("💬 Sent text input message")
        
        # Step 6: Wait for LLM response
        print("⏳ Waiting for LLM response...")
        try:
            llm_response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
            llm_data = json.loads(llm_response)
            print(f"🤖 LLM Response: {llm_data}")
        except asyncio.TimeoutError:
            print("⏰ LLM response timeout (services might not be connected)")
        
        # Step 7: Test audio chunk sending (simulate STT)
        audio_chunk = {
            "type": "audio_chunk",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "audio_data": "fake_audio_bytes_base64",
                "chunk_index": 0,
                "is_final": True
            },
            "metadata": {
                "format": "wav",
                "sample_rate": 16000
            }
        }
        
        await websocket.send(json.dumps(audio_chunk))
        print("🎤 Sent audio chunk")
        
        # Step 8: Wait for STT response
        try:
            stt_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            stt_data = json.loads(stt_response)
            print(f"🗣️ STT Response: {stt_data}")
        except asyncio.TimeoutError:
            print("⏰ STT response timeout")
        
        # Step 9: Close gracefully
        await websocket.close()
        print("✅ WebSocket connection closed successfully")
        
    except websockets.exceptions.ConnectionClosed as e:
        print(f"🔌 Connection closed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

async def test_wrong_port():
    """Demonstrate what happens when connecting to wrong port"""
    print("\n🚫 Testing Wrong Port Connection (Service Port 9001)")
    print("="*50)
    
    try:
        # Try connecting to SERVICE port (this should fail)
        websocket = await websockets.connect("ws://localhost:9001")
        print("⚠️ Connected to service port (unexpected)")
        
        # Try to send a client message (this will trigger error 4000)
        test_message = {
            "type": "text_input",
            "session_id": "fake_session",
            "data": {"text": "Hello"}
        }
        
        await websocket.send(json.dumps(test_message))
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"📥 Unexpected response: {response}")
        except asyncio.TimeoutError:
            print("⏰ No response (connection likely closed)")
            
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"❌ Connection closed with error: {e.code} - {e.reason}")
        if e.code == 4000:
            print("💡 This is expected! Port 9001 is for services, not clients")
    except Exception as e:
        print(f"❌ Error: {e}")

async def main():
    """Main test function"""
    await test_proper_websocket_client()
    await test_wrong_port()
    
    print("\n📋 Summary:")
    print("✅ Clients should connect to port 9000")
    print("✅ Services should connect to port 9001") 
    print("✅ No registration needed for clients - orchestrator assigns session ID")
    print("✅ Services must register with proper service_registration message")

if __name__ == "__main__":
    asyncio.run(main())
