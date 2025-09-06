#!/usr/bin/env python3
"""
Working WebSocket Pipeline Test - Test with proper message formats
"""
import asyncio
import websockets
import json
from datetime import datetime, timezone

async def test_working_pipeline():
    """Test the working WebSocket pipeline"""
    print("🎉 Working WebSocket Pipeline Test")
    print("="*50)
    
    print("📡 Connecting to orchestrator...")
    websocket = await websockets.connect("ws://localhost:9000")
    
    # Get session
    welcome = await websocket.recv()
    session_data = json.loads(welcome)
    session_id = session_data.get('session_id')
    print(f"✅ Connected - Session: {session_id}")
    
    # Test LLM with proper format
    print("\n🤖 Testing LLM Service...")
    llm_message = {
        "type": "text_input",
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": {
            "text": "Hello! This is a WebSocket pipeline test. Please respond with a short message.",
            "user_id": "test_user"
        },
        "metadata": {
            "test": True,
            "client_type": "test_client"
        }
    }
    
    await websocket.send(json.dumps(llm_message))
    print("📤 Sent LLM message")
    
    # Collect streaming response
    print("📥 Collecting LLM streaming response...")
    response_parts = []
    timeout_count = 0
    
    while timeout_count < 3:
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            response_data = json.loads(response)
            response_type = response_data.get('type', 'unknown')
            
            if response_type in ['text_response', 'llm_token', 'llm_stream_complete']:
                if response_type == 'text_response':
                    text = response_data.get('data', {}).get('text', '')
                    response_parts.append(text)
                    print(f"✅ LLM Response: '{text[:60]}...'")
                elif response_type == 'llm_token':
                    token = response_data.get('data', {}).get('token', '')
                    response_parts.append(token)
                elif response_type == 'llm_stream_complete':
                    print("✅ LLM streaming completed")
                    break
            else:
                print(f"📦 Other message: {response_type}")
                
        except asyncio.TimeoutError:
            timeout_count += 1
            if timeout_count >= 3:
                print("⏰ LLM response collection timeout")
                break
    
    if response_parts:
        full_response = ''.join(response_parts)
        print(f"🎯 Complete LLM Response: '{full_response}'")
    
    await websocket.close()
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_working_pipeline())
