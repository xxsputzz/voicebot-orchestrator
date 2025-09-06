#!/usr/bin/env python3
"""
Quick test to check if STT service routing works with proper error logging
"""

import asyncio
import json
import websockets
import time
import base64

async def test_stt_routing_with_debug():
    """Test STT routing and check for detailed error info"""
    print("🔍 Testing STT Routing with Debug Info")
    print("=" * 50)
    
    try:
        websocket = await websockets.connect("ws://localhost:9000")
        session_id = f"debug_stt_{int(time.time())}"
        
        # Wait for session start
        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        print(f"✅ Connected: {json.loads(response).get('type')}")
        
        # Create a very small audio sample for testing
        print("📤 Sending small test audio...")
        
        # Create a small WAV-like audio data (just a few samples)
        dummy_audio = b'\x00\x01' * 100  # 200 bytes of alternating data
        audio_b64 = base64.b64encode(dummy_audio).decode()
        
        audio_message = {
            "type": "audio_chunk", 
            "session_id": session_id,
            "timestamp": time.time(),
            "data": {
                "audio_data": audio_b64,
                "format": "wav",
                "sample_rate": 16000,
                "channels": 1
            }
        }
        
        await websocket.send(json.dumps(audio_message))
        print("📤 Audio message sent successfully")
        
        # Wait for responses with detailed logging
        print("⏳ Waiting for responses (15 second timeout)...")
        
        response_count = 0
        for i in range(15):  # 15 second timeout
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                response_data = json.loads(response)
                response_count += 1
                
                response_type = response_data.get('type')
                session_id_resp = response_data.get('session_id', 'unknown')
                
                print(f"📨 Response {response_count}: {response_type} (session: {session_id_resp})")
                
                # Log specific response types
                if response_type == 'error':
                    error_msg = response_data.get('data', {}).get('error', 'unknown error')
                    print(f"   ❌ Error: {error_msg}")
                elif response_type in ['transcript_partial', 'transcript_final']:
                    text = response_data.get('data', {}).get('text', 'no text')
                    print(f"   🎯 Transcript: '{text}'")
                    if response_type == 'transcript_final':
                        print("   ✅ Final transcript received - STT working!")
                        break
                else:
                    # Print first 100 chars of data for other response types  
                    data_str = str(response_data.get('data', {}))
                    if len(data_str) > 100:
                        data_str = data_str[:100] + "..."
                    print(f"   📄 Data: {data_str}")
                    
            except asyncio.TimeoutError:
                print(f"   ⏳ Timeout {i+1}/15...")
                continue
                
        if response_count == 0:
            print("❌ No responses received at all")
        else:
            print(f"📊 Total responses: {response_count}")
            
        await websocket.close()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(test_stt_routing_with_debug())
