#!/usr/bin/env python3
"""
Test Zonos TTS WebSocket Integration
"""
import asyncio
import json
import websockets
import time

async def test_tts_pipeline():
    """Test the complete TTS pipeline through WebSocket"""
    print("🎵 Testing Zonos TTS WebSocket Integration")
    print("=" * 50)
    
    try:
        # Connect to orchestrator
        print("📡 Connecting to WebSocket orchestrator...")
        websocket = await websockets.connect("ws://localhost:9000")
        print("✅ Connected!")
        
        # Send TTS request
        print("\n🎤 Sending TTS request...")
        tts_request = {
            "type": "tts_request",
            "session_id": f"test_tts_{int(time.time())}",
            "data": {
                "text": "Hello! This is a test of the Zonos TTS WebSocket integration. The voice should sound clear and natural.",
                "voice": "aria",
                "audio_format": "wav"
            }
        }
        
        await websocket.send(json.dumps(tts_request))
        print("📤 TTS request sent")
        
        # Wait for response
        print("\n📥 Waiting for TTS response...")
        timeout_count = 0
        
        while timeout_count < 5:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                response_data = json.loads(response)
                response_type = response_data.get('type', 'unknown')
                
                print(f"📦 Received: {response_type}")
                
                if response_type == 'tts_response':
                    audio_data = response_data.get('data', {})
                    audio_length = len(audio_data.get('audio_base64', ''))
                    print(f"✅ TTS Response received!")
                    print(f"   Voice: {audio_data.get('voice', 'unknown')}")
                    print(f"   Audio format: {audio_data.get('audio_format', 'unknown')}")
                    print(f"   Audio size: {audio_length} characters (base64)")
                    print(f"   Text: '{audio_data.get('text', '')[:50]}...'")
                    break
                elif response_type == 'error':
                    error_msg = response_data.get('data', {}).get('error', 'Unknown error')
                    print(f"❌ TTS Error: {error_msg}")
                    break
                
            except asyncio.TimeoutError:
                timeout_count += 1
                print(f"⏰ Timeout {timeout_count}/5...")
        
        await websocket.close()
        print("\n✅ TTS test completed!")
        
    except Exception as e:
        print(f"💥 Test error: {e}")

if __name__ == "__main__":
    asyncio.run(test_tts_pipeline())
