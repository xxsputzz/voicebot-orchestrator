#!/usr/bin/env python3
"""
Direct Zonos TTS WebSocket Test
Tests the Zonos TTS service specifically via WebSocket
"""

import asyncio
import json
import websockets
import time

async def test_zonos_tts_direct():
    """Test Zonos TTS service directly"""
    print("🎵 Direct Zonos TTS WebSocket Test")
    print("============================================")
    
    try:
        # Connect to orchestrator
        websocket = await websockets.connect("ws://localhost:9000")
        session_id = f"zonos_test_{int(time.time())}"
        print(f"✅ Connected - Session: {session_id}")
        
        # Send TTS request for Zonos
        tts_request = {
            "type": "tts_request",
            "session_id": session_id,
            "timestamp": time.time(),
            "data": {
                "text": "Hello! This is a test of the Zonos neural text-to-speech system. I'm testing the clean, artifact-free voice generation with WebSocket streaming.",
                "voice": "neutral"
            }
        }
        
        print("📤 Sending TTS request to Zonos...")
        print(f"🎭 Voice: neutral")
        print(f"📝 Text: '{tts_request['data']['text'][:50]}...'")
        
        await websocket.send(json.dumps(tts_request))
        
        # Wait for TTS response
        print("📥 Waiting for TTS response...")
        response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
        response_data = json.loads(response)
        
        print(f"\n🎯 Response Type: {response_data.get('type')}")
        
        if response_data.get('type') == 'audio_output':
            data = response_data.get('data', {})
            print("✅ TTS Audio Generated Successfully!")
            print(f"🎵 Format: {data.get('audio_format')}")
            print(f"🎵 Sample Rate: {data.get('sample_rate')} Hz")
            print(f"🎵 Channels: {data.get('channels')}")
            print(f"🎵 Voice: {data.get('voice')}")
            print(f"🎵 Engine: {response_data.get('metadata', {}).get('engine')}")
            
            if data.get('simulated'):
                print("⚠️  Response: Simulated audio (TTS engine not fully active)")
                print(f"🕐 Simulated duration: {data.get('duration')} seconds")
            else:
                print("🎉 Response: Real Zonos TTS audio generated!")
                if data.get('audio_data'):
                    print(f"📊 Audio data size: {len(data.get('audio_data', ''))} bytes")
                    
        elif response_data.get('type') == 'error':
            error_msg = response_data.get('data', {}).get('error', 'Unknown error')
            print(f"❌ TTS Error: {error_msg}")
            
        else:
            print(f"📦 Unexpected response: {response_data.get('type')}")
            print(f"📄 Data: {response_data.get('data', {})}")
        
        await websocket.close()
        
    except asyncio.TimeoutError:
        print("⏰ TTS request timed out")
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    print("\n🎵 Zonos TTS test complete!")

if __name__ == "__main__":
    asyncio.run(test_zonos_tts_direct())
