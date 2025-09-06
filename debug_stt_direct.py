#!/usr/bin/env python3
"""
Direct STT Service Debug Test
Send a message directly to the STT service to test if it responds
"""

import asyncio
import websockets
import json
import base64
import wave
import io
from datetime import datetime

async def test_stt_direct():
    """Test STT service directly"""
    print("🧪 Direct STT Service Test")
    print("=" * 50)
    
    # Create a simple test audio (silence)
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    silence_samples = int(sample_rate * duration)
    audio_data = bytes([0] * (silence_samples * 2))  # 16-bit silence
    
    # Create WAV format
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    
    wav_bytes = wav_buffer.getvalue()
    audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
    
    print(f"📤 Created test audio: {len(wav_bytes)} bytes WAV, {len(audio_base64)} chars base64")
    
    try:
        # Connect to orchestrator
        uri = "ws://localhost:9001"
        print(f"🔗 Connecting to orchestrator at {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to orchestrator")
            
            # Send test message
            session_id = "direct_test_123"
            test_message = {
                "type": "audio_chunk",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "data": {
                    "audio_data": audio_base64,
                    "chunk_index": 0,
                    "is_final": True
                },
                "metadata": {
                    "test": True,
                    "direct": True
                }
            }
            
            print(f"📤 Sending audio_chunk message...")
            await websocket.send(json.dumps(test_message))
            print("✅ Message sent, waiting for response...")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                response_data = json.loads(response)
                print(f"📥 Received response: {response_data.get('type', 'unknown')}")
                print(f"   Data: {response_data.get('data', {})}")
                
                if response_data.get('type') in ['transcript_partial', 'transcript_final']:
                    text = response_data.get('data', {}).get('text', 'NO TEXT')
                    print(f"🎯 TRANSCRIPT: '{text}'")
                    print("✅ STT Service is working!")
                else:
                    print(f"❌ Unexpected response type: {response_data.get('type')}")
                    
            except asyncio.TimeoutError:
                print("❌ No response received within 15 seconds")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_stt_direct())
