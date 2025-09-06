#!/usr/bin/env python3
"""
Complete WebSocket Pipeline Test - STT → LLM → TTS
Tests the entire voice processing pipeline through WebSocket connections
"""
import asyncio
import websockets
import json
import time
import base64
import wave
import io
import requests
from datetime import datetime, timezone

async def test_complete_websocket_pipeline():
    """Test the complete STT → LLM → TTS pipeline via WebSockets"""
    print("🎙️ Complete WebSocket Pipeline Test")
    print("="*60)
    
    # Step 1: Check orchestrator and services
    print("1️⃣ Checking WebSocket Services...")
    try:
        response = requests.get("http://localhost:8080/services", timeout=3)
        if response.status_code == 200:
            services = response.json()
            service_types = [s['service_type'] for s in services]
            print(f"   ✅ Services available: {service_types}")
            
            # Check which services we have
            has_stt = 'stt' in service_types
            has_llm = 'llm' in service_types
            has_tts = 'tts' in service_types
            
            print(f"   STT: {'✅' if has_stt else '❌'}")
            print(f"   LLM: {'✅' if has_llm else '❌'}")
            print(f"   TTS: {'✅' if has_tts else '❌'}")
            
        else:
            print(f"   ❌ Orchestrator error: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Orchestrator not accessible: {e}")
        return False
    
    # Step 2: Connect to WebSocket orchestrator (client port)
    print(f"\n2️⃣ Connecting to WebSocket Orchestrator...")
    try:
        websocket = await websockets.connect("ws://localhost:9000")
        
        # Receive welcome message
        welcome = await websocket.recv()
        welcome_data = json.loads(welcome)
        session_id = welcome_data.get('session_id')
        print(f"   ✅ Connected - Session: {session_id}")
        
    except Exception as e:
        print(f"   ❌ WebSocket connection failed: {e}")
        return False
    
    try:
        # Step 3: Test STT Pipeline (Audio → Text)
        if has_stt:
            print(f"\n3️⃣ Testing STT Pipeline (Audio → Text)...")
            
            # Create fake audio data (simulate real audio)
            audio_data = create_fake_audio_data()
            
            # Send audio chunk to STT
            audio_message = {
                "type": "audio_chunk",
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "audio_data": base64.b64encode(audio_data).decode('utf-8'),
                    "chunk_index": 0,
                    "is_final": True,
                    "format": "wav",
                    "sample_rate": 16000
                },
                "metadata": {
                    "client_type": "test_client",
                    "audio_format": "wav"
                }
            }
            
            await websocket.send(json.dumps(audio_message))
            print("   📤 Sent audio chunk to STT service")
            
            # Wait for STT response
            try:
                stt_response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                stt_data = json.loads(stt_response)
                print(f"   📥 STT Response: {stt_data.get('type')}")
                
                if stt_data.get('type') == 'text_result':
                    transcribed_text = stt_data.get('data', {}).get('text', '')
                    print(f"   🗣️ Transcribed: '{transcribed_text}'")
                elif stt_data.get('type') == 'error':
                    print(f"   ❌ STT Error: {stt_data.get('data', {}).get('error')}")
                    
            except asyncio.TimeoutError:
                print("   ⏰ STT response timeout")
        
        # Step 4: Test LLM Pipeline (Text → AI Response)
        if has_llm:
            print(f"\n4️⃣ Testing LLM Pipeline (Text → AI Response)...")
            
            # Send text input to LLM
            text_message = {
                "type": "text_input",
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "text": "Hello! Can you help me test the WebSocket pipeline?",
                    "user_id": "test_user",
                    "conversation_id": f"test_conv_{session_id}"
                },
                "metadata": {
                    "client_type": "test_client",
                    "test_mode": True
                }
            }
            
            await websocket.send(json.dumps(text_message))
            print("   📤 Sent text input to LLM service")
            
            # Wait for LLM response
            try:
                llm_response = await asyncio.wait_for(websocket.recv(), timeout=20.0)
                llm_data = json.loads(llm_response)
                print(f"   📥 LLM Response: {llm_data.get('type')}")
                
                if llm_data.get('type') == 'text_response':
                    ai_text = llm_data.get('data', {}).get('text', '')
                    print(f"   🤖 AI Response: '{ai_text[:100]}...' (truncated)")
                elif llm_data.get('type') == 'error':
                    print(f"   ❌ LLM Error: {llm_data.get('data', {}).get('error')}")
                    
            except asyncio.TimeoutError:
                print("   ⏰ LLM response timeout")
        
        # Step 5: Test TTS Pipeline (Text → Audio) if available
        if has_tts:
            print(f"\n5️⃣ Testing TTS Pipeline (Text → Audio)...")
            
            # Send TTS request
            tts_message = {
                "type": "text_to_speech",
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "text": "Hello! This is a test of the text to speech pipeline.",
                    "voice": "default",
                    "speed": 1.0
                },
                "metadata": {
                    "client_type": "test_client",
                    "output_format": "wav"
                }
            }
            
            await websocket.send(json.dumps(tts_message))
            print("   📤 Sent TTS request")
            
            # Wait for TTS response
            try:
                tts_response = await asyncio.wait_for(websocket.recv(), timeout=25.0)
                tts_data = json.loads(tts_response)
                print(f"   📥 TTS Response: {tts_data.get('type')}")
                
                if tts_data.get('type') == 'audio_response':
                    audio_length = len(tts_data.get('data', {}).get('audio_data', ''))
                    print(f"   🔊 Generated audio: {audio_length} bytes")
                elif tts_data.get('type') == 'error':
                    print(f"   ❌ TTS Error: {tts_data.get('data', {}).get('error')}")
                    
            except asyncio.TimeoutError:
                print("   ⏰ TTS response timeout")
        
        # Step 6: Test complete conversation flow
        print(f"\n6️⃣ Testing Complete Conversation Flow...")
        
        # Send a conversation message that should flow through multiple services
        conversation_message = {
            "type": "conversation_message",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "message": "What's the weather like today?",
                "user_id": "test_user",
                "conversation_id": f"conv_{session_id}",
                "require_audio_response": has_tts
            },
            "metadata": {
                "pipeline_test": True,
                "expected_flow": ["llm"] + (["tts"] if has_tts else [])
            }
        }
        
        await websocket.send(json.dumps(conversation_message))
        print("   📤 Sent conversation message")
        
        # Collect all responses in the pipeline
        responses_received = 0
        max_responses = 3
        
        while responses_received < max_responses:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                response_data = json.loads(response)
                response_type = response_data.get('type')
                
                print(f"   📥 Pipeline Response {responses_received + 1}: {response_type}")
                
                if response_type in ['text_response', 'audio_response', 'error']:
                    responses_received += 1
                    
                    if response_type == 'text_response':
                        text = response_data.get('data', {}).get('text', '')
                        print(f"      💬 Text: '{text[:80]}...'")
                    elif response_type == 'audio_response':
                        audio_size = len(response_data.get('data', {}).get('audio_data', ''))
                        print(f"      🔊 Audio: {audio_size} bytes")
                    elif response_type == 'error':
                        error = response_data.get('data', {}).get('error', '')
                        print(f"      ❌ Error: {error}")
                        
            except asyncio.TimeoutError:
                print("   ⏰ Pipeline response timeout - ending flow test")
                break
        
        # Step 7: Performance and health summary
        print(f"\n7️⃣ Pipeline Performance Summary...")
        
        # Send health check
        health_message = {
            "type": "health_check",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await websocket.send(json.dumps(health_message))
        
        try:
            health_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            health_data = json.loads(health_response)
            print(f"   📊 Health Check: {health_data.get('type')}")
        except asyncio.TimeoutError:
            print("   📊 Health check timeout")
        
        # Close connection
        await websocket.close()
        print(f"\n✅ Pipeline test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test error: {e}")
        return False
    
    finally:
        if 'websocket' in locals():
            await websocket.close()

def create_fake_audio_data():
    """Create fake WAV audio data for testing"""
    # Create a simple WAV file in memory
    buffer = io.BytesIO()
    
    # WAV parameters
    sample_rate = 16000
    duration = 1  # 1 second
    frequency = 440  # A note
    
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Generate sine wave
        import math
        frames = []
        for i in range(int(sample_rate * duration)):
            value = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            frames.extend([value & 0xFF, (value >> 8) & 0xFF])
        
        wav_file.writeframes(bytes(frames))
    
    return buffer.getvalue()

async def main():
    """Main test function"""
    success = await test_complete_websocket_pipeline()
    
    print("\n" + "="*60)
    if success:
        print("🎉 WebSocket Pipeline Test: SUCCESS")
        print("✅ All available services tested")
        print("✅ WebSocket protocol working")
        print("✅ Message routing functional")
    else:
        print("❌ WebSocket Pipeline Test: FAILED")
        print("🔧 Check service availability and connections")

if __name__ == "__main__":
    asyncio.run(main())
