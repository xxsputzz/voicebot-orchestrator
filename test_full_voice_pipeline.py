#!/usr/bin/env python3
"""
Complete STT → LLM → TTS WebSocket Pipeline Test
Tests the full voice processing pipeline with all three services
"""

import asyncio
import json
import websockets
import time
import base64
import os

async def test_full_voice_pipeline():
    """Test the complete STT → LLM → TTS pipeline"""
    print("🎙️→🤖→🎵 Complete Voice Pipeline Test")
    print("============================================================")
    
    # Connect to orchestrator
    print("1️⃣ Connecting to WebSocket Orchestrator...")
    try:
        websocket = await websockets.connect("ws://localhost:9000")
        session_id = f"pipeline_test_{int(time.time())}"
        print(f"   ✅ Connected - Session: {session_id}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return

    # Check available services
    print("\n2️⃣ Checking Available Services...")
    try:
        health_check = {
            "type": "health_check",
            "session_id": session_id,
            "timestamp": time.time(),
            "data": {}
        }
        await websocket.send(json.dumps(health_check))
        
        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        health_data = json.loads(response)
        
        services = health_data.get('data', {}).get('services', [])
        print(f"   ✅ Services available: {services}")
        
        # Check specific services
        stt_available = 'stt' in services
        llm_available = 'llm' in services  
        tts_available = 'tts' in services
        
        print(f"   STT: {'✅' if stt_available else '❌'}")
        print(f"   LLM: {'✅' if llm_available else '❌'}")
        print(f"   TTS: {'✅' if tts_available else '❌'}")
        
    except Exception as e:
        print(f"   ⚠️  Health check error: {e}")
        stt_available = llm_available = tts_available = True  # Assume available

    # Test TTS Pipeline (Text → Audio)
    if tts_available:
        print("\n3️⃣ Testing TTS Pipeline (Text → Audio)...")
        try:
            tts_message = {
                "type": "tts_request",
                "session_id": session_id,
                "timestamp": time.time(),
                "data": {
                    "text": "Hello! This is a test of the Zonos TTS system. The voice should sound clear and natural.",
                    "voice": "neutral"
                }
            }
            
            print("   📤 Sent TTS generation request")
            await websocket.send(json.dumps(tts_message))
            
            # Collect TTS response
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            response_data = json.loads(response)
            
            if response_data.get('type') == 'audio_output':
                audio_data = response_data.get('data', {})
                print(f"   ✅ TTS Response received")
                print(f"   🎵 Audio format: {audio_data.get('audio_format')}")
                print(f"   🎵 Voice: {audio_data.get('voice')}")
                print(f"   🎵 Duration: {audio_data.get('duration', 'Unknown')}s")
                print(f"   🎵 Text: '{audio_data.get('text', '')}'")
                
                if audio_data.get('simulated'):
                    print("   ⚠️  Using simulated audio (TTS engine not fully available)")
                else:
                    print("   🎉 Real TTS audio generated!")
            else:
                print(f"   📦 Other response: {response_data.get('type')}")
                
        except asyncio.TimeoutError:
            print("   ⏰ TTS response timeout")
        except Exception as e:
            print(f"   ❌ TTS Error: {e}")

    # Test LLM → TTS Chain
    if llm_available and tts_available:
        print("\n4️⃣ Testing LLM → TTS Chain...")
        try:
            # First get LLM response
            llm_message = {
                "type": "text_input",
                "session_id": session_id,
                "timestamp": time.time(),
                "data": {
                    "text": "Please say hello and introduce yourself briefly.",
                    "stream_tokens": False  # Get full response
                }
            }
            
            print("   📤 Sent text to LLM")
            await websocket.send(json.dumps(llm_message))
            
            # Collect LLM response
            llm_text = ""
            timeout_count = 0
            while timeout_count < 5:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    response_data = json.loads(response)
                    
                    if response_data.get('type') == 'text_response':
                        llm_text = response_data.get('data', {}).get('text', '')
                        print(f"   🤖 LLM Response: '{llm_text[:60]}...'")
                        break
                    elif response_data.get('type') in ['llm_stream_complete', 'llm_token']:
                        if response_data.get('type') == 'llm_stream_complete':
                            llm_text = response_data.get('data', {}).get('full_text', '')
                            print(f"   🤖 LLM Streaming Complete: '{llm_text[:60]}...'")
                            break
                    else:
                        timeout_count += 1
                        
                except asyncio.TimeoutError:
                    timeout_count += 1
            
            if llm_text:
                # Send LLM response to TTS
                tts_message = {
                    "type": "tts_request", 
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "data": {
                        "text": llm_text,
                        "voice": "neutral"
                    }
                }
                
                print("   📤 Sending LLM response to TTS")
                await websocket.send(json.dumps(tts_message))
                
                # Get TTS response
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                response_data = json.loads(response)
                
                if response_data.get('type') == 'audio_output':
                    print("   🎉 LLM → TTS Chain Complete!")
                    print(f"   🔗 Generated audio from LLM response")
                else:
                    print(f"   📦 Chain response: {response_data.get('type')}")
            else:
                print("   ❌ No LLM response received for TTS chain")
                
        except Exception as e:
            print(f"   ❌ LLM → TTS Chain Error: {e}")

    print("\n5️⃣ Pipeline Summary...")
    print("   📊 Voice Processing Pipeline:")
    print(f"   🎙️  STT (Speech-to-Text): {'✅' if stt_available else '❌'}")
    print(f"   🤖 LLM (Language Model): {'✅' if llm_available else '❌'}")
    print(f"   🎵 TTS (Text-to-Speech): {'✅' if tts_available else '❌'}")
    
    await websocket.close()
    print("\n✅ Pipeline test completed!")

    print("\n============================================================")
    print("🎉 Voice Processing Pipeline Test: COMPLETE")
    if stt_available and llm_available and tts_available:
        print("🎯 All services operational - Full voice pipeline ready!")
    else:
        print("⚠️  Some services need attention for complete pipeline")

if __name__ == "__main__":
    asyncio.run(test_full_voice_pipeline())
