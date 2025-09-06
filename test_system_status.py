#!/usr/bin/env python3
"""
Complete System Status Check
Checks orchestrator, service registration, and message routing
"""

import asyncio
import json
import websockets
import time

async def check_system_status():
    """Check complete system status"""
    print("🔍 Complete System Status Check")
    print("=================================")
    
    try:
        # Connect to orchestrator
        websocket = await websockets.connect("ws://localhost:9000")
        session_id = f"status_check_{int(time.time())}"
        print(f"✅ Connected to orchestrator - Session: {session_id}")
        
        # 1. Check service health
        print("\n1️⃣ Checking Service Health...")
        health_check = {
            "type": "health_check",
            "session_id": session_id,
            "timestamp": time.time(),
            "data": {}
        }
        await websocket.send(json.dumps(health_check))
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            health_data = json.loads(response)
            services = health_data.get('data', {}).get('services', [])
            print(f"   📋 Registered services: {services}")
            
            # Check specific services
            stt_available = 'stt' in services
            llm_available = 'llm' in services  
            tts_available = 'tts' in services
            
            print(f"   STT: {'✅' if stt_available else '❌'}")
            print(f"   LLM: {'✅' if llm_available else '❌'}")
            print(f"   TTS: {'✅' if tts_available else '❌'}")
            
        except Exception as e:
            print(f"   ❌ Health check failed: {e}")
            stt_available = llm_available = tts_available = False
        
        # 2. Test LLM (known to work)
        print("\n2️⃣ Testing LLM (Control Test)...")
        llm_test = {
            "type": "text_input",
            "session_id": session_id,
            "timestamp": time.time(),
            "data": {
                "text": "Say hello briefly",
                "stream_tokens": False
            }
        }
        await websocket.send(json.dumps(llm_test))
        
        llm_worked = False
        timeout_count = 0
        while timeout_count < 3:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                response_data = json.loads(response)
                
                if response_data.get('type') in ['text_response', 'llm_stream_complete']:
                    print(f"   ✅ LLM working: {response_data.get('type')}")
                    llm_worked = True
                    break
                elif response_data.get('type') == 'llm_token':
                    print(f"   🔄 LLM streaming: receiving tokens...")
                else:
                    print(f"   📦 LLM response: {response_data.get('type')}")
                    
            except asyncio.TimeoutError:
                timeout_count += 1
        
        if not llm_worked:
            print("   ❌ LLM test failed")
        
        # 3. Test TTS
        if tts_available:
            print("\n3️⃣ Testing TTS...")
            tts_test = {
                "type": "tts_request",
                "session_id": session_id,
                "timestamp": time.time(),
                "data": {
                    "text": "Testing TTS system",
                    "voice": "neutral"
                }
            }
            await websocket.send(json.dumps(tts_test))
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=8.0)
                response_data = json.loads(response)
                
                print(f"   🎯 TTS Response: {response_data.get('type')}")
                
                if response_data.get('type') == 'audio_output':
                    data = response_data.get('data', {})
                    print(f"   ✅ TTS SUCCESS!")
                    print(f"   🎵 Format: {data.get('audio_format')}")
                    print(f"   🎵 Size: {data.get('audio_size')} bytes")
                    print(f"   🎵 Voice: {data.get('voice')}")
                    print(f"   🎵 Engine: {response_data.get('metadata', {}).get('engine')}")
                elif response_data.get('type') == 'error':
                    print(f"   ❌ TTS Error: {response_data.get('data', {}).get('error')}")
                else:
                    print(f"   📦 Unexpected TTS response: {response_data.get('type')}")
                    
            except asyncio.TimeoutError:
                print("   ⏰ TTS test timeout")
        else:
            print("\n3️⃣ Skipping TTS test - service not available")
        
        await websocket.close()
        
    except Exception as e:
        print(f"❌ System check error: {e}")
    
    print("\n🔍 System status check complete!")

if __name__ == "__main__":
    asyncio.run(check_system_status())
