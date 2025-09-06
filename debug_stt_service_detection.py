#!/usr/bin/env python3
"""
Debug STT Service Detection
"""

import asyncio
import json
import websockets
import time
import requests

async def debug_stt_service():
    """Debug STT service detection"""
    print("🔍 Debugging STT Service Detection")
    print("=" * 50)
    
    # Check HTTP services first
    print("1️⃣ Checking HTTP Service Registry:")
    try:
        response = requests.get("http://localhost:8080/services", timeout=5)
        if response.status_code == 200:
            services = response.json()
            print(f"   📋 Total services: {len(services)}")
            for service in services:
                service_type = service.get('service_type', 'unknown')
                service_id = service.get('service_id', 'unknown')
                print(f"   - {service_type.upper()}: {service_id}")
                if service_type == 'stt':
                    print(f"     🎯 STT service found: {service_id}")
        else:
            print(f"   ❌ HTTP error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ HTTP request failed: {e}")
    
    # Check WebSocket services
    print("\n2️⃣ Checking WebSocket Service Detection:")
    try:
        websocket = await websockets.connect("ws://localhost:9000")
        session_id = f"debug_{int(time.time())}"
        
        # Wait for session start
        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        print(f"   📡 Connected: {json.loads(response).get('type')}")
        
        # Send health check
        health_check = {
            "type": "health_check",
            "session_id": session_id,
            "timestamp": time.time(),
            "data": {}
        }
        await websocket.send(json.dumps(health_check))
        
        # Get health response
        response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
        health_data = json.loads(response)
        
        print(f"   📊 Health Response Type: {health_data.get('type')}")
        
        if health_data.get('type') == 'health_status':
            data = health_data.get('data', {})
            available_services = data.get('services', [])
            registered_services = data.get('registered_services', [])
            service_details = data.get('service_details', {})
            
            print(f"   🎯 Available Services: {available_services}")
            print(f"   📋 Registered Services: {registered_services}")
            print(f"   🔧 Service Details: {service_details}")
            
            if 'stt' in available_services:
                print("   ✅ STT service is available via WebSocket")
            else:
                print("   ❌ STT service NOT available via WebSocket")
                if 'stt' in service_details:
                    print(f"      Status: {service_details['stt']}")
        
        await websocket.close()
        
    except Exception as e:
        print(f"   ❌ WebSocket test failed: {e}")
    
    # Test direct STT message
    print("\n3️⃣ Testing Direct STT Message:")
    try:
        websocket = await websockets.connect("ws://localhost:9000")
        session_id = f"stt_test_{int(time.time())}"
        
        # Wait for session start
        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        
        # Send audio chunk (dummy data)
        import base64
        dummy_audio = b'\x00' * 1000  # 1000 bytes of silence
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
        
        print("   📤 Sending dummy audio chunk...")
        await websocket.send(json.dumps(audio_message))
        
        # Wait for any response
        print("   ⏳ Waiting for STT response...")
        timeout_counter = 0
        response_received = False
        
        while timeout_counter < 10:  # 10 second timeout
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                response_data = json.loads(response)
                response_type = response_data.get('type')
                
                print(f"   📨 Response: {response_type}")
                
                if response_type in ['transcript_partial', 'transcript_final', 'error']:
                    print(f"      Data: {response_data.get('data', {})}")
                    response_received = True
                    break
                    
            except asyncio.TimeoutError:
                timeout_counter += 1
                continue
        
        if not response_received:
            print("   ❌ No STT response received")
        else:
            print("   ✅ STT service responded")
        
        await websocket.close()
        
    except Exception as e:
        print(f"   ❌ Direct STT test failed: {e}")

if __name__ == "__main__":
    asyncio.run(debug_stt_service())
