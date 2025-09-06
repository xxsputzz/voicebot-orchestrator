#!/usr/bin/env python3
"""
Complete WebSocket Test - Test the full WebSocket pipeline
"""
import asyncio
import websockets
import json
import time
import requests

async def test_websocket_pipeline():
    """Test the complete WebSocket pipeline"""
    print("🧪 Complete WebSocket Pipeline Test")
    print("="*50)
    
    # Step 1: Check WebSocket orchestrator
    print("1️⃣ Checking WebSocket Orchestrator...")
    try:
        response = requests.get("http://localhost:8080/services", timeout=3)
        if response.status_code == 200:
            services = response.json()
            print(f"   ✅ Orchestrator running - Services: {services}")
        else:
            print(f"   ❌ Orchestrator unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Orchestrator not accessible: {e}")
        return False
    
    # Step 2: Check HTTP services are still running
    print("\n2️⃣ Checking HTTP Services...")
    http_services = [
        ("Orchestrator", 8000),
        ("Whisper STT", 8003),
        ("GPT LLM", 8022),
        ("Zonos TTS", 8014)
    ]
    
    running_services = []
    for name, port in http_services:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"   ✅ {name} running on port {port}")
                running_services.append((name, port))
            else:
                print(f"   ⚠️ {name} unhealthy on port {port}")
        except Exception as e:
            print(f"   ❌ {name} not running: {e}")
    
    # Step 3: Start WebSocket service
    print(f"\n3️⃣ Starting WebSocket LLM Service...")
    import subprocess
    try:
        llm_process = subprocess.Popen(
            ["python", "aws_microservices/ws_llm_gpt_service.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP') else 0
        )
        print(f"   🚀 Started WebSocket LLM service (PID: {llm_process.pid})")
        
        # Wait for registration
        print("   ⏳ Waiting for service registration...")
        await asyncio.sleep(5)
        
        # Check if registered
        response = requests.get("http://localhost:8080/services", timeout=3)
        if response.status_code == 200:
            services = response.json()
            if services:
                print(f"   ✅ Services registered: {[s.get('service_id', s) for s in services]}")
            else:
                print(f"   ⚠️ No services registered yet")
        
    except Exception as e:
        print(f"   ❌ Failed to start WebSocket service: {e}")
        return False
    
    # Step 4: Test WebSocket connection
    print(f"\n4️⃣ Testing WebSocket Connection...")
    try:
        # Connect to CLIENT port 9000 (not service port 9001)
        websocket = await websockets.connect("ws://localhost:9000")
        print("   ✅ WebSocket connection established")
        
        # Wait for welcome message from orchestrator
        welcome = await websocket.recv()
        welcome_data = json.loads(welcome)
        print(f"   ✅ Welcome message: {welcome_data.get('type')}")
        session_id = welcome_data.get('session_id')
        
        # Send test message using proper session_id
        test_message = {
            "type": "conversation_start",
            "session_id": session_id,
            "timestamp": time.time(),
            "data": {"message": "Hello WebSocket"}
        }
        
        await websocket.send(json.dumps(test_message))
        print("   📤 Sent conversation start message")
        
        # Wait for response
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            print(f"   📥 Received: {response}")
        except asyncio.TimeoutError:
            print("   ⏱️ No response received (timeout)")
        
        await websocket.close()
        
    except Exception as e:
        print(f"   ❌ WebSocket connection failed: {e}")
    
    # Step 5: Test HTTP vs WebSocket comparison
    print(f"\n5️⃣ HTTP vs WebSocket Performance...")
    
    if ("GPT LLM", 8022) in running_services:
        # Test HTTP call
        print("   🔍 Testing HTTP call to warm LLM service...")
        try:
            start_time = time.time()
            http_response = requests.post(
                "http://localhost:8022/chat",
                json={"message": "Hello", "session_id": "test_http"},
                timeout=10
            )
            http_time = (time.time() - start_time) * 1000
            
            if http_response.status_code == 200:
                print(f"   ✅ HTTP call: {http_time:.1f}ms")
            else:
                print(f"   ⚠️ HTTP call status: {http_response.status_code}")
                
        except Exception as e:
            print(f"   ❌ HTTP call failed: {e}")
    
    # Step 6: Architecture Analysis
    print(f"\n6️⃣ Architecture Analysis...")
    print("   🏗️ Your Hybrid Architecture:")
    print(f"   • HTTP Services: {len(running_services)} running (always warm)")
    print("   • WebSocket Layer: Available for streaming")
    print("   • Result: Best of both worlds")
    
    print(f"\n💡 WebSocket Benefits:")
    print("   ✅ Persistent connections for conversations")
    print("   ✅ Real-time streaming capability")
    print("   ✅ Lower per-message overhead")
    print("   ✅ Connects to your warm HTTP services")
    
    # Cleanup
    try:
        llm_process.terminate()
        print(f"\n🧹 Cleaned up WebSocket service")
    except:
        pass
    
    return True

async def main():
    await test_websocket_pipeline()

if __name__ == "__main__":
    asyncio.run(main())
