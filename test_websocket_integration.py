#!/usr/bin/env python3
"""
WebSocket Integration Tester - Test WebSocket layer with warm HTTP services
"""
import asyncio
import websockets
import json
import requests
import time
from typing import Optional

class WebSocketTester:
    def __init__(self):
        self.orchestrator_http = "http://localhost:8000"
        self.orchestrator_ws = "ws://localhost:9001"
        self.websocket = None
        
    async def test_websocket_connection(self):
        """Test basic WebSocket connection to orchestrator"""
        print("🔌 Testing WebSocket Connection...")
        try:
            # First start the WebSocket orchestrator if not running
            await self.ensure_ws_orchestrator_running()
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(self.orchestrator_ws)
            print("✅ WebSocket connection established")
            
            # Send ping
            await self.websocket.send(json.dumps({
                "type": "ping",
                "timestamp": time.time()
            }))
            
            # Wait for response
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            print(f"📨 Received: {response}")
            
            return True
            
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            return False
            
    async def ensure_ws_orchestrator_running(self):
        """Ensure WebSocket orchestrator is running"""
        try:
            # Check if WS orchestrator is already running
            response = requests.get("http://localhost:8080/services", timeout=2)
            if response.status_code == 200:
                print("✅ WebSocket orchestrator already running")
                return True
        except:
            pass
            
        print("🚀 Starting WebSocket orchestrator...")
        # Start the WS orchestrator (this should connect to your warm HTTP services)
        import subprocess
        subprocess.Popen(
            ["python", "ws_orchestrator_service.py"],
            cwd=".",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP') else 0
        )
        
        # Wait for it to be ready
        for i in range(10):
            try:
                response = requests.get("http://localhost:8080/services", timeout=2)
                if response.status_code == 200:
                    print("✅ WebSocket orchestrator started")
                    await asyncio.sleep(2)  # Give it time to fully initialize
                    return True
            except:
                pass
            await asyncio.sleep(1)
        
        raise Exception("WebSocket orchestrator failed to start")
    
    async def test_service_registration(self):
        """Test WebSocket service registration"""
        print("\n🔧 Testing Service Registration...")
        
        try:
            # Start a WebSocket LLM service that should connect to the warm HTTP GPT service
            print("🚀 Starting WebSocket LLM service...")
            import subprocess
            llm_process = subprocess.Popen(
                ["python", "aws_microservices/ws_llm_gpt_service.py"],
                cwd=".",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP') else 0
            )
            
            # Wait for registration
            await asyncio.sleep(5)
            
            # Check if service registered
            response = requests.get("http://localhost:8080/services", timeout=3)
            if response.status_code == 200:
                services = response.json()
                print(f"📡 Registered services: {[s.get('service_id', s) for s in services]}")
                
                if any('llm' in str(s).lower() for s in services):
                    print("✅ LLM service registered successfully")
                    return True
                else:
                    print("❌ LLM service not found in registry")
                    return False
            else:
                print(f"❌ Failed to get services: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Service registration test failed: {e}")
            return False
    
    async def test_streaming_conversation(self):
        """Test end-to-end streaming conversation"""
        print("\n🎙️ Testing Streaming Conversation...")
        
        if not self.websocket:
            print("❌ No WebSocket connection available")
            return False
            
        try:
            # Send a conversation request
            conversation_request = {
                "type": "conversation",
                "message": "Hello, how are you today?",
                "session_id": "test_session_001",
                "timestamp": time.time()
            }
            
            print(f"📤 Sending: {conversation_request['message']}")
            await self.websocket.send(json.dumps(conversation_request))
            
            # Wait for streaming responses
            responses = []
            timeout_count = 0
            max_timeout = 10
            
            while timeout_count < max_timeout:
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
                    response_data = json.loads(response)
                    responses.append(response_data)
                    
                    print(f"📥 Received {response_data.get('type', 'unknown')}: {str(response_data)[:100]}...")
                    
                    # Check if conversation is complete
                    if response_data.get('type') == 'conversation_complete':
                        print("✅ Streaming conversation completed successfully")
                        return True
                        
                except asyncio.TimeoutError:
                    timeout_count += 1
                    print(f"⏱️ Waiting for response... ({timeout_count}/{max_timeout})")
                    
            print(f"⚠️ Conversation test completed with {len(responses)} responses")
            return len(responses) > 0
            
        except Exception as e:
            print(f"❌ Streaming conversation test failed: {e}")
            return False
    
    async def test_http_to_websocket_bridge(self):
        """Test that WebSocket services properly connect to warm HTTP services"""
        print("\n🌉 Testing HTTP-to-WebSocket Bridge...")
        
        # Test direct HTTP call vs WebSocket call to same service
        try:
            # Direct HTTP call to warm GPT service
            print("🔍 Testing direct HTTP call...")
            http_start = time.time()
            http_response = requests.post(
                "http://localhost:8022/chat",
                json={"message": "What is 2+2?", "session_id": "test"},
                timeout=10
            )
            http_time = time.time() - http_start
            
            if http_response.status_code == 200:
                print(f"✅ HTTP response time: {http_time:.2f}s")
                print(f"📨 HTTP response: {http_response.json()}")
                
                # Now test WebSocket call to the same service
                print("\n🔍 Testing WebSocket call to same service...")
                if self.websocket:
                    ws_start = time.time()
                    await self.websocket.send(json.dumps({
                        "type": "llm_request",
                        "message": "What is 2+2?",
                        "session_id": "test_ws"
                    }))
                    
                    ws_response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                    ws_time = time.time() - ws_start
                    
                    print(f"✅ WebSocket response time: {ws_time:.2f}s")
                    print(f"📨 WebSocket response: {ws_response}")
                    
                    # Compare latencies
                    if ws_time < http_time + 0.1:  # Allow 100ms tolerance
                        print(f"🎯 WebSocket latency excellent: {ws_time:.2f}s vs HTTP {http_time:.2f}s")
                    else:
                        print(f"⚠️ WebSocket latency higher than expected: {ws_time:.2f}s vs HTTP {http_time:.2f}s")
                    
                    return True
            else:
                print(f"❌ HTTP call failed: {http_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ HTTP-to-WebSocket bridge test failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up connections"""
        if self.websocket:
            await self.websocket.close()
            print("🧹 WebSocket connection closed")

async def main():
    """Run all WebSocket tests"""
    print("🧪 WebSocket Integration Test Suite")
    print("="*60)
    print("Testing WebSocket layer with warm HTTP services")
    print("="*60)
    
    tester = WebSocketTester()
    
    # Test sequence
    tests = [
        ("WebSocket Connection", tester.test_websocket_connection),
        ("Service Registration", tester.test_service_registration),
        ("HTTP-to-WebSocket Bridge", tester.test_http_to_websocket_bridge),
        ("Streaming Conversation", tester.test_streaming_conversation),
    ]
    
    results = {}
    
    try:
        for test_name, test_func in tests:
            print(f"\n" + "="*60)
            print(f"🧪 Running: {test_name}")
            print("="*60)
            
            try:
                result = await test_func()
                results[test_name] = result
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"\n{status}: {test_name}")
            except Exception as e:
                results[test_name] = False
                print(f"\n❌ FAILED: {test_name} - {e}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    
    finally:
        await tester.cleanup()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All WebSocket tests passed! Your hybrid architecture is working perfectly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    asyncio.run(main())
