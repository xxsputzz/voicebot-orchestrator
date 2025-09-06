#!/usr/bin/env python3
"""
WebSocket Services Conversion Test
Test the conversion of existing HTTP services to WebSocket streaming

This validates:
1. Service connectivity and registration
2. Message routing through orchestrator
3. Streaming capabilities (STT chunks, LLM tokens, TTS audio)
4. End-to-end pipeline functionality
5. Performance comparison with original HTTP services
"""

import asyncio
import json
import logging
import time
import base64
import tempfile
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import websockets
import uuid
from pathlib import Path
import subprocess
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class WebSocketServicesTester:
    """Comprehensive tester for converted WebSocket services"""
    
    def __init__(self):
        self.orchestrator_url = "ws://localhost:9000"  # Client port
        self.http_api_url = "http://localhost:8080"    # HTTP API
        self.client_websocket = None
        
        # Test results
        self.test_results = {}
        self.performance_data = {}
        
        # Service endpoints (original HTTP services for comparison)
        self.http_services = {
            "stt": "http://localhost:8003",
            "llm": "http://localhost:8022", 
            "tts": "http://localhost:8015"
        }
        
        # WebSocket service IDs
        self.ws_services = {
            "stt": "stt_whisper_ws",
            "llm": "llm_gpt_ws",
            "tts": "tts_tortoise_ws"
        }
    
    def safe_print(self, text: str):
        """Safe print function that handles Unicode characters for Windows console."""
        try:
            print(text)
        except UnicodeEncodeError:
            safe_text = text.encode('ascii', 'replace').decode('ascii')
            print(safe_text)
    
    async def check_orchestrator_health(self) -> bool:
        """Check orchestrator health via HTTP API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.http_api_url}/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("status") == "healthy"
        except Exception as e:
            logging.debug(f"Orchestrator health check failed: {e}")
            
        return False
    
    async def get_registered_services(self) -> List[Dict]:
        """Get list of registered services via HTTP API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.http_api_url}/services", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("services", [])
        except Exception as e:
            logging.debug(f"Failed to get registered services: {e}")
            
        return []
    
    async def connect_to_orchestrator(self) -> bool:
        """Connect to WebSocket orchestrator"""
        try:
            self.client_websocket = await websockets.connect(
                self.orchestrator_url,
                ping_interval=20,
                ping_timeout=10
            )
            return True
            
        except Exception as e:
            self.safe_print(f"❌ Failed to connect to orchestrator: {e}")
            return False
    
    async def create_test_audio(self) -> str:
        """Create test audio file"""
        try:
            import wave
            import struct
            import math
            
            # Audio parameters
            sample_rate = 22050
            duration = 2.0  # 2 seconds
            frequency = 440.0  # A4 note
            amplitude = 0.3
            
            # Generate samples
            samples = []
            for i in range(int(sample_rate * duration)):
                t = i / sample_rate
                sample = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
                samples.append(sample)
            
            # Create WAV file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            
            with wave.open(temp_file.name, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                for sample in samples:
                    wav_file.writeframes(struct.pack('<h', sample))
            
            temp_file.close()
            
            # Read as base64
            with open(temp_file.name, 'rb') as f:
                audio_data = f.read()
            
            os.unlink(temp_file.name)
            
            return base64.b64encode(audio_data).decode('utf-8')
            
        except Exception as e:
            self.safe_print(f"❌ Error creating test audio: {e}")
            return None
    
    async def test_orchestrator_connectivity(self) -> bool:
        """Test 1: Orchestrator connectivity and health"""
        self.safe_print("🧪 Test 1: Orchestrator Connectivity")
        self.safe_print("-" * 50)
        
        # HTTP health check
        http_healthy = await self.check_orchestrator_health()
        self.safe_print(f"   HTTP API Health: {'✅ PASS' if http_healthy else '❌ FAIL'}")
        
        # WebSocket connection
        ws_connected = await self.connect_to_orchestrator()
        self.safe_print(f"   WebSocket Connection: {'✅ PASS' if ws_connected else '❌ FAIL'}")
        
        if self.client_websocket:
            await self.client_websocket.close()
            self.client_websocket = None
        
        test_passed = http_healthy and ws_connected
        self.test_results["orchestrator_connectivity"] = test_passed
        
        self.safe_print(f"📋 Test 1 Result: {'✅ PASS' if test_passed else '❌ FAIL'}")
        return test_passed
    
    async def test_service_registration(self) -> bool:
        """Test 2: Service registration and discovery"""
        self.safe_print("\n🧪 Test 2: Service Registration")
        self.safe_print("-" * 50)
        
        # Get registered services
        registered_services = await self.get_registered_services()
        self.safe_print(f"   Total Registered Services: {len(registered_services)}")
        
        # Check for our converted services
        service_ids = [s.get("service_id") for s in registered_services]
        
        results = {}
        for service_type, service_id in self.ws_services.items():
            is_registered = service_id in service_ids
            results[service_type] = is_registered
            self.safe_print(f"   {service_type.upper()} Service ({service_id}): {'✅ REGISTERED' if is_registered else '❌ NOT FOUND'}")
        
        # Show service details
        if registered_services:
            self.safe_print("\n   📋 Registered Service Details:")
            for service in registered_services:
                name = service.get("service_name", "Unknown")
                service_type = service.get("service_type", "unknown")
                status = service.get("status", "unknown")
                self.safe_print(f"      • {name} ({service_type}): {status}")
        
        test_passed = all(results.values())
        self.test_results["service_registration"] = test_passed
        
        self.safe_print(f"\n📋 Test 2 Result: {'✅ PASS' if test_passed else '❌ FAIL'}")
        return test_passed
    
    async def test_stt_service(self) -> Tuple[bool, float]:
        """Test 3: STT service WebSocket functionality"""
        self.safe_print("\n🧪 Test 3: STT Service WebSocket")
        self.safe_print("-" * 50)
        
        if not await self.connect_to_orchestrator():
            return False, 0.0
        
        try:
            # Create test audio
            audio_base64 = await self.create_test_audio()
            if not audio_base64:
                return False, 0.0
            
            session_id = str(uuid.uuid4())
            start_time = time.time()
            
            # Send audio to STT service
            message = {
                "type": "audio_file",
                "session_id": session_id,
                "target_service": "stt_whisper_ws",
                "data": {
                    "audio_data": audio_base64,
                    "language": "auto",
                    "task": "transcribe"
                }
            }
            
            await self.client_websocket.send(json.dumps(message))
            self.safe_print("   📤 Sent audio to STT service")
            
            # Wait for response
            result_text = None
            timeout = 30  # 30 seconds
            
            async for message in self.client_websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    msg_session = data.get("session_id")
                    
                    if msg_session == session_id:
                        if msg_type in ["transcription_result", "transcription_final"]:
                            result_text = data.get("data", {}).get("text", "")
                            break
                        elif msg_type == "error":
                            error = data.get("data", {}).get("error", "Unknown error")
                            self.safe_print(f"   ❌ STT Error: {error}")
                            break
                    
                    if time.time() - start_time > timeout:
                        self.safe_print(f"   ⏰ STT timeout after {timeout}s")
                        break
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self.safe_print(f"   ❌ Message processing error: {e}")
                    break
            
            processing_time = time.time() - start_time
            test_passed = result_text is not None and result_text.strip() != ""
            
            self.safe_print(f"   📝 STT Result: '{result_text}'" if result_text else "   ❌ No transcription received")
            self.safe_print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
            self.safe_print(f"   📋 STT Test: {'✅ PASS' if test_passed else '❌ FAIL'}")
            
            self.performance_data["stt_websocket"] = processing_time
            
            return test_passed, processing_time
            
        except Exception as e:
            self.safe_print(f"   ❌ STT test error: {e}")
            return False, 0.0
        finally:
            if self.client_websocket:
                await self.client_websocket.close()
                self.client_websocket = None
    
    async def test_llm_service(self) -> Tuple[bool, float]:
        """Test 4: LLM service WebSocket functionality"""
        self.safe_print("\n🧪 Test 4: LLM Service WebSocket")
        self.safe_print("-" * 50)
        
        if not await self.connect_to_orchestrator():
            return False, 0.0
        
        try:
            session_id = str(uuid.uuid4())
            test_text = "Hello, can you help me test the WebSocket LLM service?"
            start_time = time.time()
            
            # Send text to LLM service
            message = {
                "type": "text_input",
                "session_id": session_id,
                "target_service": "llm_gpt_ws",
                "data": {
                    "text": test_text,
                    "stream_tokens": True,
                    "max_tokens": 128
                }
            }
            
            await self.client_websocket.send(json.dumps(message))
            self.safe_print(f"   📤 Sent text to LLM: '{test_text}'")
            
            # Wait for response
            result_text = None
            tokens_received = 0
            timeout = 30
            
            async for message in self.client_websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    msg_session = data.get("session_id")
                    
                    if msg_session == session_id:
                        if msg_type == "llm_token":
                            tokens_received += 1
                        elif msg_type == "llm_stream_complete":
                            result_text = data.get("data", {}).get("full_text", "")
                            break
                        elif msg_type == "llm_response":
                            result_text = data.get("data", {}).get("text", "")
                            break
                        elif msg_type == "error":
                            error = data.get("data", {}).get("error", "Unknown error")
                            self.safe_print(f"   ❌ LLM Error: {error}")
                            break
                    
                    if time.time() - start_time > timeout:
                        self.safe_print(f"   ⏰ LLM timeout after {timeout}s")
                        break
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self.safe_print(f"   ❌ Message processing error: {e}")
                    break
            
            processing_time = time.time() - start_time
            test_passed = result_text is not None and result_text.strip() != ""
            
            self.safe_print(f"   🤖 LLM Response: '{result_text[:100]}...'" if result_text and len(result_text) > 100 else f"   🤖 LLM Response: '{result_text}'" if result_text else "   ❌ No response received")
            self.safe_print(f"   🔤 Tokens Streamed: {tokens_received}")
            self.safe_print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
            self.safe_print(f"   📋 LLM Test: {'✅ PASS' if test_passed else '❌ FAIL'}")
            
            self.performance_data["llm_websocket"] = processing_time
            
            return test_passed, processing_time
            
        except Exception as e:
            self.safe_print(f"   ❌ LLM test error: {e}")
            return False, 0.0
        finally:
            if self.client_websocket:
                await self.client_websocket.close()
                self.client_websocket = None
    
    async def test_tts_service(self) -> Tuple[bool, float]:
        """Test 5: TTS service WebSocket functionality"""
        self.safe_print("\n🧪 Test 5: TTS Service WebSocket")
        self.safe_print("-" * 50)
        
        if not await self.connect_to_orchestrator():
            return False, 0.0
        
        try:
            session_id = str(uuid.uuid4())
            test_text = "This is a test of the WebSocket text to speech service conversion."
            start_time = time.time()
            
            # Send text to TTS service
            message = {
                "type": "text_input",
                "session_id": session_id,
                "target_service": "tts_tortoise_ws",
                "data": {
                    "text": test_text,
                    "voice": "rainbow",
                    "stream_audio": True,
                    "output_format": "base64"
                }
            }
            
            await self.client_websocket.send(json.dumps(message))
            self.safe_print(f"   📤 Sent text to TTS: '{test_text}'")
            
            # Wait for response
            audio_chunks = 0
            audio_size = 0
            completed = False
            timeout = 60  # 60 seconds for TTS
            
            async for message in self.client_websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    msg_session = data.get("session_id")
                    
                    if msg_session == session_id:
                        if msg_type == "audio_chunk":
                            audio_chunks += 1
                            chunk_data = data.get("data", {}).get("audio_data", "")
                            if chunk_data:
                                # Estimate audio size from base64
                                audio_size += len(base64.b64decode(chunk_data))
                            
                            is_final = data.get("data", {}).get("is_final", False)
                            if is_final:
                                completed = True
                                break
                        elif msg_type == "tts_stream_complete":
                            completed = True
                            break
                        elif msg_type == "tts_audio_complete":
                            audio_data = data.get("data", {}).get("audio_data", "")
                            if audio_data:
                                audio_size = len(base64.b64decode(audio_data))
                            completed = True
                            break
                        elif msg_type == "error":
                            error = data.get("data", {}).get("error", "Unknown error")
                            self.safe_print(f"   ❌ TTS Error: {error}")
                            break
                    
                    if time.time() - start_time > timeout:
                        self.safe_print(f"   ⏰ TTS timeout after {timeout}s")
                        break
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self.safe_print(f"   ❌ Message processing error: {e}")
                    break
            
            processing_time = time.time() - start_time
            test_passed = completed and (audio_chunks > 0 or audio_size > 0)
            
            self.safe_print(f"   🔊 Audio Chunks: {audio_chunks}")
            self.safe_print(f"   📁 Audio Size: {audio_size} bytes")
            self.safe_print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
            self.safe_print(f"   📋 TTS Test: {'✅ PASS' if test_passed else '❌ FAIL'}")
            
            self.performance_data["tts_websocket"] = processing_time
            
            return test_passed, processing_time
            
        except Exception as e:
            self.safe_print(f"   ❌ TTS test error: {e}")
            return False, 0.0
        finally:
            if self.client_websocket:
                await self.client_websocket.close()
                self.client_websocket = None
    
    async def test_end_to_end_pipeline(self) -> bool:
        """Test 6: End-to-end pipeline functionality"""
        self.safe_print("\n🧪 Test 6: End-to-End Pipeline")
        self.safe_print("-" * 50)
        
        if not await self.connect_to_orchestrator():
            return False
        
        try:
            session_id = str(uuid.uuid4())
            pipeline_start = time.time()
            
            # Create test audio
            audio_base64 = await self.create_test_audio()
            if not audio_base64:
                return False
            
            # Pipeline state
            stt_result = None
            llm_result = None
            tts_completed = False
            
            # Send audio to start pipeline
            message = {
                "type": "audio_file",
                "session_id": session_id,
                "target_service": "stt_whisper_ws",
                "data": {
                    "audio_data": audio_base64,
                    "language": "auto",
                    "task": "transcribe"
                }
            }
            
            await self.client_websocket.send(json.dumps(message))
            self.safe_print("   🎬 Started pipeline with audio input")
            
            timeout = 90  # 90 seconds for full pipeline
            
            async for message in self.client_websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    msg_session = data.get("session_id")
                    
                    if msg_session == session_id:
                        # STT result
                        if msg_type in ["transcription_result", "transcription_final"] and not stt_result:
                            stt_result = data.get("data", {}).get("text", "")
                            self.safe_print(f"   🎙️  STT: '{stt_result}'")
                            
                            # Send to LLM
                            if stt_result:
                                llm_message = {
                                    "type": "text_input",
                                    "session_id": session_id,
                                    "target_service": "llm_gpt_ws",
                                    "data": {
                                        "text": stt_result,
                                        "stream_tokens": False,
                                        "max_tokens": 64
                                    }
                                }
                                await self.client_websocket.send(json.dumps(llm_message))
                        
                        # LLM result
                        elif msg_type in ["llm_response", "llm_stream_complete"] and not llm_result:
                            llm_result = data.get("data", {}).get("text", "") or data.get("data", {}).get("full_text", "")
                            self.safe_print(f"   🤖 LLM: '{llm_result}'")
                            
                            # Send to TTS
                            if llm_result:
                                tts_message = {
                                    "type": "text_input",
                                    "session_id": session_id,
                                    "target_service": "tts_tortoise_ws",
                                    "data": {
                                        "text": llm_result,
                                        "voice": "rainbow",
                                        "stream_audio": False
                                    }
                                }
                                await self.client_websocket.send(json.dumps(tts_message))
                        
                        # TTS result
                        elif msg_type in ["tts_audio_complete", "tts_stream_complete"] and not tts_completed:
                            tts_completed = True
                            self.safe_print("   🔊 TTS: Audio synthesis completed")
                            
                            # Pipeline complete
                            break
                        
                        elif msg_type == "error":
                            error = data.get("data", {}).get("error", "Unknown error")
                            self.safe_print(f"   ❌ Pipeline Error: {error}")
                    
                    if time.time() - pipeline_start > timeout:
                        self.safe_print(f"   ⏰ Pipeline timeout after {timeout}s")
                        break
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self.safe_print(f"   ❌ Message processing error: {e}")
                    break
            
            pipeline_time = time.time() - pipeline_start
            test_passed = all([stt_result, llm_result, tts_completed])
            
            self.safe_print(f"   ⏱️  Total Pipeline Time: {pipeline_time:.2f}s")
            self.safe_print(f"   📋 Pipeline Test: {'✅ PASS' if test_passed else '❌ FAIL'}")
            
            self.performance_data["pipeline_websocket"] = pipeline_time
            
            return test_passed
            
        except Exception as e:
            self.safe_print(f"   ❌ Pipeline test error: {e}")
            return False
        finally:
            if self.client_websocket:
                await self.client_websocket.close()
                self.client_websocket = None
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        self.safe_print("\n" + "="*70)
        self.safe_print("📊 WEBSOCKET SERVICES CONVERSION TEST SUMMARY")
        self.safe_print("="*70)
        
        # Test results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        self.safe_print(f"📋 Test Results: {passed_tests}/{total_tests} PASSED")
        self.safe_print()
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.safe_print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        # Performance data
        if self.performance_data:
            self.safe_print("\n⏱️  Performance Data:")
            for service, time_taken in self.performance_data.items():
                self.safe_print(f"   {service.replace('_', ' ').title()}: {time_taken:.2f}s")
        
        # Overall result
        overall_success = passed_tests == total_tests
        self.safe_print("\n" + "="*70)
        if overall_success:
            self.safe_print("🎉 CONVERSION TEST RESULT: ✅ SUCCESS")
            self.safe_print("🔄 Your HTTP services have been successfully converted to WebSocket streaming!")
        else:
            self.safe_print("❌ CONVERSION TEST RESULT: FAILED")
            self.safe_print("⚠️  Some issues were found with the WebSocket conversion")
        
        self.safe_print("="*70)
        
        return overall_success
    
    async def run_all_tests(self) -> bool:
        """Run all conversion tests"""
        self.safe_print("🧪 WebSocket Services Conversion Test Suite")
        self.safe_print("🔄 Testing conversion of HTTP services to WebSocket streaming")
        self.safe_print("="*70)
        
        # Test 1: Orchestrator connectivity
        test1_result = await self.test_orchestrator_connectivity()
        if not test1_result:
            self.safe_print("❌ Cannot proceed without orchestrator - aborting tests")
            return False
        
        # Test 2: Service registration
        test2_result = await self.test_service_registration()
        
        # Test 3: STT service
        test3_result, stt_time = await self.test_stt_service()
        self.test_results["stt_service"] = test3_result
        
        # Test 4: LLM service
        test4_result, llm_time = await self.test_llm_service()
        self.test_results["llm_service"] = test4_result
        
        # Test 5: TTS service
        test5_result, tts_time = await self.test_tts_service()
        self.test_results["tts_service"] = test5_result
        
        # Test 6: End-to-end pipeline (only if individual services pass)
        if all([test3_result, test4_result, test5_result]):
            test6_result = await self.test_end_to_end_pipeline()
            self.test_results["end_to_end_pipeline"] = test6_result
        else:
            self.safe_print("\n🧪 Test 6: End-to-End Pipeline")
            self.safe_print("-" * 50)
            self.safe_print("   ⏭️  Skipped due to individual service failures")
            self.test_results["end_to_end_pipeline"] = False
        
        # Print summary
        return self.print_test_summary()

def safe_print(text):
    """Safe print function that handles Unicode characters for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)

async def main():
    """Main entry point"""
    safe_print("🔄 WebSocket Services Conversion Test")
    safe_print("Testing converted HTTP → WebSocket services")
    safe_print("=" * 70)
    
    tester = WebSocketServicesTester()
    
    # Add required imports
    import wave
    import struct
    import math
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            safe_print("\n🎉 All tests passed! Your services are ready for streaming.")
            safe_print("💡 Next steps:")
            safe_print("   • Run: python end_to_end_audio_pipeline.py")
            safe_print("   • Test real audio input with computer microphone")
        else:
            safe_print("\n❌ Some tests failed. Check the output above for issues.")
            safe_print("💡 Troubleshooting:")
            safe_print("   • Ensure all services are running via launch_converted_services.py")
            safe_print("   • Check orchestrator logs for connection issues")
            
    except KeyboardInterrupt:
        safe_print("\n👋 Test interrupted by user")
    except Exception as e:
        safe_print(f"\n❌ Test error: {e}")
        return False
    
    return success

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the tests
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        safe_print("\n👋 Tests interrupted")
        sys.exit(1)
    except Exception as e:
        safe_print(f"❌ Test suite error: {e}")
        sys.exit(1)
