#!/usr/bin/env python3

"""
Comprehensive Test Suite for Independent Microservices
Following patterns from existing tests/ folder
"""
import os
import sys
import time
import requests
import asyncio
import subprocess
from pathlib import Path
import json
import wave
import tempfile
import base64

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class IndependentServiceTester:
    """Comprehensive testing for independent microservices following existing patterns"""
    
    def __init__(self):
        self.services = {
            "stt": {"port": 8002, "name": "STT Service"},
            "kokoro_tts": {"port": 8011, "name": "Kokoro TTS"},
            "hira_dia_tts": {"port": 8012, "name": "Hira Dia TTS"},
            "mistral_llm": {"port": 8021, "name": "Mistral LLM"},
            "gpt_llm": {"port": 8022, "name": "GPT LLM"}
        }
        
        self.audio_output_dir = project_root / "tests" / "audio_samples"
        self.audio_output_dir.mkdir(exist_ok=True)
        
        self.test_results = {}
    
    def check_service_availability(self, service_key: str) -> bool:
        """Check if a service is available and healthy"""
        service = self.services[service_key]
        try:
            response = requests.get(f"http://localhost:{service['port']}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_services(self) -> dict:
        """Get list of available services"""
        available = {}
        print("🔍 Checking service availability...")
        
        for service_key, service_info in self.services.items():
            is_available = self.check_service_availability(service_key)
            available[service_key] = is_available
            status_icon = "✅" if is_available else "❌"
            print(f"  {status_icon} {service_info['name']} (Port {service_info['port']})")
        
        return available
    
    def test_stt_service(self) -> dict:
        """Test STT service following existing patterns"""
        print("\n🎤 Testing STT Service...")
        result = {"name": "STT Service", "passed": False, "details": []}
        
        if not self.check_service_availability("stt"):
            result["details"].append("❌ Service not available")
            return result
        
        try:
            # Test 1: Health check
            stt_port = self.services["stt"]["port"]
            response = requests.get(f"http://localhost:{stt_port}/health", timeout=5)
            if response.status_code == 200:
                result["details"].append("✅ Health check passed")
            else:
                result["details"].append("❌ Health check failed")
            
            # Test 2: Create fake audio for transcription test
            fake_audio = self.create_test_audio_data()
            files = {"audio": ("test.wav", fake_audio, "audio/wav")}
            
            response = requests.post(f"http://localhost:{stt_port}/transcribe", files=files, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if "text" in data:
                    result["details"].append(f"✅ Transcription test passed: '{data['text'][:50]}...'")
                    result["passed"] = True
                else:
                    result["details"].append("❌ No transcription text returned")
            else:
                result["details"].append(f"❌ Transcription failed (Status: {response.status_code})")
            
        except Exception as e:
            result["details"].append(f"❌ STT test error: {e}")
        
        return result
    
    def test_tts_service(self, service_key: str) -> dict:
        """Test TTS service following existing patterns"""
        service_info = self.services[service_key]
        print(f"\n🔊 Testing {service_info['name']}...")
        
        result = {"name": service_info['name'], "passed": False, "details": []}
        
        if not self.check_service_availability(service_key):
            result["details"].append("❌ Service not available")
            return result
        
        try:
            port = service_info["port"]
            
            # Test 1: Health check
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                result["details"].append("✅ Health check passed")
            else:
                result["details"].append("❌ Health check failed")
            
            # Test 2: Text synthesis (following test_real_kokoro.py patterns)
            test_texts = [
                "Hello, this is a test of the text to speech system.",
                "The quick brown fox jumps over the lazy dog.",
                "I am testing the voice synthesis capabilities."
            ]
            
            success_count = 0
            for i, text in enumerate(test_texts):
                payload = {
                    "text": text,
                    "return_audio": True
                }
                
                # Adjust timeout based on TTS service type
                if service_key == "hira_dia_tts":
                    # Hira Dia takes ~8+ minutes for high quality generation
                    timeout = 600  # 10 minutes
                else:
                    # Kokoro and other TTS services are much faster
                    timeout = 30  # 30 seconds
                
                response = requests.post(f"http://localhost:{port}/synthesize", json=payload, timeout=timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("audio_base64"):
                        # Save audio to file (following existing patterns)
                        audio_data = base64.b64decode(data["audio_base64"])
                        output_file = self.audio_output_dir / f"{service_key}_test_{i+1}.wav"
                        
                        with open(output_file, "wb") as f:
                            f.write(audio_data)
                        
                        result["details"].append(f"✅ Audio {i+1} generated: {output_file.name}")
                        success_count += 1
                    else:
                        result["details"].append(f"❌ Audio {i+1}: No audio data returned")
                else:
                    result["details"].append(f"❌ Audio {i+1} failed (Status: {response.status_code})")
            
            if success_count == len(test_texts):
                result["passed"] = True
                result["details"].append(f"🎉 All {success_count} audio tests passed")
            else:
                result["details"].append(f"⚠️ {success_count}/{len(test_texts)} audio tests passed")
            
        except Exception as e:
            result["details"].append(f"❌ TTS test error: {e}")
        
        return result
    
    def test_llm_service(self, service_key: str) -> dict:
        """Test LLM service following existing patterns"""
        service_info = self.services[service_key]
        print(f"\n🧠 Testing {service_info['name']}...")
        
        result = {"name": service_info['name'], "passed": False, "details": []}
        
        if not self.check_service_availability(service_key):
            result["details"].append("❌ Service not available")
            return result
        
        try:
            port = service_info["port"]
            
            # Test 1: Health check
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                result["details"].append("✅ Health check passed")
            else:
                result["details"].append("❌ Health check failed")
            
            # Test 2: Text generation (following existing LLM test patterns)
            test_prompts = [
                "What is the capital of France?",
                "Explain artificial intelligence in one sentence.",
                "How can I help you today?"
            ]
            
            success_count = 0
            for i, prompt in enumerate(test_prompts):
                payload = {
                    "text": prompt,
                    "use_cache": True
                }
                
                response = requests.post(f"http://localhost:{port}/generate", json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("response") and len(data["response"].strip()) > 0:
                        response_preview = data["response"][:100] + "..." if len(data["response"]) > 100 else data["response"]
                        result["details"].append(f"✅ Response {i+1}: '{response_preview}'")
                        success_count += 1
                    else:
                        result["details"].append(f"❌ Response {i+1}: Empty response")
                else:
                    result["details"].append(f"❌ Response {i+1} failed (Status: {response.status_code})")
            
            if success_count == len(test_prompts):
                result["passed"] = True
                result["details"].append(f"🎉 All {success_count} generation tests passed")
            else:
                result["details"].append(f"⚠️ {success_count}/{len(test_prompts)} generation tests passed")
            
        except Exception as e:
            result["details"].append(f"❌ LLM test error: {e}")
        
        return result
    
    def test_service_combinations(self) -> dict:
        """Test service combinations following existing patterns"""
        print("\n🔄 Testing Service Combinations...")
        
        result = {"name": "Service Combinations", "passed": False, "details": []}
        available = self.get_available_services()
        
        # Test combinations based on available services
        combinations = [
            {
                "name": "Fast Combo", 
                "services": ["stt", "kokoro_tts", "mistral_llm"],
                "description": "Real-time processing"
            },
            {
                "name": "Quality Combo",
                "services": ["stt", "hira_dia_tts", "gpt_llm"],
                "description": "Maximum quality"
            }
        ]
        
        success_count = 0
        for combo in combinations:
            combo_available = all(available.get(service, False) for service in combo["services"])
            
            if combo_available:
                result["details"].append(f"✅ {combo['name']}: All services available")
                
                # Test end-to-end flow if possible
                if self.test_end_to_end_flow(combo["services"], combo["name"]):
                    result["details"].append(f"🎉 {combo['name']}: End-to-end test passed")
                    success_count += 1
                else:
                    result["details"].append(f"⚠️ {combo['name']}: End-to-end test issues")
            else:
                missing = [s for s in combo["services"] if not available.get(s, False)]
                result["details"].append(f"❌ {combo['name']}: Missing services: {missing}")
        
        # Add GPT compatibility test if available
        if available.get("gpt_llm"):
            gpt_test_passed = self.test_gpt_compatibility()
            if gpt_test_passed:
                result["details"].append("✅ GPT LLM: Advanced compatibility test passed")
                success_count += 1
            else:
                result["details"].append("❌ GPT LLM: Advanced compatibility test failed")
        
        result["passed"] = success_count > 0
        return result
    
    def test_end_to_end_flow(self, services: list, combo_name: str) -> bool:
        """Test end-to-end flow with available services"""
        try:
            print(f"    🔄 Testing {combo_name} end-to-end flow...")
            
            # Step 1: Check that all services respond to health checks
            for service_key in services:
                if service_key in self.services:
                    port = self.services[service_key]["port"]
                    response = requests.get(f"http://localhost:{port}/health", timeout=5)
                    if response.status_code != 200:
                        print(f"    ❌ {service_key} health check failed")
                        return False
            
            # Step 2: Test actual service interaction (simplified workflow)
            test_text = "What is my account balance?"
            
            # If we have LLM service, test text generation
            llm_response = None
            if "mistral_llm" in services:
                llm_port = self.services["mistral_llm"]["port"]
                llm_payload = {"text": test_text, "use_cache": True}
                llm_resp = requests.post(f"http://localhost:{llm_port}/generate", 
                                       json=llm_payload, timeout=30)
                if llm_resp.status_code == 200:
                    llm_response = llm_resp.json().get("response", "")
                    print(f"    ✅ Mistral LLM response: {llm_response[:50]}...")
                else:
                    print(f"    ❌ Mistral LLM failed: {llm_resp.status_code}")
                    return False
            
            elif "gpt_llm" in services:
                gpt_port = self.services["gpt_llm"]["port"]
                gpt_payload = {"text": test_text, "use_cache": True}
                gpt_resp = requests.post(f"http://localhost:{gpt_port}/generate", 
                                       json=gpt_payload, timeout=30)
                if gpt_resp.status_code == 200:
                    llm_response = gpt_resp.json().get("response", "")
                    print(f"    ✅ GPT LLM response: {llm_response[:50]}...")
                else:
                    print(f"    ❌ GPT LLM failed: {gpt_resp.status_code}")
                    return False
            
            # If we have TTS service, test audio generation
            if llm_response and ("kokoro_tts" in services or "hira_dia_tts" in services):
                tts_service = "kokoro_tts" if "kokoro_tts" in services else "hira_dia_tts"
                tts_port = self.services[tts_service]["port"]
                tts_payload = {"text": llm_response[:100]}  # Limit length for testing
                
                # Use appropriate timeout
                timeout = 600 if tts_service == "hira_dia_tts" else 30
                
                tts_resp = requests.post(f"http://localhost:{tts_port}/synthesize", 
                                       json=tts_payload, timeout=timeout)
                if tts_resp.status_code == 200:
                    print(f"    ✅ {tts_service.replace('_', ' ').title()} audio generated")
                else:
                    print(f"    ❌ {tts_service} failed: {tts_resp.status_code}")
                    return False
            
            print(f"    🎉 {combo_name} end-to-end flow completed successfully")
            return True
            
        except Exception as e:
            print(f"    ❌ End-to-end test error: {e}")
            return False
    
    def test_gpt_compatibility(self) -> bool:
        """Deep compatibility test for GPT LLM service"""
        try:
            print("    🧠 Running GPT Advanced Compatibility Test...")
            port = self.services["gpt_llm"]["port"]
            
            # Test 1: Basic health and capability
            health_resp = requests.get(f"http://localhost:{port}/health", timeout=5)
            if health_resp.status_code != 200:
                print("    ❌ GPT health check failed")
                return False
            
            # Test 2: Multiple conversation contexts
            test_scenarios = [
                {
                    "text": "What is artificial intelligence?",
                    "domain_context": "technology",
                    "expected_keywords": ["ai", "artificial", "intelligence", "technology"]
                },
                {
                    "text": "How do I check my account balance?",
                    "domain_context": "banking",
                    "expected_keywords": ["account", "balance", "bank"]
                },
                {
                    "text": "Can you help me with a loan application?",
                    "domain_context": "banking",
                    "expected_keywords": ["loan", "application", "help"]
                }
            ]
            
            success_count = 0
            for i, scenario in enumerate(test_scenarios):
                payload = {
                    "text": scenario["text"],
                    "domain_context": scenario.get("domain_context"),
                    "use_cache": False,  # Force fresh responses
                    "max_tokens": 150,
                    "temperature": 0.7
                }
                
                response = requests.post(f"http://localhost:{port}/generate", 
                                       json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get("response", "").lower()
                    
                    # Check if response contains expected keywords
                    keyword_matches = sum(1 for keyword in scenario["expected_keywords"] 
                                        if keyword.lower() in response_text)
                    
                    if keyword_matches > 0:
                        print(f"    ✅ Scenario {i+1}: Response relevant ({keyword_matches} keywords matched)")
                        success_count += 1
                    else:
                        print(f"    ⚠️ Scenario {i+1}: Response may be off-topic")
                    
                    # Check metadata
                    if "processing_time_seconds" in data:
                        proc_time = data["processing_time_seconds"]
                        if proc_time < 10:  # Reasonable response time
                            print(f"    ✅ Scenario {i+1}: Good response time ({proc_time:.2f}s)")
                        else:
                            print(f"    ⚠️ Scenario {i+1}: Slow response ({proc_time:.2f}s)")
                else:
                    print(f"    ❌ Scenario {i+1}: HTTP {response.status_code}")
            
            # Test 3: Conversation memory (if supported)
            print("    🔄 Testing conversation context...")
            conversation_payload = {
                "text": "What did we just discuss?",
                "conversation_history": [
                    {"role": "user", "content": "Tell me about banking services"},
                    {"role": "assistant", "content": "Banking services include checking accounts, savings, and loans"}
                ],
                "use_cache": False
            }
            
            conv_response = requests.post(f"http://localhost:{port}/generate", 
                                        json=conversation_payload, timeout=30)
            
            if conv_response.status_code == 200:
                conv_data = conv_response.json()
                if "banking" in conv_data.get("response", "").lower():
                    print("    ✅ Conversation context maintained")
                    success_count += 1
                else:
                    print("    ⚠️ Conversation context not maintained")
            
            # Test 4: Cache functionality
            print("    💾 Testing cache functionality...")
            cache_test_payload = {"text": "Test cache functionality", "use_cache": True}
            
            # First request (should miss cache)
            first_resp = requests.post(f"http://localhost:{port}/generate", 
                                     json=cache_test_payload, timeout=30)
            # Second request (should hit cache)
            second_resp = requests.post(f"http://localhost:{port}/generate", 
                                      json=cache_test_payload, timeout=30)
            
            if (first_resp.status_code == 200 and second_resp.status_code == 200):
                first_time = first_resp.json().get("processing_time_seconds", 0)
                second_time = second_resp.json().get("processing_time_seconds", 0)
                
                if second_time < first_time * 0.5:  # Cache should be significantly faster
                    print("    ✅ Cache functionality working")
                    success_count += 1
                else:
                    print("    ⚠️ Cache may not be working optimally")
            
            print(f"    📊 GPT Compatibility: {success_count}/7 tests passed")
            return success_count >= 5  # Require at least 5/7 tests to pass
            
        except Exception as e:
            print(f"    ❌ GPT compatibility test error: {e}")
            return False
    
    def create_test_audio_data(self) -> bytes:
        """Create test audio data for STT testing (following existing patterns)"""
        # Create a simple WAV file with silence
        # This is a minimal WAV header + some audio data
        sample_rate = 16000
        duration = 1  # 1 second
        num_samples = sample_rate * duration
        
        # WAV header (44 bytes)
        wav_header = b'RIFF'
        wav_header += (36 + num_samples * 2).to_bytes(4, 'little')  # File size
        wav_header += b'WAVE'
        wav_header += b'fmt '
        wav_header += (16).to_bytes(4, 'little')  # PCM chunk size
        wav_header += (1).to_bytes(2, 'little')   # Audio format (PCM)
        wav_header += (1).to_bytes(2, 'little')   # Number of channels
        wav_header += sample_rate.to_bytes(4, 'little')  # Sample rate
        wav_header += (sample_rate * 2).to_bytes(4, 'little')  # Byte rate
        wav_header += (2).to_bytes(2, 'little')   # Block align
        wav_header += (16).to_bytes(2, 'little')  # Bits per sample
        wav_header += b'data'
        wav_header += (num_samples * 2).to_bytes(4, 'little')  # Data chunk size
        
        # Add some audio data (silence)
        audio_data = b'\x00\x00' * num_samples
        
        return wav_header + audio_data
    
    def run_comprehensive_tests(self) -> dict:
        """Run all tests and return results"""
        print("🧪 Starting Comprehensive Independent Services Test Suite")
        print("=" * 70)
        
        all_results = []
        
        # Get available services first
        available = self.get_available_services()
        
        if not any(available.values()):
            print("\n❌ No services are currently running!")
            print("Please start some services first using the enhanced_service_manager.py")
            return {"total_tests": 0, "passed_tests": 0, "results": []}
        
        # Test individual services
        if available.get("stt"):
            all_results.append(self.test_stt_service())
        
        for tts_service in ["kokoro_tts", "hira_dia_tts"]:
            if available.get(tts_service):
                all_results.append(self.test_tts_service(tts_service))
        
        for llm_service in ["mistral_llm", "gpt_llm"]:
            if available.get(llm_service):
                all_results.append(self.test_llm_service(llm_service))
        
        # Test combinations
        all_results.append(self.test_service_combinations())
        
        # Generate summary
        total_tests = len(all_results)
        passed_tests = sum(1 for result in all_results if result["passed"])
        
        print(f"\n📊 Test Summary")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for result in all_results:
            status_icon = "✅" if result["passed"] else "❌"
            print(f"\n{status_icon} {result['name']}:")
            for detail in result["details"]:
                print(f"    {detail}")
        
        if self.audio_output_dir.exists() and list(self.audio_output_dir.glob("*.wav")):
            print(f"\n🔊 Audio files saved to: {self.audio_output_dir}")
            audio_files = list(self.audio_output_dir.glob("*.wav"))
            for audio_file in audio_files[-3:]:  # Show last 3 files
                print(f"    {audio_file.name}")
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "results": all_results
        }

def main():
    """Main entry point following existing test patterns"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Independent Services Test Suite")
    parser.add_argument("--service", choices=["stt", "kokoro_tts", "hira_dia_tts", "mistral_llm", "gpt_llm"],
                        help="Test specific service only")
    parser.add_argument("--combinations", action="store_true", help="Test service combinations only")
    parser.add_argument("--quick", action="store_true", help="Quick tests only")
    
    args = parser.parse_args()
    
    tester = IndependentServiceTester()
    
    try:
        if args.service:
            # Test specific service
            print(f"🎯 Testing specific service: {args.service}")
            if args.service == "stt":
                result = tester.test_stt_service()
            elif args.service in ["kokoro_tts", "hira_dia_tts"]:
                result = tester.test_tts_service(args.service)
            elif args.service in ["mistral_llm", "gpt_llm"]:
                result = tester.test_llm_service(args.service)
            
            print(f"\n📊 Result for {result['name']}:")
            for detail in result["details"]:
                print(f"  {detail}")
                
        elif args.combinations:
            # Test combinations only
            result = tester.test_service_combinations()
            print(f"\n📊 Combination Test Results:")
            for detail in result["details"]:
                print(f"  {detail}")
                
        else:
            # Run comprehensive tests (default)
            tester.run_comprehensive_tests()
    
    except KeyboardInterrupt:
        print("\n👋 Testing interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    main()
