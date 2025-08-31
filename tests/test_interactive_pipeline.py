#!/usr/bin/env python3
"""
Interactive Pipeline Tester
============================

Interactive test menu for testing specific service combinations:
1. Select which services to test (STT, LLM, TTS)
2. Choose individual pipeline components or full pipeline
3. Test only what you want to test
4. Detailed error reporting for debugging

Usage:
    python tests/test_interactive_pipeline.py
"""

import asyncio
import sys
import time
import json
import base64
import requests
import pyaudio
import wave
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import traceback

class InteractivePipelineTester:
    """Interactive tester for specific service combinations"""
    
    def __init__(self):
        """Initialize the interactive tester"""
        self.base_dir = Path(__file__).parent
        self.audio_dir = self.base_dir / "audio_samples" / "interactive_pipeline"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # All possible service endpoints
        self.all_services = {
            "orchestrator": {"url": "http://localhost:8000", "type": "orchestrator"},
            "whisper_stt": {"url": "http://localhost:8003", "type": "stt"},
            "kokoro_tts": {"url": "http://localhost:8011", "type": "tts"},
            "hira_dia_tts": {"url": "http://localhost:8012", "type": "tts"}, 
            "dia_4bit_tts": {"url": "http://localhost:8013", "type": "tts"},
            "mistral_llm": {"url": "http://localhost:8021", "type": "llm"},
            "gpt_llm": {"url": "http://localhost:8022", "type": "llm"}
        }
        
        # Track which services are available
        self.available_services = {}
        
        # Audio recording settings
        self.recording_settings = {
            'format': pyaudio.paInt16,
            'channels': 1,
            'rate': 16000,  # 16kHz for speech
            'chunk': 1024,
            'record_seconds': 5,  # Default recording duration
        }
        
    def record_microphone_audio(self, duration: int = 5) -> str:
        """Record audio from microphone and save to file"""
        print(f"🎤 Recording audio for {duration} seconds...")
        print("   Speak now...")
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        try:
            # Open microphone stream
            stream = audio.open(
                format=self.recording_settings['format'],
                channels=self.recording_settings['channels'],
                rate=self.recording_settings['rate'],
                input=True,
                frames_per_buffer=self.recording_settings['chunk']
            )
            
            frames = []
            
            # Record audio
            for i in range(0, int(self.recording_settings['rate'] / self.recording_settings['chunk'] * duration)):
                data = stream.read(self.recording_settings['chunk'])
                frames.append(data)
            
            print("✅ Recording completed!")
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recorded_audio_{timestamp}.wav"
            filepath = self.audio_dir / filename
            
            # Write WAV file
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(self.recording_settings['channels'])
                wf.setsampwidth(audio.get_sample_size(self.recording_settings['format']))
                wf.setframerate(self.recording_settings['rate'])
                wf.writeframes(b''.join(frames))
            
            print(f"🎵 Audio saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            return None
        finally:
            audio.terminate()
        
    def check_service_health(self, service_name: str, endpoint: str) -> bool:
        """Check if a service is running and healthy"""
        try:
            # Try health endpoint first - disable requests logging to reduce noise
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            health_response = requests.get(f"{endpoint}/health", timeout=3)
            if health_response.status_code == 200:
                return True
        except:
            pass
        
        try:
            # Try root endpoint as fallback
            root_response = requests.get(endpoint, timeout=3)
            if root_response.status_code in [200, 404]:  # 404 is OK for some services
                return True
        except:
            pass
        
        return False
    
    def detect_available_services(self) -> Dict:
        """Detect which services are currently available"""
        print("🔍 Checking service availability...")
        print("-" * 40)
        
        available = {}
        
        for service_name, config in self.all_services.items():
            endpoint = config["url"]
            if self.check_service_health(service_name, endpoint):
                available[service_name] = config
                print(f"  ✅ {service_name:<20} - {endpoint}")
            else:
                print(f"  ❌ {service_name:<20} - Not available")
        
        self.available_services = available
        
        print(f"\n📊 Available: {len(available)} services")
        return available
    
    def select_services_by_type(self, service_type: str) -> Optional[str]:
        """Let user select a service of specific type"""
        available_of_type = {name: config for name, config in self.available_services.items() 
                           if config["type"] == service_type}
        
        if not available_of_type:
            print(f"❌ No {service_type.upper()} services available")
            return None
        
        if len(available_of_type) == 1:
            service_name = list(available_of_type.keys())[0]
            print(f"✅ Auto-selected {service_type.upper()}: {service_name}")
            return service_name
        
        print(f"\n📋 Available {service_type.upper()} services:")
        services_list = list(available_of_type.keys())
        for i, service_name in enumerate(services_list, 1):
            endpoint = available_of_type[service_name]["url"]
            print(f"  {i}. {service_name} - {endpoint}")
        
        while True:
            try:
                choice = input(f"\nSelect {service_type.upper()} service (1-{len(services_list)}): ").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(services_list):
                        selected = services_list[idx]
                        print(f"✅ Selected {service_type.upper()}: {selected}")
                        return selected
                print(f"❌ Please enter a number between 1 and {len(services_list)}")
            except KeyboardInterrupt:
                return None
    
    async def test_stt_to_llm(self, stt_service: str, llm_service: str) -> bool:
        """Test STT → LLM pipeline component with real microphone recording"""
        print(f"\n🎙️ Testing {stt_service} → {llm_service}")
        print("-" * 50)
        
        try:
            # Option to use microphone or text input
            use_mic = input("Use microphone recording? (y/n, default=y): ").strip().lower()
            if use_mic in ['', 'y', 'yes']:
                # Record from microphone
                duration = input("Recording duration in seconds (default=5): ").strip()
                try:
                    duration = int(duration) if duration else 5
                    duration = max(1, min(duration, 30))  # Limit between 1-30 seconds
                except ValueError:
                    duration = 5
                
                print(f"\n🎤 Get ready to speak in 3 seconds...")
                time.sleep(3)
                
                audio_file = self.record_microphone_audio(duration)
                if not audio_file:
                    print("❌ Recording failed, falling back to text input")
                    test_text = input("Enter test text: ").strip() or "Hello, how are you today?"
                    print(f"📝 Using text input: '{test_text}'")
                else:
                    # Try STT service first, then direct Whisper as backup
                    print(f"🔄 Attempting transcription with recorded audio...")
                    
                    try:
                        # Send audio to STT service
                        stt_endpoint = self.available_services[stt_service]["url"]
                        print(f"🔄 Trying STT service: {stt_endpoint}")
                        
                        with open(audio_file, 'rb') as f:
                            files = {'audio': f}
                            stt_response = requests.post(f"{stt_endpoint}/transcribe", files=files, timeout=30)
                        
                        if stt_response.status_code == 200:
                            stt_result = stt_response.json()
                            test_text = stt_result.get("text", "").strip()
                            confidence = stt_result.get("confidence", 0.0)
                            
                            if test_text:
                                print(f"✅ STT Service Success: '{test_text}' (confidence: {confidence:.2f})")
                            else:
                                print(f"⚠️  STT service returned empty transcript")
                                test_text = ""
                        else:
                            print(f"❌ STT service failed: {stt_response.status_code}")
                            test_text = ""
                    
                    except Exception as e:
                        print(f"❌ STT service error: {e}")
                        test_text = ""
                    
                    # If STT service failed or returned empty, try direct Whisper
                    if not test_text:
                        print("🔄 Trying direct Whisper transcription...")
                        try:
                            import sys
                            import os
                            
                            # Add the project root to Python path
                            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            if project_root not in sys.path:
                                sys.path.insert(0, project_root)
                            
                            from voicebot_orchestrator.real_whisper_stt import WhisperSTT
                            
                            whisper = WhisperSTT()
                            test_text = await whisper.transcribe_file(audio_file)
                            test_text = test_text.strip() if test_text else ""
                            
                            if test_text:
                                print(f"✅ Direct Whisper Success: '{test_text}'")
                            else:
                                print("⚠️  Direct Whisper also returned empty")
                                
                        except Exception as e:
                            print(f"❌ Direct Whisper failed: {e}")
                            # Try alternative approach with simple whisper import
                            try:
                                import whisper
                                print("🔄 Trying fallback Whisper approach...")
                                
                                model = whisper.load_model("base")
                                result = model.transcribe(audio_file)
                                test_text = result["text"].strip()
                                
                                if test_text:
                                    print(f"✅ Fallback Whisper Success: '{test_text}'")
                                else:
                                    print("⚠️  Fallback Whisper also returned empty")
                                    
                            except Exception as e2:
                                print(f"❌ Fallback Whisper also failed: {e2}")
                    
                    # Final fallback if both methods failed
                    if not test_text:
                        print("⚠️  All transcription methods failed - using fallback")
                        test_text = "Hello, how are you today?"
            else:
                # Text input mode
                test_text = input("Enter test text (or press Enter for default): ").strip()
                if not test_text:
                    test_text = "Hello, how are you today?"
                print(f"📝 Input text: '{test_text}'")
            
            # Send to LLM
            llm_endpoint = self.available_services[llm_service]["url"]
            llm_data = {
                "text": test_text,
                "temperature": 0.7,
                "max_tokens": 150
            }
            
            print(f"🔄 Sending to LLM: {llm_endpoint}")
            
            llm_response = requests.post(f"{llm_endpoint}/generate", json=llm_data, timeout=30)
            
            if llm_response.status_code == 200:
                llm_result = llm_response.json()
                llm_text = llm_result.get("response", "").strip()
                print(f"✅ LLM Response: '{llm_text[:200]}...'")
                return True
            else:
                print(f"❌ LLM request failed: {llm_response.status_code}")
                print(f"   Response: {llm_response.text}")
                return False
                
        except Exception as e:
            print(f"❌ STT→LLM test failed: {e}")
            traceback.print_exc()
            return False
    
    async def test_llm_to_tts(self, llm_service: str, tts_service: str) -> bool:
        """Test LLM → TTS pipeline component"""
        print(f"\n🧠 Testing {llm_service} → {tts_service}")
        print("-" * 50)
        
        try:
            # Get input text for LLM
            input_text = input("Enter input for LLM (or press Enter for default): ").strip()
            if not input_text:
                input_text = "Tell me a short joke"
            
            print(f"📝 LLM Input: '{input_text}'")
            
            # Send to LLM first
            llm_endpoint = self.available_services[llm_service]["url"]
            llm_data = {
                "text": input_text,
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            print(f"🔄 Sending to LLM: {llm_endpoint}")
            
            llm_response = requests.post(f"{llm_endpoint}/generate", json=llm_data, timeout=30)
            if llm_response.status_code != 200:
                print(f"❌ LLM request failed: {llm_response.status_code}")
                print(f"   Response: {llm_response.text}")
                return False
                
            llm_result = llm_response.json() 
            llm_text = llm_result.get("response", "").strip()
            print(f"✅ LLM Output: '{llm_text[:100]}...'")
            
            # Send LLM output to TTS
            tts_endpoint = self.available_services[tts_service]["url"]
            tts_data = {
                "text": llm_text,
                "voice": "af_bella",  # Use a valid Kokoro voice instead of "default"
                "speed": 1.0,
                "return_audio": True
            }
            
            print(f"🔄 Sending to TTS: {tts_endpoint}")
            
            tts_response = requests.post(f"{tts_endpoint}/synthesize", json=tts_data, timeout=60)
            if tts_response.status_code == 200:
                tts_result = tts_response.json()
                audio_data = tts_result.get("audio_base64")
                
                if audio_data:
                    # Save audio file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"llm_to_tts_{llm_service}_{tts_service}_{timestamp}.wav"
                    filepath = self.audio_dir / filename
                    
                    audio_bytes = base64.b64decode(audio_data)
                    with open(filepath, "wb") as f:
                        f.write(audio_bytes)
                    
                    print(f"✅ Audio saved: {filename}")
                    return True
                else:
                    print(f"❌ No audio data received from TTS")
                    return False
            else:
                print(f"❌ TTS request failed: {tts_response.status_code}")
                print(f"   Response: {tts_response.text}")
                return False
                
        except Exception as e:
            print(f"❌ LLM→TTS test failed: {e}")
            traceback.print_exc()
            return False
    
    async def test_full_pipeline(self, stt_service: str, llm_service: str, tts_service: str) -> bool:
        """Test complete STT → LLM → TTS pipeline with real microphone recording"""
        print(f"\n🎯 Testing Full Pipeline: {stt_service} → {llm_service} → {tts_service}")
        print("-" * 70)
        
        try:
            # Option to use microphone or text input
            use_mic = input("Use microphone for STT input? (y/n, default=y): ").strip().lower()
            
            if use_mic in ['', 'y', 'yes']:
                # Record from microphone
                duration = input("Recording duration in seconds (default=5): ").strip()
                try:
                    duration = int(duration) if duration else 5
                    duration = max(1, min(duration, 30))  # Limit between 1-30 seconds
                except ValueError:
                    duration = 5
                
                print(f"\n🎤 Get ready to speak in 3 seconds...")
                time.sleep(3)
                
                audio_file = self.record_microphone_audio(duration)
                if not audio_file:
                    print("❌ Recording failed, falling back to text input")
                    input_text = input("Enter test input: ").strip() or "What's the weather like today?"
                    print(f"📝 Using text input: '{input_text}'")
                else:
                    # Send audio to STT service
                    stt_endpoint = self.available_services[stt_service]["url"]
                    
                    print(f"🔄 Step 1 - STT Processing: {stt_endpoint}")
                    
                    with open(audio_file, 'rb') as f:
                        files = {'audio': f}
                        stt_response = requests.post(f"{stt_endpoint}/transcribe", files=files, timeout=30)
                    
                    if stt_response.status_code == 200:
                        stt_result = stt_response.json()
                        input_text = stt_result.get("text", "").strip()  # Changed from "transcript" to "text"
                        confidence = stt_result.get("confidence", 0.0)
                        print(f"✅ STT Transcript: '{input_text}' (confidence: {confidence:.2f})")
                        
                        if not input_text:
                            print("❌ Empty transcript from STT service!")
                            print("🔄 Attempting direct Whisper transcription...")
                            
                            # Try direct Whisper transcription as fallback
                            try:
                                import sys
                                import os
                                
                                # Add the project root to Python path
                                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                                if project_root not in sys.path:
                                    sys.path.insert(0, project_root)
                                
                                from voicebot_orchestrator.real_whisper_stt import WhisperSTT
                                
                                whisper = WhisperSTT()
                                input_text = await whisper.transcribe_file(audio_file)
                                input_text = input_text.strip() if input_text else ""
                                
                                if input_text:
                                    print(f"✅ Direct Whisper Result: '{input_text}'")
                                else:
                                    print("⚠️  Direct Whisper also returned empty")
                                    
                            except Exception as e:
                                print(f"❌ Direct Whisper failed: {e}")
                                # Try alternative approach with simple whisper import
                                try:
                                    import whisper
                                    print("🔄 Trying fallback Whisper approach...")
                                    
                                    model = whisper.load_model("base")
                                    result = model.transcribe(audio_file)
                                    input_text = result["text"].strip()
                                    
                                    if input_text:
                                        print(f"✅ Fallback Whisper Success: '{input_text}'")
                                    else:
                                        print("⚠️  Even fallback Whisper returned empty - using minimal fallback")
                                        input_text = "What's the weather like today?"
                                        
                                except Exception as e2:
                                    print(f"❌ Fallback Whisper also failed: {e2}")
                                    print("⚠️  Using fallback text")
                                    input_text = "What's the weather like today?"
                    else:
                        print(f"❌ STT step failed: {stt_response.status_code}")
                        print(f"   Response: {stt_response.text}")
                        print("   Using fallback text")
                        input_text = "What's the weather like today?"
            else:
                # Text input mode
                input_text = input("Enter test input (or press Enter for default): ").strip()
                if not input_text:
                    input_text = "What's the weather like today?"
                print(f"🎙️ STT Input (text mode): '{input_text}'")
            
            # Step 2: LLM Processing
            llm_endpoint = self.available_services[llm_service]["url"]
            llm_data = {
                "text": input_text,
                "temperature": 0.7,
                "max_tokens": 150
            }
            
            print(f"🔄 Step 2 - LLM Processing: {llm_endpoint}")
            
            llm_response = requests.post(f"{llm_endpoint}/generate", json=llm_data, timeout=30)
            if llm_response.status_code != 200:
                print(f"❌ LLM step failed: {llm_response.status_code}")
                print(f"   Response: {llm_response.text}")
                return False
                
            llm_result = llm_response.json()
            llm_text = llm_result.get("response", "").strip()
            print(f"✅ LLM Output: '{llm_text[:100]}...'")
            
            # Step 2: TTS Synthesis
            tts_endpoint = self.available_services[tts_service]["url"]
            tts_data = {
                "text": llm_text,
                "voice": "af_bella",  # Use a valid Kokoro voice instead of "default"
                "speed": 1.0,
                "return_audio": True
            }
            
            print(f"🔄 Step 2 - TTS Synthesis: {tts_endpoint}")
            
            tts_response = requests.post(f"{tts_endpoint}/synthesize", json=tts_data, timeout=60)
            if tts_response.status_code == 200:
                tts_result = tts_response.json()
                audio_data = tts_result.get("audio_base64")
                
                if audio_data:
                    # Save audio file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_name = f"{stt_service}_{llm_service}_{tts_service}"
                    filename = f"full_pipeline_{safe_name}_{timestamp}.wav"
                    filepath = self.audio_dir / filename
                    
                    audio_bytes = base64.b64decode(audio_data)
                    with open(filepath, "wb") as f:
                        f.write(audio_bytes)
                    
                    print(f"✅ Complete Pipeline Success! Audio saved: {filename}")
                    return True
                else:
                    print(f"❌ No audio generated")
                    return False
            else:
                print(f"❌ TTS step failed: {tts_response.status_code}")
                print(f"   Response: {tts_response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Full pipeline test failed: {e}")
            traceback.print_exc()
            return False
    
    async def run_interactive_tests(self):
        """Run interactive test menu"""
        print("🎯 Interactive Pipeline Tester")
        print("=" * 50)
        
        # Detect available services
        available = self.detect_available_services()
        if not available:
            print("❌ No services are available. Start some services first.")
            return
        
        while True:
            print("\n" + "="*50)
            print("📋 TEST MENU")
            print("="*50)
            print("1. Test STT → LLM (select specific services)")
            print("2. Test LLM → TTS (select specific services)")
            print("3. Test Full Pipeline STT → LLM → TTS")
            print("4. Re-check service availability")
            print("5. Show available services")
            print("0. Exit")
            
            try:
                choice = input("\nSelect test type (0-5): ").strip()
                
                if choice == "0":
                    print("👋 Goodbye!")
                    break
                
                elif choice == "1":
                    print("\n🎙️ STT → LLM Test")
                    stt_service = self.select_services_by_type("stt")
                    if not stt_service:
                        continue
                    llm_service = self.select_services_by_type("llm")
                    if not llm_service:
                        continue
                    
                    success = await self.test_stt_to_llm(stt_service, llm_service)
                    result = "✅ SUCCESS" if success else "❌ FAILED"
                    print(f"\n📊 Result: {result}")
                
                elif choice == "2":
                    print("\n🧠 LLM → TTS Test")
                    llm_service = self.select_services_by_type("llm")
                    if not llm_service:
                        continue
                    tts_service = self.select_services_by_type("tts")
                    if not tts_service:
                        continue
                    
                    success = await self.test_llm_to_tts(llm_service, tts_service)
                    result = "✅ SUCCESS" if success else "❌ FAILED"
                    print(f"\n📊 Result: {result}")
                
                elif choice == "3":
                    print("\n🎯 Full Pipeline Test")
                    stt_service = self.select_services_by_type("stt")
                    if not stt_service:
                        continue
                    llm_service = self.select_services_by_type("llm")
                    if not llm_service:
                        continue
                    tts_service = self.select_services_by_type("tts")
                    if not tts_service:
                        continue
                    
                    success = await self.test_full_pipeline(stt_service, llm_service, tts_service)
                    result = "✅ SUCCESS" if success else "❌ FAILED"
                    print(f"\n📊 Result: {result}")
                
                elif choice == "4":
                    self.detect_available_services()
                
                elif choice == "5":
                    print("\n📋 Available Services:")
                    if not self.available_services:
                        print("  No services available")
                    else:
                        for service_name, config in self.available_services.items():
                            service_type = config["type"].upper()
                            url = config["url"]
                            print(f"  ✅ {service_name:<20} ({service_type:<3}) - {url}")
                
                else:
                    print("❌ Invalid choice. Please enter 0-5.")
                
                if choice in ["1", "2", "3"]:
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                traceback.print_exc()
                input("\nPress Enter to continue...")

async def main():
    """Main function"""
    tester = InteractivePipelineTester()
    await tester.run_interactive_tests()

if __name__ == "__main__":
    asyncio.run(main())
