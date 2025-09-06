#!/usr/bin/env python3
"""
Live Microphone Testing Suite for WebSocket Services
Provides comprehensive testing of STT→LLM→TTS pipeline with real microphone input
"""

import asyncio
import json
import logging
import sys
import os
import time
import threading
import wave
import pyaudio
import websockets
from typing import Dict, Any, List, Optional
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MicrophoneTestSuite:
    """Complete microphone testing suite for WebSocket services"""
    
    def __init__(self):
        self.orchestrator_ws_url = "ws://localhost:9000"
        self.orchestrator_http_url = "http://localhost:8080"
        
        # Audio configuration
        self.audio_config = {
            'format': pyaudio.paInt16,
            'channels': 1,
            'rate': 16000,
            'chunk': 1024,
            'record_seconds': 5
        }
        
        # Test configurations
        self.test_configs = {
            "stt_only": {
                "name": "🎤 STT Only Test",
                "description": "Record audio and test speech-to-text conversion",
                "services_required": ["stt"],
                "test_flow": ["record_audio", "send_to_stt", "display_transcript"]
            },
            "stt_to_llm": {
                "name": "🎤➡️🧠 STT → LLM Test", 
                "description": "Speech to text, then LLM response generation",
                "services_required": ["stt", "llm"],
                "test_flow": ["record_audio", "send_to_stt", "send_to_llm", "display_response"]
            },
            "llm_to_tts": {
                "name": "🧠➡️🗣️ LLM → TTS Test",
                "description": "Text input to LLM, then speech synthesis",
                "services_required": ["llm", "tts"], 
                "test_flow": ["text_input", "send_to_llm", "send_to_tts", "play_audio"]
            },
            "tts_only": {
                "name": "🗣️ TTS Only Test",
                "description": "Text input directly to speech synthesis",
                "services_required": ["tts"],
                "test_flow": ["text_input", "send_to_tts", "play_audio"]
            },
            "full_pipeline": {
                "name": "🎤➡️🧠➡️🗣️ Full STT → LLM → TTS Pipeline",
                "description": "Complete voice conversation: speak, get AI response as speech",
                "services_required": ["stt", "llm", "tts"],
                "test_flow": ["record_audio", "send_to_stt", "send_to_llm", "send_to_tts", "play_audio"]
            },
            "conversation_loop": {
                "name": "🔄 Continuous Conversation Mode",
                "description": "Ongoing voice conversation with the AI",
                "services_required": ["stt", "llm", "tts"],
                "test_flow": ["conversation_loop"]
            }
        }
        
        self.websocket = None
        self.session_id = None
        self.audio_interface = None
        
    def safe_print(self, text: str):
        """Safe print function that handles Unicode characters"""
        try:
            print(text)
        except UnicodeEncodeError:
            safe_text = text.encode('ascii', 'replace').decode('ascii')
            print(safe_text)
    
    def check_audio_system(self) -> bool:
        """Check if audio system is available"""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            
            # Check for input devices
            input_devices = []
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    input_devices.append(device_info['name'])
            
            p.terminate()
            
            if input_devices:
                self.safe_print(f"✅ Audio system available with {len(input_devices)} input devices")
                self.safe_print(f"   Primary input: {input_devices[0]}")
                return True
            else:
                self.safe_print("❌ No audio input devices found")
                return False
                
        except ImportError:
            self.safe_print("❌ PyAudio not available. Install with: pip install pyaudio")
            return False
        except Exception as e:
            self.safe_print(f"❌ Audio system check failed: {e}")
            return False
    
    def check_services_availability(self, required_services: List[str]) -> Dict[str, bool]:
        """Check which required services are available"""
        try:
            response = requests.get(f"{self.orchestrator_http_url}/health", timeout=5)
            if response.status_code != 200:
                return {service: False for service in required_services}
                
            # Also check registered services
            services_response = requests.get(f"{self.orchestrator_http_url}/services", timeout=5)
            registered_services = []
            
            if services_response.status_code == 200:
                services_data = services_response.json()
                if isinstance(services_data, list):
                    registered_services = [s.get('service_type', '') for s in services_data]
                else:
                    registered_services = [s.get('service_type', '') for s in services_data.get('services', [])]
            
            service_status = {}
            for service in required_services:
                service_status[service] = service in registered_services
            
            return service_status
            
        except Exception as e:
            self.safe_print(f"❌ Service check failed: {e}")
            return {service: False for service in required_services}
    
    async def connect_websocket(self) -> bool:
        """Connect to WebSocket orchestrator"""
        try:
            self.websocket = await websockets.connect(self.orchestrator_ws_url)
            self.session_id = f"mic_test_{int(time.time())}"
            
            # Wait for session_start message
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            if response_data.get('type') == 'session_start':
                self.safe_print(f"✅ Connected to orchestrator - Session: {self.session_id}")
                return True
            else:
                self.safe_print(f"❌ Unexpected response: {response_data}")
                return False
                
        except Exception as e:
            self.safe_print(f"❌ WebSocket connection failed: {e}")
            return False
    
    async def disconnect_websocket(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.safe_print("📡 Disconnected from orchestrator")
    
    def record_audio(self, duration: int = 5) -> bytes:
        """Record audio from microphone"""
        try:
            import pyaudio
            
            p = pyaudio.PyAudio()
            
            self.safe_print(f"🎙️ Recording audio for {duration} seconds...")
            self.safe_print("   Speak clearly into your microphone!")
            
            stream = p.open(
                format=self.audio_config['format'],
                channels=self.audio_config['channels'],
                rate=self.audio_config['rate'],
                input=True,
                frames_per_buffer=self.audio_config['chunk']
            )
            
            frames = []
            for i in range(0, int(self.audio_config['rate'] / self.audio_config['chunk'] * duration)):
                data = stream.read(self.audio_config['chunk'])
                frames.append(data)
                
                # Show progress
                progress = (i + 1) / (self.audio_config['rate'] / self.audio_config['chunk'] * duration)
                progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
                print(f"\\r   Recording: [{progress_bar}] {progress*100:.1f}%", end="", flush=True)
            
            print()  # New line after progress bar
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            audio_data = b''.join(frames)
            self.safe_print(f"✅ Recorded {len(audio_data)} bytes of audio")
            
            return audio_data
            
        except Exception as e:
            self.safe_print(f"❌ Audio recording failed: {e}")
            return b''
    
    async def send_audio_to_stt(self, audio_data: bytes) -> str:
        """Send audio data to STT service"""
        try:
            if not self.websocket:
                raise Exception("WebSocket not connected")
            
            import base64
            audio_b64 = base64.b64encode(audio_data).decode()
            
            message = {
                "type": "audio_chunk",
                "session_id": self.session_id,
                "timestamp": time.time(),
                "data": {
                    "audio_data": audio_b64,
                    "format": "wav",
                    "sample_rate": self.audio_config['rate'],
                    "channels": self.audio_config['channels'],
                    "chunk_index": 0,
                    "is_final": True  # This is the complete audio recording
                }
            }
            
            self.safe_print("📤 Sending audio to STT service...")
            await self.websocket.send(json.dumps(message))
            
            # Wait for transcript response
            self.safe_print("⏳ Waiting for transcript...")
            transcript = ""
            
            timeout_counter = 0
            while timeout_counter < 30:  # 30 second timeout
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    response_data = json.loads(response)
                    
                    if response_data.get('type') in ['transcript_partial', 'transcript_final']:
                        transcript = response_data.get('data', {}).get('text', '')
                        if response_data.get('type') == 'transcript_final':
                            break
                        elif transcript:
                            print(f"\\r   Partial: {transcript}", end="", flush=True)
                    
                except asyncio.TimeoutError:
                    timeout_counter += 1
                    continue
            
            if transcript:
                print()  # New line after partial transcript
                self.safe_print(f"🎯 Transcript: \"{transcript}\"")
                return transcript
            else:
                self.safe_print("❌ No transcript received")
                return ""
                
        except Exception as e:
            self.safe_print(f"❌ STT request failed: {e}")
            return ""
    
    async def send_text_to_llm(self, text: str) -> str:
        """Send text to LLM service"""
        try:
            if not self.websocket:
                raise Exception("WebSocket not connected")
            
            message = {
                "type": "text_input",
                "session_id": self.session_id,
                "timestamp": time.time(),
                "data": {
                    "text": text,
                    "stream_tokens": True
                }
            }
            
            self.safe_print(f"📤 Sending to LLM: \"{text}\"")
            await self.websocket.send(json.dumps(message))
            
            # Wait for LLM response
            self.safe_print("🧠 Generating AI response...")
            response_text = ""
            
            timeout_counter = 0
            while timeout_counter < 30:
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    response_data = json.loads(response)
                    
                    response_type = response_data.get('type')
                    if response_type == 'llm_stream_start':
                        self.safe_print("   🔄 LLM started responding...")
                    elif response_type == 'llm_token':
                        token = response_data.get('data', {}).get('token', '')
                        response_text += token
                        print(f"\\r   Response: {response_text}", end="", flush=True)
                    elif response_type == 'llm_stream_complete':
                        print()  # New line
                        self.safe_print("✅ LLM response complete")
                        break
                    elif response_type in ['llm_response', 'text_response']:
                        response_text = response_data.get('data', {}).get('text', response_text)
                        break
                        
                except asyncio.TimeoutError:
                    timeout_counter += 1
                    continue
            
            if response_text:
                self.safe_print(f"🎯 AI Response: \"{response_text}\"")
                return response_text
            else:
                self.safe_print("❌ No LLM response received")
                return ""
                
        except Exception as e:
            self.safe_print(f"❌ LLM request failed: {e}")
            return ""
    
    async def send_text_to_tts(self, text: str) -> bytes:
        """Send text to TTS service and get audio"""
        try:
            if not self.websocket:
                raise Exception("WebSocket not connected")
            
            message = {
                "type": "tts_request",
                "session_id": self.session_id,
                "timestamp": time.time(),
                "data": {
                    "text": text,
                    "voice": "neutral"
                }
            }
            
            self.safe_print(f"📤 Sending to TTS: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            await self.websocket.send(json.dumps(message))
            
            # Wait for TTS response
            self.safe_print("🗣️ Generating speech...")
            audio_data = None
            
            timeout_counter = 0
            while timeout_counter < 30:
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    response_data = json.loads(response)
                    
                    if response_data.get('type') == 'audio_output':
                        audio_hex = response_data.get('data', {}).get('audio_data', '')
                        if audio_hex:
                            import base64
                            try:
                                # Try hex decoding first
                                audio_data = bytes.fromhex(audio_hex)
                            except ValueError:
                                # Fallback to base64 if not hex
                                audio_data = base64.b64decode(audio_hex)
                            break
                        
                except asyncio.TimeoutError:
                    timeout_counter += 1
                    continue
            
            if audio_data:
                self.safe_print(f"✅ Received {len(audio_data)} bytes of audio")
                return audio_data
            else:
                self.safe_print("❌ No TTS audio received")
                return b''
                
        except Exception as e:
            self.safe_print(f"❌ TTS request failed: {e}")
            return b''
    
    def play_audio(self, audio_data: bytes) -> bool:
        """Play audio data through speakers"""
        try:
            if not audio_data:
                self.safe_print("❌ No audio data to play")
                return False
            
            import pyaudio
            
            p = pyaudio.PyAudio()
            
            # Create temporary wav file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Read the audio file
                wf = wave.open(temp_file_path, 'rb')
                
                stream = p.open(
                    format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                
                self.safe_print("🔊 Playing audio response...")
                
                # Play audio in chunks
                chunk = 1024
                data = wf.readframes(chunk)
                while data:
                    stream.write(data)
                    data = wf.readframes(chunk)
                
                stream.stop_stream()
                stream.close()
                wf.close()
                
                self.safe_print("✅ Audio playback complete")
                
            finally:
                # Clean up temp file
                os.unlink(temp_file_path)
                p.terminate()
            
            return True
            
        except Exception as e:
            self.safe_print(f"❌ Audio playback failed: {e}")
            return False
    
    def get_text_input(self, prompt: str = "Enter text") -> str:
        """Get text input from user"""
        try:
            text = input(f"💭 {prompt}: ").strip()
            return text
        except KeyboardInterrupt:
            return ""
    
    async def run_stt_only_test(self):
        """Run STT-only test"""
        self.safe_print("")
        self.safe_print("🎤 Starting STT Only Test")
        self.safe_print("=" * 50)
        
        # Record audio
        audio_data = self.record_audio()
        if not audio_data:
            return False
        
        # Send to STT
        transcript = await self.send_audio_to_stt(audio_data)
        
        if transcript:
            self.safe_print(f"✅ STT Test Complete - Transcript: \"{transcript}\"")
            return True
        else:
            self.safe_print("❌ STT Test Failed")
            return False
    
    async def run_stt_to_llm_test(self):
        """Run STT → LLM test"""
        self.safe_print("")
        self.safe_print("🎤➡️🧠 Starting STT → LLM Test")
        self.safe_print("=" * 50)
        
        # Record audio
        audio_data = self.record_audio()
        if not audio_data:
            return False
        
        # Send to STT
        transcript = await self.send_audio_to_stt(audio_data)
        if not transcript:
            return False
        
        # Send to LLM
        response = await self.send_text_to_llm(transcript)
        
        if response:
            self.safe_print(f"✅ STT → LLM Test Complete")
            return True
        else:
            self.safe_print("❌ STT → LLM Test Failed")
            return False
    
    async def run_llm_to_tts_test(self):
        """Run LLM → TTS test"""
        self.safe_print("")
        self.safe_print("🧠➡️🗣️ Starting LLM → TTS Test")
        self.safe_print("=" * 50)
        
        # Get text input
        text = self.get_text_input("Enter text for AI to respond to")
        if not text:
            return False
        
        # Send to LLM
        response = await self.send_text_to_llm(text)
        if not response:
            return False
        
        # Send to TTS
        audio_data = await self.send_text_to_tts(response)
        if not audio_data:
            return False
        
        # Play audio
        success = self.play_audio(audio_data)
        
        if success:
            self.safe_print("✅ LLM → TTS Test Complete")
            return True
        else:
            self.safe_print("❌ LLM → TTS Test Failed")
            return False
    
    async def run_tts_only_test(self):
        """Run TTS-only test"""
        self.safe_print("")
        self.safe_print("🗣️ Starting TTS Only Test")
        self.safe_print("=" * 50)
        
        # Get text input
        text = self.get_text_input("Enter text to convert to speech")
        if not text:
            return False
        
        # Send to TTS
        audio_data = await self.send_text_to_tts(text)
        if not audio_data:
            return False
        
        # Play audio
        success = self.play_audio(audio_data)
        
        if success:
            self.safe_print("✅ TTS Only Test Complete")
            return True
        else:
            self.safe_print("❌ TTS Only Test Failed")
            return False
    
    async def run_full_pipeline_test(self):
        """Run complete STT → LLM → TTS pipeline test"""
        self.safe_print("")
        self.safe_print("🎤➡️🧠➡️🗣️ Starting Full Pipeline Test")
        self.safe_print("=" * 60)
        
        # Record audio
        audio_data = self.record_audio()
        if not audio_data:
            return False
        
        # Send to STT
        transcript = await self.send_audio_to_stt(audio_data)
        if not transcript:
            return False
        
        # Send to LLM
        response = await self.send_text_to_llm(transcript)
        if not response:
            return False
        
        # Send to TTS
        audio_data = await self.send_text_to_tts(response)
        if not audio_data:
            return False
        
        # Play audio
        success = self.play_audio(audio_data)
        
        if success:
            self.safe_print("✅ Full Pipeline Test Complete")
            self.safe_print("🎉 Voice conversation successful!")
            return True
        else:
            self.safe_print("❌ Full Pipeline Test Failed")
            return False
    
    async def run_conversation_loop(self):
        """Run continuous conversation mode"""
        self.safe_print("")
        self.safe_print("🔄 Starting Continuous Conversation Mode")
        self.safe_print("=" * 60)
        self.safe_print("💡 Press Ctrl+C to exit conversation mode")
        self.safe_print("🎙️ Each round: Speak → AI processes → AI responds")
        
        round_num = 1
        
        try:
            while True:
                self.safe_print(f"")
                self.safe_print(f"--- Conversation Round {round_num} ---")
                
                # Record audio
                self.safe_print("🎤 Ready to record (press Enter when ready, Ctrl+C to exit)")
                try:
                    input()
                except KeyboardInterrupt:
                    break
                
                audio_data = self.record_audio(duration=7)  # Longer recording for conversation
                if not audio_data:
                    continue
                
                # Process through pipeline
                transcript = await self.send_audio_to_stt(audio_data)
                if not transcript:
                    continue
                
                response = await self.send_text_to_llm(transcript)
                if not response:
                    continue
                
                audio_response = await self.send_text_to_tts(response)
                if audio_response:
                    self.play_audio(audio_response)
                
                round_num += 1
                
        except KeyboardInterrupt:
            self.safe_print("")
            self.safe_print("🔄 Conversation mode ended by user")
    
    def show_test_menu(self):
        """Show available tests menu"""
        self.safe_print("")
        self.safe_print("🧪 Microphone Testing Suite")
        self.safe_print("=" * 50)
        
        for i, (test_id, config) in enumerate(self.test_configs.items(), 1):
            self.safe_print(f"  {i}. {config['name']}")
            self.safe_print(f"     {config['description']}")
            
            # Show service requirements
            services_str = " + ".join([s.upper() for s in config['services_required']])
            self.safe_print(f"     Services: {services_str}")
            
            self.safe_print("")
        
        self.safe_print("  0. Back to Main Menu")
        self.safe_print("-" * 50)
        
        return list(self.test_configs.keys())
    
    async def run_selected_test(self, test_id: str) -> bool:
        """Run the selected test"""
        config = self.test_configs[test_id]
        
        # Check service availability
        service_status = self.check_services_availability(config['services_required'])
        missing_services = [s for s, available in service_status.items() if not available]
        
        if missing_services:
            self.safe_print(f"❌ Missing services: {', '.join(missing_services)}")
            self.safe_print("   Please start required services first")
            return False
        
        # Connect to WebSocket
        if not await self.connect_websocket():
            return False
        
        try:
            # Run the appropriate test
            if test_id == "stt_only":
                return await self.run_stt_only_test()
            elif test_id == "stt_to_llm":
                return await self.run_stt_to_llm_test()
            elif test_id == "llm_to_tts":
                return await self.run_llm_to_tts_test()
            elif test_id == "tts_only":
                return await self.run_tts_only_test()
            elif test_id == "full_pipeline":
                return await self.run_full_pipeline_test()
            elif test_id == "conversation_loop":
                await self.run_conversation_loop()
                return True
            else:
                self.safe_print(f"❌ Unknown test: {test_id}")
                return False
                
        finally:
            await self.disconnect_websocket()
    
    async def main_menu(self):
        """Main testing menu"""
        while True:
            try:
                # Check audio system
                if not self.check_audio_system():
                    self.safe_print("❌ Audio system not available. Cannot run microphone tests.")
                    input("Press Enter to return to main launcher...")
                    return
                
                test_ids = self.show_test_menu()
                
                choice = input("Enter your choice (0-6): ").strip()
                
                if choice == "0":
                    return
                
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(test_ids):
                        test_id = test_ids[choice_idx]
                        self.safe_print(f"")
                        self.safe_print(f"🚀 Running {self.test_configs[test_id]['name']}")
                        
                        success = await self.run_selected_test(test_id)
                        
                        if success:
                            self.safe_print("")
                            self.safe_print("✅ Test completed successfully!")
                        else:
                            self.safe_print("")
                            self.safe_print("❌ Test failed")
                        
                        input("\nPress Enter to continue...")
                    else:
                        self.safe_print("❌ Invalid choice")
                        
                except ValueError:
                    self.safe_print("❌ Invalid input")
                    
            except KeyboardInterrupt:
                self.safe_print("")
                self.safe_print("👋 Microphone testing cancelled")
                return

async def run_microphone_tests():
    """Entry point for microphone testing"""
    test_suite = MicrophoneTestSuite()
    await test_suite.main_menu()

if __name__ == "__main__":
    asyncio.run(run_microphone_tests())
