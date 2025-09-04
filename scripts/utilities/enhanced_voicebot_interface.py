"""
Enhanced Voicebot Interface with Audio Testing
Integrates with independent microservices and provides audio I/O
"""
import asyncio
import logging
import time
import base64
import json
import wave
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ PyAudio not available. Audio I/O will be simulated.")

try:
    import requests
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
    print("❌ Requests not available. Cannot connect to services.")

class VoicebotInterface:
    """Enhanced interface for voicebot with audio capabilities"""
    
    def __init__(self):
        self.running = False
        self.services_config = {
            "stt": "http://localhost:8001",
            "kokoro_tts": "http://localhost:8011",
            "hira_dia_tts": "http://localhost:8012", 
            "mistral_llm": "http://localhost:8021",
            "gpt_llm": "http://localhost:8022"
        }
        
        # Audio settings
        self.audio_format = pyaudio.paInt16 if AUDIO_AVAILABLE else None
        self.channels = 1
        self.sample_rate = 16000
        self.chunk_size = 1024
        
        # Current configuration
        self.current_tts = "kokoro"  # or "hira_dia"
        self.current_llm = "mistral"  # or "gpt"
        
        # Conversation history
        self.conversation_history = []
    
    def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy"""
        if not HTTP_AVAILABLE:
            return False
            
        try:
            url = self.services_config.get(service_name)
            if not url:
                return False
                
            response = requests.get(f"{url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_services(self) -> Dict[str, bool]:
        """Get status of all services"""
        status = {}
        for service_name in self.services_config.keys():
            status[service_name] = self.check_service_health(service_name)
        return status
    
    def test_tts_service(self, tts_engine: str) -> bool:
        """Test TTS service with sample text"""
        service_name = f"{tts_engine}_tts"
        
        if not self.check_service_health(service_name):
            print(f"❌ {tts_engine.title()} TTS service not available")
            return False
        
        print(f"🔊 Testing {tts_engine.title()} TTS...")
        
        try:
            url = self.services_config[service_name]
            payload = {
                "text": f"Hello, this is a test of {tts_engine} text-to-speech engine.",
                "return_audio": True
            }
            
            response = requests.post(f"{url}/synthesize", json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                metadata = result.get("metadata", {})
                
                print(f"✅ {tts_engine.title()} TTS Test Successful!")
                print(f"   Processing time: {metadata.get('processing_time_seconds', 0):.3f}s")
                print(f"   Engine used: {metadata.get('engine_used', 'unknown')}")
                print(f"   Audio size: {len(result.get('audio_base64', ''))} chars")
                
                # Optionally save audio for testing
                if result.get("audio_base64"):
                    self.save_test_audio(result["audio_base64"], f"{tts_engine}_test.wav")
                
                return True
            else:
                print(f"❌ {tts_engine.title()} TTS Test Failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ {tts_engine.title()} TTS Test Error: {e}")
            return False
    
    def test_llm_service(self, llm_model: str) -> bool:
        """Test LLM service with sample query"""
        service_name = f"{llm_model}_llm"
        
        if not self.check_service_health(service_name):
            print(f"❌ {llm_model.title()} LLM service not available")
            return False
        
        print(f"🧠 Testing {llm_model.title()} LLM...")
        
        try:
            url = self.services_config[service_name]
            payload = {
                "text": f"Hello, can you tell me about {llm_model} language model capabilities?",
                "use_cache": True,
                "max_tokens": 100
            }
            
            response = requests.post(f"{url}/generate", json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ {llm_model.title()} LLM Test Successful!")
                print(f"   Processing time: {result.get('processing_time_seconds', 0):.3f}s")
                print(f"   Model used: {result.get('model_used', 'unknown')}")
                print(f"   Cache hit: {result.get('cache_hit', False)}")
                print(f"   Response: {result.get('response', '')[:100]}...")
                
                return True
            else:
                print(f"❌ {llm_model.title()} LLM Test Failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ {llm_model.title()} LLM Test Error: {e}")
            return False
    
    def save_test_audio(self, audio_base64: str, filename: str):
        """Save audio data to file for testing"""
        try:
            audio_bytes = base64.b64decode(audio_base64)
            output_dir = Path("demos/audio_output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = output_dir / filename
            with open(filepath, "wb") as f:
                f.write(audio_bytes)
            
            print(f"💾 Audio saved to: {filepath}")
        except Exception as e:
            print(f"⚠️ Failed to save audio: {e}")
    
    def record_audio(self, duration: float = 5.0) -> Optional[bytes]:
        """Record audio from microphone"""
        if not AUDIO_AVAILABLE:
            print("❌ PyAudio not available for recording")
            return None
        
        print(f"🎤 Recording for {duration} seconds... Speak now!")
        
        try:
            audio = pyaudio.PyAudio()
            
            stream = audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            for _ in range(int(self.sample_rate / self.chunk_size * duration)):
                data = stream.read(self.chunk_size)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            print("✅ Recording complete!")
            
            # Convert to bytes
            audio_bytes = b''.join(frames)
            return audio_bytes
            
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            return None
    
    def play_audio(self, audio_base64: str) -> bool:
        """Play audio through speakers/headset"""
        if not AUDIO_AVAILABLE:
            print("❌ PyAudio not available for playback")
            return False
        
        try:
            # Decode audio
            audio_bytes = base64.b64decode(audio_base64)
            
            print("🔊 Playing audio...")
            
            audio = pyaudio.PyAudio()
            
            stream = audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )
            
            # Play audio in chunks
            chunk_size = 1024
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                stream.write(chunk)
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            print("✅ Audio playback complete!")
            return True
            
        except Exception as e:
            print(f"❌ Audio playback failed: {e}")
            return False
    
    def process_voice_conversation(self, audio_data: bytes) -> Dict[str, Any]:
        """Process complete voice conversation pipeline"""
        print(f"\n🎯 Processing voice conversation with {self.current_tts.title()} TTS + {self.current_llm.title()} LLM")
        
        pipeline_start = time.time()
        
        try:
            # Step 1: Speech-to-Text
            print("🎤 Transcribing speech...")
            
            if not self.check_service_health("stt"):
                return {"success": False, "error": "STT service not available"}
            
            files = {"audio": ("recording.wav", audio_data, "audio/wav")}
            response = requests.post(f"{self.services_config['stt']}/transcribe", files=files, timeout=30)
            
            if response.status_code != 200:
                return {"success": False, "error": f"STT failed: {response.status_code}"}
            
            stt_result = response.json()
            transcript = stt_result.get("transcript", "")
            print(f"📝 Transcript: '{transcript}'")
            
            if not transcript.strip():
                return {"success": False, "error": "No speech detected"}
            
            # Step 2: Generate LLM Response
            print(f"🧠 Generating response with {self.current_llm.title()}...")
            
            llm_service = f"{self.current_llm}_llm"
            if not self.check_service_health(llm_service):
                return {"success": False, "error": f"{self.current_llm.title()} LLM service not available"}
            
            llm_payload = {
                "text": transcript,
                "conversation_history": self.conversation_history[-10:],  # Last 10 exchanges
                "use_cache": True
            }
            
            response = requests.post(f"{self.services_config[llm_service]}/generate", json=llm_payload, timeout=60)
            
            if response.status_code != 200:
                return {"success": False, "error": f"LLM failed: {response.status_code}"}
            
            llm_result = response.json()
            response_text = llm_result.get("response", "")
            print(f"💭 Response: '{response_text}'")
            
            # Step 3: Text-to-Speech
            print(f"🔊 Synthesizing with {self.current_tts.title()}...")
            
            tts_service = f"{self.current_tts}_tts"
            if not self.check_service_health(tts_service):
                return {"success": False, "error": f"{self.current_tts.title()} TTS service not available"}
            
            tts_payload = {
                "text": response_text,
                "return_audio": True
            }
            
            response = requests.post(f"{self.services_config[tts_service]}/synthesize", json=tts_payload, timeout=300)
            
            if response.status_code != 200:
                return {"success": False, "error": f"TTS failed: {response.status_code}"}
            
            tts_result = response.json()
            audio_base64 = tts_result.get("audio_base64")
            
            # Update conversation history
            self.conversation_history.append({
                "user": transcript,
                "assistant": response_text,
                "timestamp": time.time()
            })
            
            total_time = time.time() - pipeline_start
            
            return {
                "success": True,
                "transcript": transcript,
                "response_text": response_text,
                "audio_base64": audio_base64,
                "total_time": total_time,
                "services_used": {
                    "tts": self.current_tts,
                    "llm": self.current_llm
                }
            }
            
        except Exception as e:
            return {"success": False, "error": f"Pipeline error: {str(e)}"}
    
    def make_voice_call(self):
        """Make a voice call (record -> process -> play)"""
        print("\n📞 Starting Voice Call Session")
        print(f"Current setup: {self.current_tts.title()} TTS + {self.current_llm.title()} LLM")
        
        while True:
            print("\n🎤 Press Enter to start recording (or 'q' to quit)...")
            user_input = input().strip().lower()
            
            if user_input == 'q':
                break
            
            # Record audio
            audio_data = self.record_audio(duration=5.0)
            if not audio_data:
                continue
            
            # Process conversation
            result = self.process_voice_conversation(audio_data)
            
            if result["success"]:
                print(f"✅ Conversation processed in {result['total_time']:.3f}s")
                
                # Play response
                if result.get("audio_base64"):
                    self.play_audio(result["audio_base64"])
                
            else:
                print(f"❌ Conversation failed: {result['error']}")
        
        print("📞 Voice call session ended")
    
    def run_interface(self):
        """Run the main interface"""
        self.running = True
        
        while self.running:
            try:
                print("\n🎭 Enhanced Voicebot Interface")
                print("=" * 50)
                
                # Show current config
                print(f"Current Setup: {self.current_tts.title()} TTS + {self.current_llm.title()} LLM")
                
                # Show service status
                services = self.get_available_services()
                healthy_services = [name for name, healthy in services.items() if healthy]
                print(f"Available Services: {len(healthy_services)}/{len(services)}")
                
                print("\n🔧 Configuration:")
                print("1. Set Fast Setup (Kokoro TTS + Mistral LLM)")
                print("2. Set Quality Setup (Hira Dia TTS + Mistral LLM)")
                print("3. Set Premium Setup (Hira Dia TTS + GPT LLM)")
                print("4. Set Testing Setup (Kokoro TTS + GPT LLM)")
                
                print("\n🧪 Testing:")
                print("5. Test Current TTS Engine")
                print("6. Test Current LLM Model")
                print("7. Test All Available Services")
                print("8. Check Service Health")
                
                print("\n📞 Voice Interaction:")
                print("9. Make Voice Call (Record -> Process -> Play)")
                print("10. Text Conversation (Type -> Process -> Play)")
                print("11. Record Audio Test")
                print("12. Play Test Audio")
                
                print("\n⚙️ Management:")
                print("13. View Conversation History")
                print("14. Clear Conversation History")
                print("15. Exit")
                
                choice = input("\nEnter your choice (1-15): ").strip()
                
                if choice == "1":
                    self.current_tts = "kokoro"
                    self.current_llm = "mistral"
                    print("✅ Set to Fast Setup")
                elif choice == "2":
                    self.current_tts = "hira_dia"
                    self.current_llm = "mistral"
                    print("✅ Set to Quality Setup")
                elif choice == "3":
                    self.current_tts = "hira_dia"
                    self.current_llm = "gpt"
                    print("✅ Set to Premium Setup")
                elif choice == "4":
                    self.current_tts = "kokoro"
                    self.current_llm = "gpt"
                    print("✅ Set to Testing Setup")
                elif choice == "5":
                    self.test_tts_service(self.current_tts)
                elif choice == "6":
                    self.test_llm_service(self.current_llm)
                elif choice == "7":
                    self.test_all_services()
                elif choice == "8":
                    self.show_service_health()
                elif choice == "9":
                    self.make_voice_call()
                elif choice == "10":
                    self.text_conversation()
                elif choice == "11":
                    self.record_audio_test()
                elif choice == "12":
                    self.play_test_audio()
                elif choice == "13":
                    self.show_conversation_history()
                elif choice == "14":
                    self.conversation_history.clear()
                    print("✅ Conversation history cleared")
                elif choice == "15":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-15.")
                
                if choice != "9":  # Don't pause after voice call
                    input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
    
    def test_all_services(self):
        """Test all available services"""
        print("\n🧪 Testing All Available Services...")
        
        services = self.get_available_services()
        healthy_services = [name for name, healthy in services.items() if healthy]
        
        if not healthy_services:
            print("❌ No services available to test")
            return
        
        # Test TTS services
        for tts_engine in ["kokoro", "hira_dia"]:
            if f"{tts_engine}_tts" in healthy_services:
                self.test_tts_service(tts_engine)
        
        # Test LLM services
        for llm_model in ["mistral", "gpt"]:
            if f"{llm_model}_llm" in healthy_services:
                self.test_llm_service(llm_model)
    
    def show_service_health(self):
        """Show detailed service health"""
        print("\n🏥 Service Health Status:")
        services = self.get_available_services()
        
        for service_name, healthy in services.items():
            status_icon = "✅" if healthy else "❌"
            url = self.services_config[service_name]
            print(f"  {status_icon} {service_name}: {url}")
    
    def text_conversation(self):
        """Text-based conversation (Type -> Process -> Play)"""
        print("\n💬 Text Conversation Mode")
        print("Type your message, and I'll respond with audio")
        
        while True:
            user_text = input("\nYou: ").strip()
            
            if user_text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_text:
                continue
            
            # Simulate audio data for text input
            fake_audio = user_text.encode('utf-8')
            
            # Create a mock STT response
            try:
                # Generate LLM response
                llm_service = f"{self.current_llm}_llm"
                if not self.check_service_health(llm_service):
                    print(f"❌ {self.current_llm.title()} LLM service not available")
                    continue
                
                llm_payload = {
                    "text": user_text,
                    "conversation_history": self.conversation_history[-10:],
                    "use_cache": True
                }
                
                response = requests.post(f"{self.services_config[llm_service]}/generate", json=llm_payload, timeout=60)
                
                if response.status_code != 200:
                    print(f"❌ LLM failed: {response.status_code}")
                    continue
                
                llm_result = response.json()
                response_text = llm_result.get("response", "")
                print(f"Assistant: {response_text}")
                
                # Generate TTS
                tts_service = f"{self.current_tts}_tts"
                if self.check_service_health(tts_service):
                    tts_payload = {
                        "text": response_text,
                        "return_audio": True
                    }
                    
                    tts_response = requests.post(f"{self.services_config[tts_service]}/synthesize", json=tts_payload, timeout=300)
                    
                    if tts_response.status_code == 200:
                        tts_result = tts_response.json()
                        audio_base64 = tts_result.get("audio_base64")
                        if audio_base64:
                            self.play_audio(audio_base64)
                
                # Update conversation history
                self.conversation_history.append({
                    "user": user_text,
                    "assistant": response_text,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def record_audio_test(self):
        """Test audio recording"""
        print("\n🎤 Audio Recording Test")
        duration = float(input("Enter recording duration (seconds, default 5): ") or "5")
        
        audio_data = self.record_audio(duration)
        if audio_data:
            # Save test recording
            timestamp = int(time.time())
            filename = f"test_recording_{timestamp}.wav"
            
            try:
                output_dir = Path("demos/audio_output")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                filepath = output_dir / filename
                with open(filepath, "wb") as f:
                    f.write(audio_data)
                
                print(f"💾 Recording saved to: {filepath}")
            except Exception as e:
                print(f"⚠️ Failed to save recording: {e}")
    
    def play_test_audio(self):
        """Play a test audio file"""
        print("\n🔊 Audio Playback Test")
        
        # Look for test audio files
        audio_dir = Path("demos/audio_output")
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.wav"))
            if audio_files:
                print("Available audio files:")
                for i, file in enumerate(audio_files, 1):
                    print(f"  {i}. {file.name}")
                
                try:
                    choice = int(input("Select file number: ")) - 1
                    if 0 <= choice < len(audio_files):
                        filepath = audio_files[choice]
                        with open(filepath, "rb") as f:
                            audio_data = f.read()
                        
                        # Convert to base64 for playback
                        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                        self.play_audio(audio_base64)
                    else:
                        print("❌ Invalid choice")
                except ValueError:
                    print("❌ Invalid input")
            else:
                print("❌ No audio files found")
        else:
            print("❌ Audio directory not found")
    
    def show_conversation_history(self):
        """Show conversation history"""
        print("\n📝 Conversation History:")
        
        if not self.conversation_history:
            print("No conversations yet")
            return
        
        for i, exchange in enumerate(self.conversation_history[-10:], 1):
            timestamp = time.strftime("%H:%M:%S", time.localtime(exchange["timestamp"]))
            print(f"\n{i}. [{timestamp}]")
            print(f"   You: {exchange['user']}")
            print(f"   Assistant: {exchange['assistant']}")

def main():
    """Main entry point"""
    print("🎭 Enhanced Voicebot Interface")
    print("=" * 50)
    
    if not HTTP_AVAILABLE:
        print("❌ HTTP requests not available. Please install 'requests' package.")
        return
    
    if not AUDIO_AVAILABLE:
        print("⚠️ Audio I/O not available. Please install 'pyaudio' package for full functionality.")
        print("Some features will be simulated.")
    
    interface = VoicebotInterface()
    
    try:
        interface.run_interface()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
