#!/usr/bin/env python3
"""
Enhanced Voice Conversation Test with Switchable TTS
====================================================

This test demonstrates the complete voice conversation pipeline with the enhanced TTS system:
1. Record audio from microphone (Speech Input)
2. Speech-to-Text (STT) using Whisper
3. Large Language Model (LLM) processing using Mistral
4. Text-to-Speech (TTS) response using switchable engines (Kokoro/Nari Dia)
5. Audio playback of the response

Usage:
    python tests/test_enhanced_voice_conversation.py [--engine kokoro|nari_dia]

Controls:
    - Press SPACE to start/stop recording
    - Press 'q' to quit
    - Press 'r' to replay last response
    - Press 'k' to switch to Kokoro (fast)
    - Press 'n' to switch to Nari Dia (quality)
    - Press 's' to show TTS status
"""

import asyncio
import os
import sys
import time
import threading
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np

# Audio libraries
try:
    import pyaudio
    import wave
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ PyAudio not available. Audio recording/playback will be simulated.")

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("⚠️ Keyboard library not available. Using input() for controls.")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import enhanced TTS system
from voicebot_orchestrator.enhanced_tts_manager import EnhancedTTSManager, TTSEngine
# Import real local implementations
from faster_whisper_stt import FasterWhisperSTT
from real_llm import RealOllamaLLM

class AudioRecorder:
    """Audio recording utility using PyAudio."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_format = pyaudio.paInt16
        self.is_recording = False
        self.audio_frames = []
        self.audio_interface = None
        
        if AUDIO_AVAILABLE:
            self.audio_interface = pyaudio.PyAudio()
    
    def start_recording(self):
        """Start recording audio from microphone."""
        if not AUDIO_AVAILABLE:
            print("🎤 [SIMULATED] Recording started...")
            self.is_recording = True
            return
        
        self.is_recording = True
        self.audio_frames = []
        
        try:
            stream = self.audio_interface.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print("🎤 Recording started... (Press SPACE again to stop)")
            
            while self.is_recording:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_frames.append(data)
            
            stream.stop_stream()
            stream.close()
            print("🎤 Recording stopped.")
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
    
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
    
    def save_recording(self, filename="recording.wav"):
        """Save recorded audio to file with proper WAV format."""
        if not AUDIO_AVAILABLE:
            print(f"🎤 [SIMULATED] Saved recording to {filename}")
            # Create a dummy file for testing
            with open(filename, 'w') as f:
                f.write("dummy audio data")
            return filename
        
        if not self.audio_frames:
            print("❌ No audio recorded")
            return None
        
        try:
            # Save as proper WAV file
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio_interface.get_sample_size(self.audio_format))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b''.join(self.audio_frames))
            
            print(f"💾 Recording saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Save error: {e}")
            return None
    
    def cleanup(self):
        """Clean up audio resources."""
        if AUDIO_AVAILABLE and self.audio_interface:
            self.audio_interface.terminate()

class EnhancedVoiceConversationTest:
    """Enhanced voice conversation test with switchable TTS engines."""
    
    def __init__(self, initial_engine=TTSEngine.KOKORO):
        self.recorder = AudioRecorder()
        self.stt = None
        self.llm = None
        self.tts_manager = None
        self.last_response_file = None
        self.conversation_history = []
        self.initial_engine = initial_engine
        
        print("🎭 Enhanced Voice Conversation Test")
        print("=" * 50)
    
    async def initialize_components(self):
        """Initialize all voice processing components."""
        print("⏳ Initializing voice processing components...")
        
        # Initialize STT
        print("1️⃣ Initializing Speech-to-Text (Faster-Whisper)...")
        try:
            self.stt = FasterWhisperSTT()
            print("✅ STT initialized")
        except Exception as e:
            print(f"❌ STT failed: {e}")
            return False
        
        # Initialize LLM
        print("2️⃣ Initializing Language Model (Ollama/Mistral)...")
        try:
            self.llm = RealOllamaLLM()
            print("✅ LLM initialized")
        except Exception as e:
            print(f"❌ LLM failed: {e}")
            return False
        
        # Initialize Enhanced TTS Manager
        print("3️⃣ Initializing Enhanced TTS Manager...")
        try:
            self.tts_manager = EnhancedTTSManager()
            await self.tts_manager.initialize_engines(load_kokoro=True, load_nari=True)
            
            # Set initial engine
            if self.initial_engine in self.tts_manager.get_available_engines():
                self.tts_manager.set_engine(self.initial_engine)
            
            print("✅ Enhanced TTS Manager initialized")
        except Exception as e:
            print(f"❌ TTS Manager failed: {e}")
            return False
        
        print("🎉 All components initialized successfully!")
        return True
    
    def show_controls(self):
        """Display available controls."""
        print("\n🎮 CONTROLS:")
        print("=" * 30)
        print("SPACE - Start/Stop recording")
        print("'q'   - Quit conversation")
        print("'r'   - Replay last response")
        print("'k'   - Switch to Kokoro TTS (fast)")
        print("'n'   - Switch to Nari Dia TTS (quality)")
        print("'s'   - Show TTS engine status")
        print("=" * 30)
    
    def show_tts_status(self):
        """Show current TTS engine status."""
        if not self.tts_manager:
            print("❌ TTS Manager not initialized")
            return
        
        print("\n📊 TTS ENGINE STATUS:")
        print("=" * 40)
        
        current = self.tts_manager.get_current_engine()
        available = self.tts_manager.get_available_engines()
        
        for engine in [TTSEngine.KOKORO, TTSEngine.NARI_DIA]:
            status = "✅ LOADED" if engine in available else "❌ NOT AVAILABLE"
            active = "🎯 ACTIVE" if engine == current else ""
            
            if engine == TTSEngine.KOKORO:
                print(f"🚀 Kokoro TTS: {status} {active}")
                print("   • Speed: ~0.5s generation")
                print("   • Voice: af_bella (professional female)")
            elif engine == TTSEngine.NARI_DIA:
                print(f"🎭 Nari Dia-1.6B: {status} {active}")
                print("   • Speed: ~30s+ generation")
                print("   • Voice: Adaptive dialogue-focused")
        
        print(f"\n🔧 Current Engine: {current.value}")
        print("=" * 40)
    
    async def switch_tts_engine(self, engine: TTSEngine):
        """Switch TTS engine."""
        if not self.tts_manager:
            print("❌ TTS Manager not initialized")
            return
        
        try:
            available = self.tts_manager.get_available_engines()
            if engine not in available:
                print(f"❌ Engine {engine.value} not available")
                return
            
            self.tts_manager.set_engine(engine)
            
            if engine == TTSEngine.KOKORO:
                print("🚀 Switched to Kokoro TTS (fast, real-time)")
            elif engine == TTSEngine.NARI_DIA:
                print("🎭 Switched to Nari Dia TTS (high quality, slow)")
                print("⚠️  Note: Nari Dia takes 30+ seconds per response")
                
        except Exception as e:
            print(f"❌ Engine switch failed: {e}")
    
    async def process_voice_input(self):
        """Process recorded voice input through the pipeline."""
        # Save recording
        timestamp = datetime.now().strftime("%H%M%S")
        audio_file = f"voice_input_{timestamp}.wav"
        
        saved_file = self.recorder.save_recording(audio_file)
        if not saved_file:
            return
        
        try:
            # Speech-to-Text
            print("🎯 Processing speech-to-text...")
            start_stt = time.time()
            
            user_text = await asyncio.to_thread(self.stt.transcribe_audio, saved_file)
            stt_time = time.time() - start_stt
            
            if not user_text or user_text.strip() == "":
                print("❌ No speech detected")
                return
            
            print(f"📝 You said: \"{user_text}\"")
            print(f"⏱️  STT took: {stt_time:.2f}s")
            
            # Language Model Processing
            print("🧠 Processing with language model...")
            start_llm = time.time()
            
            # Add to conversation history
            self.conversation_history.append(f"User: {user_text}")
            
            # Get LLM response
            llm_response = await asyncio.to_thread(self.llm.get_response, user_text)
            llm_time = time.time() - start_llm
            
            print(f"🤖 AI Response: \"{llm_response}\"")
            print(f"⏱️  LLM took: {llm_time:.2f}s")
            
            # Add to conversation history
            self.conversation_history.append(f"AI: {llm_response}")
            
            # Text-to-Speech
            current_engine = self.tts_manager.get_current_engine()
            print(f"🎤 Generating speech with {current_engine.value.upper()}...")
            
            if current_engine == TTSEngine.NARI_DIA:
                print("⏳ Using Nari Dia - this will take 30+ seconds...")
            
            start_tts = time.time()
            
            audio_bytes, tts_time, used_engine = await self.tts_manager.generate_speech(
                llm_response, 
                save_path=f"ai_response_{timestamp}.wav"
            )
            
            print(f"🔊 Generated {len(audio_bytes)} bytes in {tts_time:.2f}s using {used_engine}")
            
            # Performance summary
            total_time = stt_time + llm_time + tts_time
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"   STT: {stt_time:.2f}s")
            print(f"   LLM: {llm_time:.2f}s") 
            print(f"   TTS: {tts_time:.2f}s")
            print(f"   Total: {total_time:.2f}s")
            
            if total_time < 2.0:
                print("🚀 EXCELLENT: Near real-time response!")
            elif total_time < 5.0:
                print("⚡ GOOD: Fast response time")
            elif total_time < 15.0:
                print("🔄 ACCEPTABLE: Reasonable response time")
            else:
                print("⏳ SLOW: Consider using Kokoro for faster responses")
            
            self.last_response_file = f"ai_response_{timestamp}.wav"
            
            # Audio playback (simulated for now)
            print("🔊 [Audio playback would occur here]")
            
        except Exception as e:
            print(f"❌ Voice processing failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up input file
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass
    
    async def replay_last_response(self):
        """Replay the last TTS response."""
        if not self.last_response_file or not os.path.exists(self.last_response_file):
            print("❌ No previous response to replay")
            return
        
        print(f"🔄 Replaying: {self.last_response_file}")
        print("🔊 [Audio playback would occur here]")
    
    async def run_conversation(self):
        """Run the interactive voice conversation."""
        if not await self.initialize_components():
            print("❌ Failed to initialize components")
            return
        
        self.show_controls()
        self.show_tts_status()
        
        print(f"\n🎬 Voice conversation started with {self.tts_manager.get_current_engine().value.upper()} TTS!")
        print("Press SPACE to start recording your first message...")
        
        if not KEYBOARD_AVAILABLE:
            print("⚠️ Keyboard controls not available, using input mode")
            return await self.run_input_mode()
        
        # Keyboard-based interaction
        recording = False
        
        while True:
            try:
                if keyboard.is_pressed('space'):
                    if not recording:
                        # Start recording
                        recording = True
                        thread = threading.Thread(target=self.recorder.start_recording)
                        thread.start()
                        
                        # Wait for space release and press again
                        while keyboard.is_pressed('space'):
                            time.sleep(0.1)
                        
                        print("Press SPACE again to stop recording...")
                        
                        # Wait for next space press
                        while not keyboard.is_pressed('space'):
                            if keyboard.is_pressed('q'):
                                self.recorder.stop_recording()
                                recording = False
                                print("👋 Conversation ended")
                                return
                            time.sleep(0.1)
                        
                        # Stop recording
                        self.recorder.stop_recording()
                        recording = False
                        thread.join()
                        
                        # Process the recording
                        await self.process_voice_input()
                        
                        print("\nPress SPACE to record another message...")
                
                elif keyboard.is_pressed('q'):
                    print("👋 Conversation ended")
                    break
                    
                elif keyboard.is_pressed('r'):
                    await self.replay_last_response()
                    time.sleep(0.5)  # Prevent repeat
                    
                elif keyboard.is_pressed('k'):
                    await self.switch_tts_engine(TTSEngine.KOKORO)
                    time.sleep(0.5)  # Prevent repeat
                    
                elif keyboard.is_pressed('n'):
                    await self.switch_tts_engine(TTSEngine.NARI_DIA)
                    time.sleep(0.5)  # Prevent repeat
                    
                elif keyboard.is_pressed('s'):
                    self.show_tts_status()
                    time.sleep(0.5)  # Prevent repeat
                
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\n👋 Conversation interrupted")
                break
    
    async def run_input_mode(self):
        """Run in input mode when keyboard is not available."""
        print("📝 Input mode - type commands:")
        print("'record' - Record message, 'quit' - Exit, 'kokoro'/'nari' - Switch TTS")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command in ['quit', 'q', 'exit']:
                    break
                elif command == 'record':
                    print("🎤 Recording for 5 seconds...")
                    self.recorder.start_recording()
                    await asyncio.sleep(5)
                    self.recorder.stop_recording()
                    await self.process_voice_input()
                elif command == 'kokoro':
                    await self.switch_tts_engine(TTSEngine.KOKORO)
                elif command == 'nari':
                    await self.switch_tts_engine(TTSEngine.NARI_DIA)
                elif command == 'status':
                    self.show_tts_status()
                elif command == 'replay':
                    await self.replay_last_response()
                else:
                    print("❌ Unknown command")
                    
            except KeyboardInterrupt:
                print("\n👋 Conversation interrupted")
                break
    
    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up resources...")
        
        if self.recorder:
            self.recorder.cleanup()
        
        if self.tts_manager:
            self.tts_manager.cleanup()
        
        print("✅ Cleanup complete")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced Voice Conversation Test")
    parser.add_argument("--engine", choices=["kokoro", "nari_dia"], default="kokoro",
                       help="Initial TTS engine (default: kokoro)")
    
    args = parser.parse_args()
    
    # Convert string to enum
    initial_engine = TTSEngine.KOKORO if args.engine == "kokoro" else TTSEngine.NARI_DIA
    
    test = EnhancedVoiceConversationTest(initial_engine)
    
    try:
        await test.run_conversation()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
