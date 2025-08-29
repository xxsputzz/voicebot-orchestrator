#!/usr/bin/env python3
"""
Full Voice Conversation Test
============================

This test demonstrates the complete voice conversation pipeline:
1. Record audio from microphone (Speech Input)
2. Speech-to-Text (STT) using Whisper
3. Large Language Model (LLM) processing using Mistral
4. Text-to-Speech (TTS) response using Kokoro
5. Audio playback of the response

Usage:
    python tests/test_full_voice_conversation.py

Controls:
    - Press SPACE to start/stop recording
    - Press 'q' to quit
    - Press 'r' to replay last response
    - Press 'v' to change voice
"""

import asyncio
import os
import sys
import time
import threading
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

from voicebot_orchestrator.tts import KokoroTTS
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
            self.is_recording = False
    
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
    
    def get_audio_data(self):
        """Get recorded audio data as properly formatted WAV bytes."""
        if not AUDIO_AVAILABLE:
            # Return mock audio data for testing
            return b"mock_audio_data_" + str(int(time.time())).encode()
        
        if not self.audio_frames:
            print("⚠️ No audio frames recorded")
            return b""
        
        try:
            # Convert audio frames to numpy array for proper processing
            import numpy as np
            import wave
            import io
            
            # Combine all audio frames
            audio_data = b"".join(self.audio_frames)
            
            # Convert to numpy array (16-bit PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            print(f"📊 Recorded audio: {len(audio_array)} samples, {len(audio_data)} bytes")
            
            # Check if we have enough audio
            if len(audio_array) < 1000:  # Less than ~0.06 seconds at 16kHz
                print("⚠️ Audio recording too short")
                return b""
            
            # Create proper WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)  # 16kHz
                wav_file.writeframes(audio_data)
            
            # Get WAV data
            wav_data = wav_buffer.getvalue()
            print(f"📁 Created WAV: {len(wav_data)} bytes")
            
            return wav_data
            
        except Exception as e:
            print(f"❌ Error processing audio data: {e}")
            # Fall back to raw data
            audio_data = b"".join(self.audio_frames)
            return audio_data
    
    def save_audio_to_file(self, filename):
        """Save recorded audio to WAV file."""
        if not AUDIO_AVAILABLE or not self.audio_frames:
            return False
        
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio_interface.get_sample_size(self.audio_format))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b"".join(self.audio_frames))
            return True
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
            return False
    
    def __del__(self):
        if AUDIO_AVAILABLE and self.audio_interface:
            self.audio_interface.terminate()

class AudioPlayer:
    """Audio playback utility using PyAudio."""
    
    def __init__(self):
        self.audio_interface = None
        if AUDIO_AVAILABLE:
            self.audio_interface = pyaudio.PyAudio()
    
    def play_audio_data(self, audio_data, sample_rate=24000):
        """Play audio data through speakers."""
        if not AUDIO_AVAILABLE:
            print(f"🔊 [SIMULATED] Playing audio ({len(audio_data)} bytes)")
            time.sleep(2)  # Simulate playback time
            return
        
        try:
            # Audio data is WAV format, need to parse it
            # For simplicity, assume it's raw PCM data
            stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True
            )
            
            print(f"🔊 Playing audio ({len(audio_data)} bytes)...")
            
            # Skip WAV header if present (44 bytes typically)
            audio_start = 44 if len(audio_data) > 44 and audio_data[:4] == b'RIFF' else 0
            raw_audio = audio_data[audio_start:]
            
            # Play in chunks
            chunk_size = 1024
            for i in range(0, len(raw_audio), chunk_size):
                chunk = raw_audio[i:i+chunk_size]
                stream.write(chunk)
            
            stream.stop_stream()
            stream.close()
            print("🔊 Playback finished.")
            
        except Exception as e:
            print(f"❌ Playback error: {e}")
    
    def __del__(self):
        if AUDIO_AVAILABLE and self.audio_interface:
            self.audio_interface.terminate()

class VoiceConversationTest:
    """Full voice conversation test orchestrator."""
    
    def __init__(self):
        self.recorder = AudioRecorder()
        self.player = AudioPlayer()
        
        # Initialize AI components with REAL local implementations
        print("🔧 Initializing REAL local AI components...")
        
        # Faster-Whisper STT (2-4x faster than regular Whisper)
        self.stt = FasterWhisperSTT(
            model_name="base",          # Good balance of speed vs accuracy
            device="auto",              # Auto-detect best device
            compute_type="auto"         # Auto-optimize compute type
        )
        
        # Real Ollama/Mistral LLM  
        self.llm = RealOllamaLLM(
            model_name="mistral",
            host="localhost:11434",
            max_tokens=256,
            temperature=0.7
        )
        
        # Kokoro TTS (already real)
        self.tts = KokoroTTS(voice="af_bella", speed=1.0)
        
        # Available voices
        self.available_voices = [
            "af_bella", "af_nicole", "af_sarah", "af_alloy", 
            "af_aoede", "af_kore", "af_nova"
        ]
        self.current_voice_index = 0
        
        # Conversation state
        self.conversation_history = []
        self.last_response_audio = None
        self.session_start_time = time.time()
        
        # Performance tracking
        self.performance_log = []
        
        # Control flags
        self.is_running = True
        self.recording_thread = None
    
    def get_log_file_path(self):
        """Get path for performance log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"tests/audio_samples/conversation_log_{timestamp}.txt"
    
    def log_performance(self, stage, duration, details=""):
        """Log performance metrics."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "stage": stage,
            "duration": duration,
            "details": details
        }
        self.performance_log.append(log_entry)
        
        # Also write to file
        log_file = self.get_log_file_path()
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'a', encoding='utf-8') as f:
            if os.path.getsize(log_file) == 0:
                f.write("Full Voice Conversation Performance Log\\n")
                f.write("=" * 60 + "\\n")
                f.write("Timestamp\\t\\tStage\\t\\tDuration(s)\\tDetails\\n")
                f.write("-" * 60 + "\\n")
            
            f.write(f"{timestamp}\\t{stage:12}\\t{duration:.3f}\\t\\t{details}\\n")
    
    async def process_voice_input(self, audio_data):
        """Process voice input through the full pipeline."""
        pipeline_start = time.time()
        
        try:
            # Step 1: Speech-to-Text
            print("\\n🎤 Processing speech...")
            stt_start = time.time()
            user_text = await self.stt.transcribe_audio(audio_data)
            stt_duration = time.time() - stt_start
            
            print(f"📝 You said: \"{user_text}\"")
            self.log_performance("STT", stt_duration, f"Text: {user_text[:50]}...")
            
            # Step 2: Language Model Processing
            print("🧠 Thinking with real Mistral via Ollama...")
            llm_start = time.time()
            response_text = await self.llm.generate_response(
                user_text, 
                conversation_history=self.conversation_history
            )
            llm_duration = time.time() - llm_start
            
            print(f"💭 AI Response: \"{response_text}\"")
            self.log_performance("LLM", llm_duration, f"Response: {response_text[:50]}...")
            
            # Step 3: Text-to-Speech
            print(f"🎵 Generating speech with {self.tts.voice}...")
            tts_start = time.time()
            response_audio = await self.tts.synthesize_speech(response_text)
            tts_duration = time.time() - tts_start
            
            print(f"🔊 Generated {len(response_audio)} bytes of audio")
            self.log_performance("TTS", tts_duration, f"Voice: {self.tts.voice}, Size: {len(response_audio)}")
            
            # Step 4: Play Response
            print("🔊 Playing response...")
            playback_start = time.time()
            self.player.play_audio_data(response_audio)
            playback_duration = time.time() - playback_start
            
            self.log_performance("Playback", playback_duration, f"Audio played")
            
            # Store for replay
            self.last_response_audio = response_audio
            
            # Add to conversation history
            self.conversation_history.append({
                "human": user_text,
                "assistant": response_text,
                "timestamp": datetime.now().isoformat()
            })
            
            # Total pipeline time
            total_duration = time.time() - pipeline_start
            self.log_performance("Total", total_duration, f"Full pipeline completed")
            
            print(f"⏱️ Pipeline completed in {total_duration:.3f} seconds")
            print(f"   STT: {stt_duration:.3f}s | LLM: {llm_duration:.3f}s | TTS: {tts_duration:.3f}s | Play: {playback_duration:.3f}s")
            
        except Exception as e:
            print(f"❌ Error in voice pipeline: {e}")
            self.log_performance("Error", 0, str(e))
    
    def change_voice(self):
        """Cycle through available voices."""
        self.current_voice_index = (self.current_voice_index + 1) % len(self.available_voices)
        new_voice = self.available_voices[self.current_voice_index]
        self.tts.set_voice_parameters(voice=new_voice)
        print(f"🎭 Changed voice to: {new_voice}")
    
    def replay_last_response(self):
        """Replay the last AI response."""
        if self.last_response_audio:
            print("🔄 Replaying last response...")
            self.player.play_audio_data(self.last_response_audio)
        else:
            print("⚠️ No previous response to replay")
    
    def print_conversation_summary(self):
        """Print a summary of the conversation."""
        print("\\n📋 Conversation Summary")
        print("=" * 60)
        print(f"Session duration: {time.time() - self.session_start_time:.1f} seconds")
        print(f"Exchanges: {len(self.conversation_history)}")
        print(f"Current voice: {self.tts.voice}")
        
        if self.conversation_history:
            print("\\nLast few exchanges:")
            for i, exchange in enumerate(self.conversation_history[-3:], 1):
                print(f"  {i}. You: {exchange['human'][:60]}...")
                print(f"     AI: {exchange['assistant'][:60]}...")
        
        # Performance summary
        if self.performance_log:
            stt_times = [log['duration'] for log in self.performance_log if log['stage'] == 'STT']
            llm_times = [log['duration'] for log in self.performance_log if log['stage'] == 'LLM']
            tts_times = [log['duration'] for log in self.performance_log if log['stage'] == 'TTS']
            
            if stt_times:
                print(f"\\nAverage STT time: {sum(stt_times)/len(stt_times):.3f}s")
            if llm_times:
                print(f"Average LLM time: {sum(llm_times)/len(llm_times):.3f}s")
            if tts_times:
                print(f"Average TTS time: {sum(tts_times)/len(tts_times):.3f}s")
        
        # Enhanced LLM metrics
        try:
            llm_metrics = self.llm.get_performance_metrics()
            print(f"\\n🧠 Real Ollama/Mistral Performance:")
            print(f"   Total calls: {llm_metrics['total_calls']}")
            print(f"   Total time: {llm_metrics['total_time']:.3f}s")
            print(f"   Average time per call: {llm_metrics['average_time_per_call']:.3f}s")
            print(f"   Model: {llm_metrics['model_name']}")
            print(f"   Ollama available: {llm_metrics['ollama_available']}")
            print(f"   Client connected: {llm_metrics['client_connected']}")
        except Exception as e:
            print(f"   Could not retrieve Ollama LLM metrics: {e}")
    
    def start_recording_thread(self):
        """Start recording in a separate thread."""
        if self.recording_thread and self.recording_thread.is_alive():
            return False
        
        self.recording_thread = threading.Thread(target=self.recorder.start_recording)
        self.recording_thread.start()
        return True
    
    def stop_recording_and_process(self):
        """Stop recording and process the audio."""
        if self.recorder.is_recording:
            self.recorder.stop_recording()
            
            if self.recording_thread:
                self.recording_thread.join()
            
            audio_data = self.recorder.get_audio_data()
            if audio_data:
                # Process in async context
                asyncio.create_task(self.process_voice_input(audio_data))
            else:
                print("⚠️ No audio data recorded")
    
    async def run_keyboard_mode(self):
        """Run with keyboard controls."""
        print("\\n🎮 Keyboard Control Mode")
        print("Controls:")
        print("  SPACE - Start/Stop recording")
        print("  'v' - Change voice")
        print("  'r' - Replay last response")
        print("  'q' - Quit")
        print("\\nPress SPACE to start your first recording...")
        
        recording_active = False
        
        while self.is_running:
            try:
                if keyboard.is_pressed('space'):
                    if not recording_active:
                        if self.start_recording_thread():
                            recording_active = True
                        await asyncio.sleep(0.5)  # Prevent double-trigger
                    else:
                        self.stop_recording_and_process()
                        recording_active = False
                        await asyncio.sleep(0.5)  # Prevent double-trigger
                
                elif keyboard.is_pressed('v'):
                    self.change_voice()
                    await asyncio.sleep(0.5)  # Prevent double-trigger
                
                elif keyboard.is_pressed('r'):
                    self.replay_last_response()
                    await asyncio.sleep(0.5)  # Prevent double-trigger
                
                elif keyboard.is_pressed('q'):
                    print("\\n👋 Exiting...")
                    self.is_running = False
                    break
                
                await asyncio.sleep(0.1)  # Small delay to prevent high CPU usage
                
            except Exception as e:
                print(f"❌ Keyboard error: {e}")
                break
    
    async def run_input_mode(self):
        """Run with input() prompts."""
        print("\\n⌨️ Input Control Mode")
        print("Commands:")
        print("  'record' - Start recording")
        print("  'voice' - Change voice")
        print("  'replay' - Replay last response")
        print("  'summary' - Show conversation summary")
        print("  'quit' - Exit")
        
        while self.is_running:
            try:
                command = input("\\n> ").strip().lower()
                
                if command == 'record':
                    print("🎤 Starting recording... Press Enter to stop.")
                    if self.start_recording_thread():
                        input()  # Wait for user to press Enter
                        self.stop_recording_and_process()
                        await asyncio.sleep(0.1)  # Allow processing to start
                
                elif command == 'voice':
                    self.change_voice()
                
                elif command == 'replay':
                    self.replay_last_response()
                
                elif command == 'summary':
                    self.print_conversation_summary()
                
                elif command in ['quit', 'exit', 'q']:
                    print("\\n👋 Exiting...")
                    self.is_running = False
                    break
                
                else:
                    print("❓ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\\n👋 Exiting...")
                self.is_running = False
                break
            except Exception as e:
                print(f"❌ Input error: {e}")
    
    async def run(self):
        """Run the voice conversation test."""
        print("🎙️ Full Voice Conversation Test")
        print("=" * 60)
        print("This test demonstrates the complete voice pipeline:")
        print("  1. 🎤 Speech Input (Microphone)")
        print("  2. 📝 Speech-to-Text (Faster-Whisper)")
        print("  3. 🧠 Language Model (Mistral via Ollama)")
        print("  4. 🎵 Text-to-Speech (Kokoro)")
        print("  5. 🔊 Audio Output (Speakers)")
        print("=" * 60)
        
        # System check
        print("\\n🔍 System Check:")
        print(f"  Audio Recording: {'✅ Available' if AUDIO_AVAILABLE else '⚠️ Simulated'}")
        print(f"  Keyboard Control: {'✅ Available' if KEYBOARD_AVAILABLE else '⚠️ Input Mode'}")
        print(f"  Current Voice: {self.tts.voice}")
        print(f"  Available Voices: {', '.join(self.available_voices)}")
        
        # Test Ollama connection
        print("\\n🔗 Testing Real AI Services...")
        try:
            connection_ok = await self.llm.test_connection()
            print(f"  Ollama/Mistral: {'✅ Connected' if connection_ok else '❌ Failed'}")
        except Exception as e:
            print(f"  Ollama/Mistral: ❌ Error - {e}")
        
        # Test Faster-Whisper
        try:
            stt_info = self.stt.get_model_info()
            perf_info = self.stt.get_performance_info()
            print(f"  Faster-Whisper STT: {'✅ Available' if stt_info['available'] else '⚠️ Mock Mode'}")
            print(f"  Whisper Model: {stt_info['model_name']} on {stt_info['device']}")
            print(f"  Compute Type: {stt_info['compute_type']}")
            print(f"  Expected Speedup: {perf_info['expected_speedup']}")
        except Exception as e:
            print(f"  Faster-Whisper STT: ⚠️ Error - {e}")
        
        # Run appropriate control mode
        if KEYBOARD_AVAILABLE:
            await self.run_keyboard_mode()
        else:
            await self.run_input_mode()
        
        # Cleanup and summary
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        
        self.print_conversation_summary()
        
        # Save final log
        if self.performance_log:
            log_file = self.get_log_file_path()
            print(f"\\n📊 Performance log saved to: {log_file}")

async def main():
    """Main function to run the voice conversation test."""
    test = VoiceConversationTest()
    
    try:
        await test.run()
    except KeyboardInterrupt:
        print("\\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Full Voice Conversation Test...")
    asyncio.run(main())
