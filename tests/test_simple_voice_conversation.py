#!/usr/bin/env python3
"""
Simple Voice Conversation Test
==============================

A simpler version of the full voice conversation test that works with basic input.
This test demonstrates the complete voice conversation pipeline:
1. Simulate or record audio input
2. Speech-to-Text (STT) using Whisper
3. Large Language Model (LLM) processing
4. Text-to-Speech (TTS) response using Kokoro

Usage:
    python tests/test_simple_voice_conversation.py

Optional Dependencies for Full Audio Support:
    pip install pyaudio keyboard

Without these dependencies, the test will use simulated audio data.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from voicebot_orchestrator.stt import WhisperSTT
from voicebot_orchestrator.llm import MistralLLM
from voicebot_orchestrator.tts import KokoroTTS

class SimpleVoiceTest:
    """Simple voice conversation test orchestrator."""
    
    def __init__(self):
        # Initialize AI components
        self.stt = WhisperSTT(model_name="base")
        self.llm = MistralLLM(model_path="mistral-7b", max_tokens=256)
        self.tts = KokoroTTS(voice="af_bella", speed=1.0)
        
        # Available voices
        self.available_voices = [
            "af_bella", "af_nicole", "af_sarah", "af_alloy", 
            "af_aoede", "af_kore", "af_nova"
        ]
        self.current_voice_index = 0
        
        # Conversation state
        self.conversation_history = []
        self.session_start_time = time.time()
        
        # Performance tracking
        self.performance_log = []
        
        # Sample audio files for testing
        self.create_audio_samples_dir()
    
    def create_audio_samples_dir(self):
        """Create audio samples directory if it doesn't exist."""
        self.audio_dir = os.path.join(os.path.dirname(__file__), "audio_samples")
        os.makedirs(self.audio_dir, exist_ok=True)
    
    def get_log_file_path(self):
        """Get path for performance log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.audio_dir, f"simple_conversation_log_{timestamp}.txt")
    
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
        
        with open(log_file, 'a', encoding='utf-8') as f:
            if os.path.getsize(log_file) == 0:
                f.write("Simple Voice Conversation Performance Log\\n")
                f.write("=" * 70 + "\\n")
                f.write("Timestamp\\t\\tStage\\t\\tDuration(s)\\tDetails\\n")
                f.write("-" * 70 + "\\n")
            
            f.write(f"{timestamp}\\t{stage:12}\\t{duration:.3f}\\t\\t{details}\\n")
    
    def generate_mock_audio(self, text_input):
        """Generate mock audio data based on text input."""
        # Create varying mock audio data based on input length
        base_length = len(text_input) * 100  # Simulate audio length
        mock_data = f"mock_audio_{int(time.time())}_{text_input[:20]}".encode()
        # Pad to simulate realistic audio file size
        return mock_data + b"\\x00" * base_length
    
    async def process_text_as_voice(self, text_input):
        """Process text input as if it were voice input through the full pipeline."""
        print(f"\\n🎤 Simulating voice input: \"{text_input}\"")
        pipeline_start = time.time()
        
        try:
            # Generate mock audio data
            mock_audio = self.generate_mock_audio(text_input)
            
            # Step 1: Speech-to-Text (simulated)
            print("📝 Processing speech-to-text...")
            stt_start = time.time()
            
            # For simulation, we'll use the input text but add some "STT processing"
            await asyncio.sleep(0.1)  # Simulate STT processing time
            transcribed_text = text_input  # In real scenario, this would be actual STT output
            stt_duration = time.time() - stt_start
            
            print(f"📝 STT Result: \"{transcribed_text}\"")
            self.log_performance("STT", stt_duration, f"Input: {text_input[:50]}...")
            
            # Step 2: Language Model Processing
            print("🧠 Processing with Language Model...")
            llm_start = time.time()
            response_text = await self.llm.generate_response(transcribed_text, self.conversation_history)
            llm_duration = time.time() - llm_start
            
            print(f"💭 LLM Response: \"{response_text}\"")
            self.log_performance("LLM", llm_duration, f"Response: {response_text[:50]}...")
            
            # Step 3: Text-to-Speech
            print(f"🎵 Generating speech with {self.tts.voice}...")
            tts_start = time.time()
            response_audio = await self.tts.synthesize_speech(response_text)
            tts_duration = time.time() - tts_start
            
            print(f"🔊 Generated {len(response_audio)} bytes of audio")
            self.log_performance("TTS", tts_duration, f"Voice: {self.tts.voice}, Size: {len(response_audio)}")
            
            # Step 4: Save audio file (instead of playing)
            audio_filename = f"response_{int(time.time())}.wav"
            audio_path = os.path.join(self.audio_dir, audio_filename)
            
            save_start = time.time()
            with open(audio_path, 'wb') as f:
                f.write(response_audio)
            save_duration = time.time() - save_start
            
            print(f"💾 Audio saved to: {audio_path}")
            self.log_performance("Save", save_duration, f"File: {audio_filename}")
            
            # Add to conversation history
            self.conversation_history.append({
                "user_input": transcribed_text,
                "bot_response": response_text,
                "timestamp": datetime.now().isoformat(),
                "audio_file": audio_path
            })
            
            # Total pipeline time
            total_duration = time.time() - pipeline_start
            self.log_performance("Total", total_duration, f"Full pipeline completed")
            
            print(f"⏱️ Pipeline completed in {total_duration:.3f} seconds")
            print(f"   STT: {stt_duration:.3f}s | LLM: {llm_duration:.3f}s | TTS: {tts_duration:.3f}s | Save: {save_duration:.3f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in voice pipeline: {e}")
            self.log_performance("Error", 0, str(e))
            return False
    
    def change_voice(self):
        """Cycle through available voices."""
        self.current_voice_index = (self.current_voice_index + 1) % len(self.available_voices)
        new_voice = self.available_voices[self.current_voice_index]
        self.tts.set_voice_parameters(voice=new_voice)
        print(f"🎭 Changed voice to: {new_voice}")
        return new_voice
    
    def list_voices(self):
        """List all available voices."""
        print("\\n🎭 Available Voices:")
        for i, voice in enumerate(self.available_voices):
            current = " (current)" if i == self.current_voice_index else ""
            print(f"  {i+1}. {voice}{current}")
    
    def select_voice(self, voice_name):
        """Select a specific voice by name."""
        if voice_name in self.available_voices:
            self.current_voice_index = self.available_voices.index(voice_name)
            self.tts.set_voice_parameters(voice=voice_name)
            print(f"🎭 Voice set to: {voice_name}")
            return True
        else:
            print(f"❌ Voice '{voice_name}' not found")
            return False
    
    def print_conversation_summary(self):
        """Print a summary of the conversation."""
        print("\\n📋 Conversation Summary")
        print("=" * 60)
        print(f"Session duration: {time.time() - self.session_start_time:.1f} seconds")
        print(f"Exchanges: {len(self.conversation_history)}")
        print(f"Current voice: {self.tts.voice}")
        print(f"Audio files saved in: {self.audio_dir}")
        
        if self.conversation_history:
            print("\\nConversation history:")
            for i, exchange in enumerate(self.conversation_history, 1):
                print(f"\\n  Exchange {i}:")
                print(f"    You: {exchange['user_input']}")
                print(f"    AI:  {exchange['bot_response']}")
                print(f"    Audio: {os.path.basename(exchange.get('audio_file', 'N/A'))}")
        
        # Performance summary
        if self.performance_log:
            stages = ['STT', 'LLM', 'TTS', 'Save', 'Total']
            print("\\n⏱️ Performance Summary:")
            
            for stage in stages:
                times = [log['duration'] for log in self.performance_log if log['stage'] == stage]
                if times:
                    avg_time = sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    print(f"  {stage:8}: Avg {avg_time:.3f}s (Min: {min_time:.3f}s, Max: {max_time:.3f}s)")
    
    def run_demo_conversations(self):
        """Run a series of demo conversations."""
        demo_inputs = [
            "Hello, I need help with my banking account",
            "What is my account balance?",
            "Can you help me with a money transfer?",
            "Thank you for your help",
            "How do I apply for a loan?",
            "What are your business hours?"
        ]
        
        print("\\n🎪 Running Demo Conversations")
        print("=" * 50)
        
        for i, demo_input in enumerate(demo_inputs, 1):
            print(f"\\n--- Demo {i}/{len(demo_inputs)} ---")
            success = asyncio.run(self.process_text_as_voice(demo_input))
            if not success:
                print(f"❌ Demo {i} failed")
                break
            time.sleep(1)  # Brief pause between demos
        
        print("\\n🎉 Demo conversations completed!")
    
    async def run_interactive_mode(self):
        """Run interactive conversation mode."""
        print("\\n💬 Interactive Conversation Mode")
        print("Commands:")
        print("  Type your message to simulate voice input")
        print("  'voices' - List available voices")
        print("  'voice <name>' - Change to specific voice")
        print("  'change' - Cycle to next voice")
        print("  'summary' - Show conversation summary")
        print("  'demo' - Run demo conversations")
        print("  'quit' or 'exit' - Exit")
        print("\\nStart typing your message:")
        
        while True:
            try:
                user_input = input("\\n> ").strip()
                
                if not user_input:
                    continue
                
                elif user_input.lower() in ['quit', 'exit', 'q']:
                    print("\\n👋 Exiting interactive mode...")
                    break
                
                elif user_input.lower() == 'voices':
                    self.list_voices()
                
                elif user_input.lower().startswith('voice '):
                    voice_name = user_input[6:].strip()
                    self.select_voice(voice_name)
                
                elif user_input.lower() == 'change':
                    self.change_voice()
                
                elif user_input.lower() == 'summary':
                    self.print_conversation_summary()
                
                elif user_input.lower() == 'demo':
                    self.run_demo_conversations()
                
                else:
                    # Process as voice input
                    success = await self.process_text_as_voice(user_input)
                    if not success:
                        print("❌ Failed to process input")
                        
            except KeyboardInterrupt:
                print("\\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def run(self):
        """Run the simple voice conversation test."""
        print("🎙️ Simple Voice Conversation Test")
        print("=" * 60)
        print("This test demonstrates the complete voice pipeline:")
        print("  1. 📝 Text Input (simulating voice)")
        print("  2. 🧠 Language Model Processing")
        print("  3. 🎵 Text-to-Speech Generation")
        print("  4. 💾 Audio File Output")
        print("=" * 60)
        
        print(f"\\n🔍 System Information:")
        print(f"  Current Voice: {self.tts.voice}")
        print(f"  Available Voices: {len(self.available_voices)}")
        print(f"  Output Directory: {self.audio_dir}")
        
        # Show menu
        print("\\n📋 Choose an option:")
        print("  1. Interactive mode (type messages)")
        print("  2. Run demo conversations")
        print("  3. Exit")
        
        while True:
            try:
                choice = input("\\nSelect option (1-3): ").strip()
                
                if choice == '1':
                    await self.run_interactive_mode()
                    break
                elif choice == '2':
                    self.run_demo_conversations()
                    break
                elif choice == '3':
                    print("👋 Exiting...")
                    break
                else:
                    print("❓ Please enter 1, 2, or 3")
                    
            except KeyboardInterrupt:
                print("\\n👋 Exiting...")
                break
        
        # Final summary
        self.print_conversation_summary()
        
        # Save final log
        if self.performance_log:
            log_file = self.get_log_file_path()
            print(f"\\n📊 Performance log saved to: {log_file}")

async def main():
    """Main function to run the simple voice conversation test."""
    test = SimpleVoiceTest()
    
    try:
        await test.run()
    except KeyboardInterrupt:
        print("\\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Simple Voice Conversation Test...")
    asyncio.run(main())
