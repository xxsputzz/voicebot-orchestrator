#!/usr/bin/env python3
"""
Production Voice Bot Test with Kokoro TTS

Real-time voice conversation test using production-grade components:
- Whisper STT for speech recognition
- Mistral LLM for intelligent responses  
- Kokoro TTS for natural speech synthesis

This shows the actual rhythm and speed of production voicebot conversations.
"""

import asyncio
import json
import time
import wave
import threading
from datetime import datetime
from typing import Optional, Dict, Any
import tempfile
import os
import sys

# Import our production voicebot components
sys.path.append('.')
from voicebot_orchestrator.session_manager import SessionManager
from voicebot_orchestrator.stt import WhisperSTT
from voicebot_orchestrator.llm import MistralLLM
from voicebot_orchestrator.tts import KokoroTTS
from voicebot_orchestrator.datetime_utils import DateTimeFormatter

# Audio processing for real-time input
try:
    import pyaudio
    import speech_recognition as sr
    AUDIO_AVAILABLE = True
    print("✅ Audio libraries available")
except ImportError:
    print("⚠️  Installing audio libraries...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "SpeechRecognition", "pyaudio"])
        import speech_recognition as sr
        AUDIO_AVAILABLE = True
        print("✅ Audio libraries installed successfully")
    except:
        AUDIO_AVAILABLE = False
        print("❌ Audio libraries not available. Install manually: pip install SpeechRecognition pyaudio")

class ProductionVoiceBot:
    """Production-grade voice bot with Kokoro TTS and real components."""
    
    def __init__(self):
        """Initialize production voice bot."""
        print("🚀 Initializing Production Voice Bot...")
        
        # Core components
        self.session_manager = SessionManager()
        self.stt = WhisperSTT()
        self.llm = MistralLLM(model_path="mistral-7b", max_tokens=150, temperature=0.7)
        self.tts = KokoroTTS(voice="default", language="en", speed=1.0)
        
        self.session_id = f"prod-voice-{int(time.time())}"
        self.conversation_history = []
        self.performance_metrics = {
            "stt_times": [],
            "llm_times": [], 
            "tts_times": [],
            "total_times": []
        }
        
        # Audio setup
        if AUDIO_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            print("🎙️  Calibrating microphone for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("✅ Microphone calibrated!")
        
        print("✅ Production Voice Bot initialized!")
    
    async def start_session(self) -> Dict[str, Any]:
        """Start a new production voice test session."""
        session_data = await self.session_manager.create_session(
            session_id=self.session_id,
            metadata={
                "test_type": "production_voice_test",
                "components": "whisper+mistral+kokoro",
                "started_at": datetime.now().isoformat(),
                "domain": "banking"
            }
        )
        
        return {
            "session_id": self.session_id,
            "status": "started",
            "components": "Production: Whisper STT + Mistral LLM + Kokoro TTS"
        }
    
    async def listen_with_whisper(self) -> Optional[str]:
        """Listen for speech and process with Whisper STT."""
        if not AUDIO_AVAILABLE:
            # Fallback to text input
            try:
                return input("🎙️  [No microphone] Type your message: ").strip()
            except KeyboardInterrupt:
                return None
        
        try:
            print("🎙️  Listening... (speak clearly)")
            
            # Record audio
            with self.microphone as source:
                print("🔴 Recording...")
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=8)
            
            print("🔄 Processing with Whisper STT...")
            start_time = time.time()
            
            # Save audio to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio.get_wav_data())
                temp_audio_path = temp_audio.name
            
            try:
                # Process with our production Whisper STT
                transcript = await self.stt.transcribe_file(temp_audio_path)
                text = transcript.strip() if transcript else ""
                
                stt_time = time.time() - start_time
                self.performance_metrics["stt_times"].append(stt_time)
                
                print(f"✅ Whisper STT ({stt_time:.2f}s): '{text}'")
                return text if text else None
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            
        except sr.WaitTimeoutError:
            print("⏰ Listening timeout - no speech detected")
            return None
        except sr.UnknownValueError:
            print("❌ Audio quality too low for recognition")
            return None
        except Exception as e:
            print(f"❌ STT error: {e}")
            return None
    
    async def process_with_mistral(self, user_input: str) -> str:
        """Process user input with Mistral LLM."""
        try:
            print("🧠 Processing with Mistral LLM...")
            start_time = time.time()
            
            # Build context for banking assistant
            context = self._build_banking_context(user_input)
            
            # Use our production LLM
            response = await self.llm.generate_response(
                user_input,
                conversation_history=None  # Could pass self.conversation_history here
            )
            
            llm_time = time.time() - start_time
            self.performance_metrics["llm_times"].append(llm_time)
            
            print(f"✅ Mistral LLM ({llm_time:.2f}s): Generated response")
            return response.strip()
            
        except Exception as e:
            print(f"❌ LLM processing error: {e}")
            return "I apologize, I'm experiencing some technical difficulties. Could you please try again?"
    
    async def speak_with_kokoro(self, text: str) -> None:
        """Speak response using Kokoro TTS."""
        try:
            print(f"🔊 Synthesizing with Kokoro TTS: '{text}'")
            start_time = time.time()
            
            # Use our production Kokoro TTS
            audio_data = await self.tts.synthesize_speech(text, output_format="wav")
            
            tts_time = time.time() - start_time
            self.performance_metrics["tts_times"].append(tts_time)
            
            print(f"✅ Kokoro TTS ({tts_time:.2f}s): Audio synthesized")
            
            # Save and play audio to demos/audio_output directory
            timestamp = time.time()
            audio_filename = DateTimeFormatter.get_audio_filename("kokoro_output", timestamp)
            
            # Create audio_output directory if it doesn't exist
            audio_dir = os.path.join(os.path.dirname(__file__), "audio_output")
            os.makedirs(audio_dir, exist_ok=True)
            audio_file = os.path.join(audio_dir, audio_filename)
            
            with open(audio_file, "wb") as f:
                f.write(audio_data)
            
            print(f"🎵 Audio saved to: {audio_file}")
            
            # Try to play audio (platform-dependent)
            await self._play_audio(audio_file)
            
        except Exception as e:
            print(f"❌ Kokoro TTS error: {e}")
            print(f"🤖 [TTS Failed] Bot would say: {text}")
    
    async def _play_audio(self, audio_file: str) -> None:
        """Play audio file (cross-platform)."""
        try:
            import os
            import sys
            
            if sys.platform.startswith('win'):
                # Windows
                os.system(f'start /min "" "{audio_file}"')
            elif sys.platform.startswith('darwin'):
                # macOS
                os.system(f'afplay "{audio_file}"')
            elif sys.platform.startswith('linux'):
                # Linux
                os.system(f'aplay "{audio_file}" 2>/dev/null || paplay "{audio_file}" 2>/dev/null')
            
            print("🔊 Playing audio...")
            
        except Exception as e:
            print(f"⚠️  Audio playback error: {e}")
            print(f"🎵 Audio file saved: {audio_file}")
    
    def _build_banking_context(self, user_input: str) -> str:
        """Build context for banking assistant."""
        recent_history = self._get_recent_conversation()
        
        context = f"""You are an intelligent banking voice assistant. You provide helpful, accurate, and professional responses about banking services.

Keep responses:
- Conversational and natural (1-2 sentences)
- Professional but friendly
- Relevant to banking/financial services
- Clear and easy to understand when spoken

Recent conversation:
{recent_history}

Customer: {user_input}
Assistant:"""
        
        return context
    
    def _get_recent_conversation(self) -> str:
        """Get recent conversation for context."""
        if not self.conversation_history:
            return "No previous conversation."
        
        # Last 3 exchanges
        recent = self.conversation_history[-6:]
        context = []
        for i in range(0, len(recent), 2):
            if i + 1 < len(recent):
                context.append(f"Customer: {recent[i]}")
                context.append(f"Assistant: {recent[i+1]}")
        
        return "\n".join(context[-6:])  # Last 3 exchanges
    
    def _show_performance_stats(self, turn_number: int) -> None:
        """Show performance statistics."""
        if not self.performance_metrics["total_times"]:
            return
        
        latest_total = self.performance_metrics["total_times"][-1]
        avg_total = sum(self.performance_metrics["total_times"]) / len(self.performance_metrics["total_times"])
        
        print(f"\n📊 Performance (Turn {turn_number}):")
        print(f"   This turn: {latest_total:.2f}s total")
        print(f"   Average: {avg_total:.2f}s per turn")
        
        if self.performance_metrics["stt_times"]:
            avg_stt = sum(self.performance_metrics["stt_times"]) / len(self.performance_metrics["stt_times"])
            print(f"   STT avg: {avg_stt:.2f}s")
        
        if self.performance_metrics["llm_times"]:
            avg_llm = sum(self.performance_metrics["llm_times"]) / len(self.performance_metrics["llm_times"])
            print(f"   LLM avg: {avg_llm:.2f}s")
        
        if self.performance_metrics["tts_times"]:
            avg_tts = sum(self.performance_metrics["tts_times"]) / len(self.performance_metrics["tts_times"])
            print(f"   TTS avg: {avg_tts:.2f}s")
    
    async def production_conversation(self):
        """Main production conversation loop."""
        print("\n" + "="*70)
        print("🎙️ PRODUCTION VOICE BOT - KOKORO TTS CONVERSATION")
        print("="*70)
        print("🏭 Production Components:")
        print("   • Whisper STT - High-accuracy speech recognition")
        print("   • Mistral LLM - Intelligent response generation")
        print("   • Kokoro TTS - Natural speech synthesis")
        print("\n💡 Instructions:")
        print("   • Speak clearly into your microphone")
        print("   • Wait for complete response before speaking again")
        print("   • Say 'goodbye' or 'exit' to end conversation")
        print("   • Performance metrics shown every 3 turns")
        print("="*70)
        
        # Start session
        session_info = await self.start_session()
        print(f"📞 Production session: {session_info['session_id']}")
        
        # Initial greeting
        greeting = "Hello! I'm your production banking assistant powered by Kokoro TTS. How can I help you today?"
        await self.speak_with_kokoro(greeting)
        self.conversation_history.append(greeting)
        
        turn_number = 0
        
        while True:
            try:
                turn_start_time = time.time()
                turn_number += 1
                
                print(f"\n{'='*20} TURN {turn_number} {'='*20}")
                
                # Listen with Whisper STT
                user_input = await self.listen_with_whisper()
                
                if not user_input:
                    continue
                
                # Check for exit
                if any(word in user_input.lower() for word in ['goodbye', 'bye', 'exit', 'quit', 'stop', 'end']):
                    farewell = "Thank you for testing our production voice bot. Have a wonderful day!"
                    await self.speak_with_kokoro(farewell)
                    break
                
                self.conversation_history.append(user_input)
                
                # Process with Mistral LLM
                bot_response = await self.process_with_mistral(user_input)
                self.conversation_history.append(bot_response)
                
                # Speak with Kokoro TTS
                await self.speak_with_kokoro(bot_response)
                
                # Update session
                await self.session_manager.add_to_history(
                    self.session_id, user_input, bot_response
                )
                
                # Track performance
                turn_total_time = time.time() - turn_start_time
                self.performance_metrics["total_times"].append(turn_total_time)
                
                # Show stats every 3 turns
                if turn_number % 3 == 0:
                    self._show_performance_stats(turn_number)
                
                print(f"✅ Turn {turn_number} completed in {turn_total_time:.2f}s")
                
            except KeyboardInterrupt:
                print("\n\n👋 Production test ended by user")
                break
            except Exception as e:
                print(f"\n❌ Error in conversation: {e}")
                continue
        
        # End session and final stats
        await self.session_manager.end_session(self.session_id)
        self._show_final_stats(turn_number)
    
    def _show_final_stats(self, total_turns: int) -> None:
        """Show final conversation statistics."""
        print(f"\n" + "="*50)
        print("📊 PRODUCTION CONVERSATION ANALYSIS")
        print("="*50)
        print(f"Total turns: {total_turns}")
        print(f"Session ID: {self.session_id}")
        
        if self.performance_metrics["total_times"]:
            avg_total = sum(self.performance_metrics["total_times"]) / len(self.performance_metrics["total_times"])
            min_total = min(self.performance_metrics["total_times"])
            max_total = max(self.performance_metrics["total_times"])
            
            print(f"\nTiming Analysis:")
            print(f"  Average per turn: {avg_total:.2f}s")
            print(f"  Fastest turn: {min_total:.2f}s")
            print(f"  Slowest turn: {max_total:.2f}s")
            
            if self.performance_metrics["stt_times"]:
                avg_stt = sum(self.performance_metrics["stt_times"]) / len(self.performance_metrics["stt_times"])
                print(f"  Whisper STT avg: {avg_stt:.2f}s")
            
            if self.performance_metrics["llm_times"]:
                avg_llm = sum(self.performance_metrics["llm_times"]) / len(self.performance_metrics["llm_times"])
                print(f"  Mistral LLM avg: {avg_llm:.2f}s")
            
            if self.performance_metrics["tts_times"]:
                avg_tts = sum(self.performance_metrics["tts_times"]) / len(self.performance_metrics["tts_times"])
                print(f"  Kokoro TTS avg: {avg_tts:.2f}s")
        
        print(f"\n🎉 Production voice bot test complete!")
        print(f"Audio files saved in current directory with prefix 'kokoro_output_'")

async def main():
    """Main entry point for production voice test."""
    print("🏭 Production Voice Bot Test - Kokoro TTS")
    print("="*50)
    
    try:
        voice_bot = ProductionVoiceBot()
        await voice_bot.production_conversation()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
