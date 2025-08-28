#!/usr/bin/env python3
"""
Real-Time Voice Bot Test

Interactive voice test for the voicebot orchestration platform.
Allows direct conversation with the AI bot through headset/microphone.
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

# Audio processing
try:
    import pyaudio
    import speech_recognition as sr
    AUDIO_AVAILABLE = True
except ImportError:
    print("⚠️  Audio libraries not available. Install with:")
    print("   pip install pyaudio SpeechRecognition")
    AUDIO_AVAILABLE = False

# TTS libraries
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("⚠️  TTS library not available. Install with:")
    print("   pip install pyttsx3")
    TTS_AVAILABLE = False

# Import our voicebot components
import sys
sys.path.append('.')
from voicebot_orchestrator.session_manager import SessionManager
from voicebot_orchestrator.llm import create_llm

class VoiceBotTest:
    """Real-time voice bot test interface."""
    
    def __init__(self):
        """Initialize voice bot test."""
        self.session_manager = SessionManager()
        self.llm = create_llm()
        self.session_id = f"voice-test-{int(time.time())}"
        self.is_listening = False
        self.conversation_history = []
        
        # Audio setup
        if AUDIO_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            print("🎙️  Adjusting for ambient noise... (speak now)")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("✅ Microphone calibrated!")
        
        # TTS setup
        if TTS_AVAILABLE:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 180)  # Speaking rate
            self.tts_engine.setProperty('volume', 0.8)  # Volume level
            
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to use a more natural voice if available
                for voice in voices:
                    if 'zira' in voice.name.lower() or 'female' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
    
    async def start_session(self) -> Dict[str, Any]:
        """Start a new voice test session."""
        session_data = await self.session_manager.create_session(
            session_id=self.session_id,
            metadata={
                "test_type": "voice_interaction",
                "started_at": datetime.now().isoformat(),
                "domain": "banking"
            }
        )
        
        return {
            "session_id": self.session_id,
            "status": "started",
            "message": "Voice bot test session started"
        }
    
    def listen_for_speech(self) -> Optional[str]:
        """Listen for speech input from microphone."""
        if not AUDIO_AVAILABLE:
            return None
            
        try:
            print("🎙️  Listening... (speak now)")
            
            # Listen for audio with timeout
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=5)
            
            print("🔄 Processing speech...")
            start_time = time.time()
            
            # Use Google's speech recognition (free tier)
            text = self.recognizer.recognize_google(audio)
            
            processing_time = time.time() - start_time
            print(f"✅ Speech recognized in {processing_time:.2f}s: '{text}'")
            
            return text
            
        except sr.WaitTimeoutError:
            print("⏰ Listening timeout - no speech detected")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand speech")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return None
    
    async def process_with_llm(self, user_input: str) -> str:
        """Process user input with LLM."""
        try:
            print("🧠 Generating AI response...")
            start_time = time.time()
            
            # Create context for banking voicebot
            context = f"""You are a helpful banking voice assistant. 
            Keep responses conversational, concise (1-2 sentences), and friendly.
            
            Previous conversation:
            {self.get_conversation_context()}
            
            Customer: {user_input}
            Assistant:"""
            
            response = await self.llm.generate_response(
                context,
                max_tokens=100,
                session_id=self.session_id
            )
            
            processing_time = time.time() - start_time
            print(f"✅ LLM response generated in {processing_time:.2f}s")
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ LLM processing error: {e}")
            return "I'm sorry, I'm having trouble processing that right now. Could you try again?"
    
    def speak_response(self, text: str) -> None:
        """Speak the response using TTS."""
        if not TTS_AVAILABLE:
            print(f"🤖 Bot would say: {text}")
            return
            
        try:
            print(f"🔊 Speaking: {text}")
            start_time = time.time()
            
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
            speaking_time = time.time() - start_time
            print(f"✅ Speech completed in {speaking_time:.2f}s")
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            print(f"🤖 Bot says: {text}")
    
    def get_conversation_context(self) -> str:
        """Get recent conversation for context."""
        if not self.conversation_history:
            return "No previous conversation."
        
        # Return last 3 exchanges
        recent = self.conversation_history[-6:]  # 3 exchanges = 6 messages
        context = []
        for i in range(0, len(recent), 2):
            if i + 1 < len(recent):
                context.append(f"Customer: {recent[i]}")
                context.append(f"Assistant: {recent[i+1]}")
        
        return "\n".join(context)
    
    async def conversation_loop(self):
        """Main conversation loop."""
        print("\n" + "="*60)
        print("🤖 VOICE BOT TEST - REAL-TIME CONVERSATION")
        print("="*60)
        print("💡 Instructions:")
        print("   - Speak clearly into your microphone")
        print("   - Wait for the bot to respond")
        print("   - Say 'goodbye' or 'exit' to end")
        print("   - Press Ctrl+C to force quit")
        print("="*60)
        
        # Start session
        session_info = await self.start_session()
        print(f"📞 Session started: {session_info['session_id']}")
        
        # Initial greeting
        greeting = "Hello! I'm your voice banking assistant. How can I help you today?"
        self.speak_response(greeting)
        self.conversation_history.append(greeting)
        
        conversation_count = 0
        total_start_time = time.time()
        
        while True:
            try:
                # Listen for user input
                print(f"\n--- Turn {conversation_count + 1} ---")
                user_input = self.listen_for_speech()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if any(word in user_input.lower() for word in ['goodbye', 'bye', 'exit', 'quit', 'stop']):
                    farewell = "Thank you for testing the voice bot. Have a great day!"
                    self.speak_response(farewell)
                    break
                
                # Record user input
                self.conversation_history.append(user_input)
                
                # Process with LLM
                bot_response = await self.process_with_llm(user_input)
                self.conversation_history.append(bot_response)
                
                # Speak response
                self.speak_response(bot_response)
                
                # Update session
                await self.session_manager.add_to_history(
                    self.session_id, user_input, bot_response
                )
                
                conversation_count += 1
                
                # Show performance stats every 3 turns
                if conversation_count % 3 == 0:
                    total_time = time.time() - total_start_time
                    avg_time = total_time / conversation_count
                    print(f"\n📊 Performance: {conversation_count} turns, avg {avg_time:.2f}s per turn")
                
            except KeyboardInterrupt:
                print("\n\n👋 Voice bot test ended by user")
                break
            except Exception as e:
                print(f"\n❌ Error in conversation loop: {e}")
                continue
        
        # End session
        await self.session_manager.end_session(self.session_id)
        
        # Final stats
        total_time = time.time() - total_start_time
        print(f"\n📊 FINAL STATS:")
        print(f"   Total turns: {conversation_count}")
        print(f"   Total time: {total_time:.2f}s")
        if conversation_count > 0:
            print(f"   Average per turn: {total_time/conversation_count:.2f}s")
        print(f"   Session ID: {self.session_id}")

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    if not AUDIO_AVAILABLE:
        missing.extend(["pyaudio", "SpeechRecognition"])
    
    if not TTS_AVAILABLE:
        missing.append("pyttsx3")
    
    return missing

async def main():
    """Main entry point."""
    print("🤖 Voice Bot Real-Time Test")
    print("="*40)
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print("⚠️  Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall with:")
        print("   pip install pyaudio SpeechRecognition pyttsx3")
        print("\nNote: pyaudio might require additional setup on some systems")
        
        # Ask if user wants to continue in text mode
        try:
            response = input("\nContinue in text-only mode? (y/n): ").lower()
            if response != 'y':
                return
        except KeyboardInterrupt:
            return
    
    # Create and run voice bot test
    voice_bot = VoiceBotTest()
    await voice_bot.conversation_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
