#!/usr/bin/env python3
"""
Production Voice Bot - Simulated Conversation Demo

This demo simulates a realistic production conversation to show the exact rhythm
and timing of a conversation using Kokoro TTS + Mistral LLM + Whisper STT.

Perfect for testing conversation flow without needing microphone input.
"""

import asyncio
import time
import os
from datetime import datetime
from typing import List, Dict, Any
import sys

# Import our production voicebot components  
sys.path.append('.')
from voicebot_orchestrator.session_manager import SessionManager
from voicebot_orchestrator.stt import WhisperSTT
from voicebot_orchestrator.llm import MistralLLM
from voicebot_orchestrator.tts import KokoroTTS
from voicebot_orchestrator.datetime_utils import DateTimeFormatter

class ProductionConversationDemo:
    """Simulated production conversation with realistic timing."""
    
    def __init__(self):
        """Initialize production demo."""
        print("🏭 Initializing Production Voice Bot Demo...")
        
        # Core production components
        self.session_manager = SessionManager()
        self.stt = WhisperSTT()
        self.llm = MistralLLM(model_path="mistral-7b", max_tokens=150, temperature=0.7)
        self.tts = KokoroTTS(voice="default", language="en", speed=1.0)
        
        self.session_id = f"demo-{int(time.time())}"
        self.performance_metrics = {
            "stt_times": [],
            "llm_times": [], 
            "tts_times": [],
            "total_times": []
        }
        
        print("✅ Production components initialized!")
    
    async def simulate_customer_speech(self, text: str) -> float:
        """Simulate processing customer speech with Whisper STT."""
        print(f"🎙️  Customer says: '{text}'")
        
        # Simulate realistic STT processing time based on text length
        words = len(text.split())
        stt_time = max(0.5, words * 0.1)  # Realistic Whisper processing time
        
        print(f"🔄 Processing with Whisper STT...")
        await asyncio.sleep(stt_time)
        
        self.performance_metrics["stt_times"].append(stt_time)
        print(f"✅ Whisper STT ({stt_time:.2f}s): Text transcribed")
        
        return stt_time
    
    async def generate_response(self, user_input: str) -> tuple[str, float]:
        """Generate response with Mistral LLM."""
        print(f"🧠 Processing with Mistral LLM...")
        start_time = time.time()
        
        # Use production LLM
        response = await self.llm.generate_response(user_input)
        
        llm_time = time.time() - start_time
        self.performance_metrics["llm_times"].append(llm_time)
        
        print(f"✅ Mistral LLM ({llm_time:.2f}s): Generated response")
        return response, llm_time
    
    async def synthesize_speech(self, text: str) -> float:
        """Synthesize speech with Kokoro TTS."""
        print(f"🔊 Synthesizing with Kokoro TTS...")
        start_time = time.time()
        
        # Use production TTS
        audio_data = await self.tts.synthesize_speech(text, output_format="wav")
        
        tts_time = time.time() - start_time
        self.performance_metrics["tts_times"].append(tts_time)
        
        # Save audio file to demos/audio_output directory
        timestamp = time.time()
        audio_filename = DateTimeFormatter.get_audio_filename("demo_kokoro", timestamp)
        
        # Create audio_output directory if it doesn't exist
        audio_dir = os.path.join(os.path.dirname(__file__), "audio_output")
        os.makedirs(audio_dir, exist_ok=True)
        audio_file = os.path.join(audio_dir, audio_filename)
        
        with open(audio_file, "wb") as f:
            f.write(audio_data)
        
        print(f"✅ Kokoro TTS ({tts_time:.2f}s): '{text}'")
        print(f"🎵 Audio saved: {audio_file}")
        
        # Simulate audio playback time (realistic speaking duration)
        words = len(text.split())
        playback_time = max(1.0, words * 0.4)  # ~150 words per minute
        print(f"🔊 Playing audio ({playback_time:.1f}s duration)...")
        await asyncio.sleep(playback_time)  # Simulate audio playback
        
        return tts_time
    
    async def conversation_turn(self, customer_input: str, turn_number: int) -> float:
        """Execute one complete conversation turn."""
        print(f"\n{'='*25} TURN {turn_number} {'='*25}")
        turn_start = time.time()
        
        # 1. Customer speaks (STT processing)
        stt_time = await self.simulate_customer_speech(customer_input)
        
        # 2. LLM generates response
        bot_response, llm_time = await self.generate_response(customer_input)
        
        # 3. TTS synthesizes and plays response
        tts_time = await self.synthesize_speech(bot_response)
        
        # Calculate total turn time
        turn_total = time.time() - turn_start
        self.performance_metrics["total_times"].append(turn_total)
        
        print(f"\n📊 Turn {turn_number} Performance:")
        print(f"   STT: {stt_time:.2f}s | LLM: {llm_time:.2f}s | TTS: {tts_time:.2f}s")
        print(f"   Total: {turn_total:.2f}s")
        
        return turn_total
    
    def show_conversation_analysis(self, total_turns: int):
        """Show detailed conversation analysis."""
        print(f"\n" + "="*60)
        print("📊 PRODUCTION CONVERSATION RHYTHM ANALYSIS")
        print("="*60)
        print(f"Session ID: {self.session_id}")
        print(f"Total conversation turns: {total_turns}")
        
        if self.performance_metrics["total_times"]:
            total_times = self.performance_metrics["total_times"]
            stt_times = self.performance_metrics["stt_times"]
            llm_times = self.performance_metrics["llm_times"]
            tts_times = self.performance_metrics["tts_times"]
            
            print(f"\n🕐 Timing Breakdown:")
            print(f"   Average turn duration: {sum(total_times)/len(total_times):.2f}s")
            print(f"   Fastest turn: {min(total_times):.2f}s")
            print(f"   Slowest turn: {max(total_times):.2f}s")
            
            print(f"\n⚡ Component Performance:")
            print(f"   Whisper STT average: {sum(stt_times)/len(stt_times):.2f}s")
            print(f"   Mistral LLM average: {sum(llm_times)/len(llm_times):.2f}s")
            print(f"   Kokoro TTS average: {sum(tts_times)/len(tts_times):.2f}s")
            
            # Calculate conversation rhythm
            total_conversation_time = sum(total_times)
            processing_time = sum(stt_times) + sum(llm_times) + sum(tts_times)
            audio_time = total_conversation_time - processing_time
            
            print(f"\n🎵 Conversation Rhythm:")
            print(f"   Total conversation: {total_conversation_time:.1f}s")
            print(f"   Processing time: {processing_time:.1f}s ({processing_time/total_conversation_time*100:.1f}%)")
            print(f"   Audio/speaking time: {audio_time:.1f}s ({audio_time/total_conversation_time*100:.1f}%)")
            
            # Production readiness assessment
            avg_turn = sum(total_times)/len(total_times)
            if avg_turn < 5:
                assessment = "🚀 EXCELLENT - Very responsive conversation"
            elif avg_turn < 8:
                assessment = "✅ GOOD - Natural conversation rhythm"
            elif avg_turn < 12:
                assessment = "⚠️  ACCEPTABLE - Slightly slow but usable"
            else:
                assessment = "❌ NEEDS OPTIMIZATION - Too slow for production"
            
            print(f"\n🎯 Production Assessment: {assessment}")
    
    async def run_demo_conversation(self):
        """Run a complete demo conversation."""
        print("\n" + "="*70)
        print("🎙️ PRODUCTION VOICE BOT - CONVERSATION RHYTHM DEMO")
        print("="*70)
        print("🏭 Production Stack:")
        print("   • Whisper STT - Enterprise speech recognition")
        print("   • Mistral LLM - Intelligent banking assistant")
        print("   • Kokoro TTS - Premium voice synthesis")
        print("\n🎬 Simulating realistic banking conversation...")
        print("="*70)
        
        # Start session
        await self.session_manager.create_session(
            session_id=self.session_id,
            metadata={
                "demo_type": "production_rhythm",
                "components": "whisper+mistral+kokoro",
                "started_at": datetime.now().isoformat()
            }
        )
        
        print(f"📞 Session started: {self.session_id}")
        
        # Initial greeting from bot
        print(f"\n🤖 Bot initiating conversation...")
        await self.synthesize_speech("Hello! I'm your banking assistant. How can I help you today?")
        
        # Demo conversation scenarios
        conversation_scenarios = [
            "Hi, I'd like to check my account balance please",
            "Thank you. Can I also see my recent transactions?", 
            "Great. Do I have any pending transfers?",
            "Perfect. Can you help me transfer $200 to my savings account?",
            "Excellent. Thank you for your help today!"
        ]
        
        # Execute conversation turns
        for i, customer_input in enumerate(conversation_scenarios, 1):
            await self.conversation_turn(customer_input, i)
            
            # Add small pause between turns (realistic conversation pacing)
            if i < len(conversation_scenarios):
                print("⏸️  Brief pause...")
                await asyncio.sleep(0.5)
        
        # Final bot response
        print(f"\n🤖 Bot concluding conversation...")
        await self.synthesize_speech("Thank you for using our banking services. Have a wonderful day!")
        
        # End session
        await self.session_manager.end_session(self.session_id)
        
        # Show analysis
        self.show_conversation_analysis(len(conversation_scenarios))
        
        print(f"\n🎉 Demo complete! Audio files saved with prefix 'demo_kokoro_'")

async def main():
    """Main entry point for production demo."""
    try:
        demo = ProductionConversationDemo()
        await demo.run_demo_conversation()
    except KeyboardInterrupt:
        print("\n👋 Demo ended by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
