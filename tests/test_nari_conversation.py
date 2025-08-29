"""
Quick test of switchable TTS engine with Nari Dia
Simulates a conversation without voice input to test the pipeline
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from voicebot_orchestrator.enhanced_tts_manager import EnhancedTTSManager, TTSEngine
from real_llm import RealOllamaLLM

async def test_conversation_with_nari():
    """Test full conversation pipeline with Nari Dia TTS"""
    print("🎭 TESTING CONVERSATION PIPELINE WITH NARI DIA")
    print("=" * 60)
    
    # Initialize components
    print("⏳ Initializing components...")
    
    # Initialize LLM
    print("1️⃣ Initializing Ollama/Mistral LLM...")
    llm = RealOllamaLLM()
    print("✅ LLM ready")
    
    # Initialize Enhanced TTS Manager
    print("2️⃣ Initializing Enhanced TTS Manager...")
    tts_manager = EnhancedTTSManager()
    await tts_manager.initialize_engines(load_kokoro=True, load_nari=True)
    
    # Switch to Nari Dia
    print("3️⃣ Switching to Nari Dia TTS...")
    tts_manager.set_engine(TTSEngine.NARI_DIA)
    print("✅ All components ready")
    
    print("\n🎬 Starting conversation simulation...")
    
    # Simulate user inputs
    user_inputs = [
        "Hello, I'd like to check my account balance",
        "What types of loans do you offer?",
        "Thank you for your help"
    ]
    
    total_conversation_time = 0
    
    for i, user_text in enumerate(user_inputs, 1):
        print(f"\n💬 CONVERSATION TURN {i}/3")
        print("=" * 40)
        print(f"👤 User: \"{user_text}\"")
        
        # LLM Processing
        print("🧠 Processing with Mistral LLM...")
        import time
        
        llm_start = time.time()
        ai_response = await llm.generate_response(user_text)
        llm_time = time.time() - llm_start
        
        print(f"🤖 AI Response: \"{ai_response}\"")
        print(f"⏱️  LLM processing: {llm_time:.2f}s")
        
        # TTS Processing with Nari Dia
        print("🎭 Generating speech with Nari Dia...")
        print("⏳ This will take 30+ seconds for high-quality voice...")
        
        tts_start = time.time()
        audio_bytes, tts_time, used_engine = await tts_manager.generate_speech(
            ai_response,
            save_path=f"nari_conversation_turn_{i}.wav"
        )
        
        print(f"✅ Generated {len(audio_bytes)} bytes in {tts_time:.2f}s using {used_engine}")
        
        # Turn summary
        turn_total = llm_time + tts_time
        total_conversation_time += turn_total
        
        print(f"📊 Turn {i} Summary:")
        print(f"   LLM: {llm_time:.2f}s")
        print(f"   TTS: {tts_time:.2f}s")
        print(f"   Total: {turn_total:.2f}s")
        
        if tts_time < 10:
            print("🚀 Surprisingly fast for Nari Dia!")
        elif tts_time < 30:
            print("⚡ Reasonable Nari Dia speed")
        else:
            print("⏳ Typical Nari Dia generation time")
    
    # Final summary
    print(f"\n🏁 CONVERSATION COMPLETE")
    print("=" * 50)
    print(f"📊 Total conversation time: {total_conversation_time:.2f}s")
    print(f"⏱️  Average per turn: {total_conversation_time/len(user_inputs):.2f}s")
    
    print(f"\n🎭 Nari Dia Performance Assessment:")
    avg_turn_time = total_conversation_time / len(user_inputs)
    if avg_turn_time < 15:
        print("✅ Acceptable for high-quality non-real-time scenarios")
    elif avg_turn_time < 60:
        print("⚠️ Slow but usable for premium quality responses")
    else:
        print("❌ Too slow for practical conversation use")
    
    print(f"\n💡 Recommendation:")
    print("• Use Nari Dia for: Pre-recorded messages, announcements, maximum quality")
    print("• Use Kokoro for: Real-time conversation, interactive responses")
    
    # Show generated files
    print(f"\n🔊 Generated audio files:")
    for i in range(1, len(user_inputs) + 1):
        filename = f"nari_conversation_turn_{i}.wav"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / 1024
            print(f"   • {filename} ({file_size:.1f} KB)")
    
    # Cleanup
    tts_manager.cleanup()
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_conversation_with_nari())
