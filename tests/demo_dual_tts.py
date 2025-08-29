"""
Simple demonstration of the working dual TTS system
Shows both engines working with the Enhanced TTS Manager
"""
import asyncio
import sys
import os

# Ensure we use the virtual environment path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

async def demo_dual_tts():
    """Demonstrate both TTS engines working"""
    print("🎭 DUAL TTS SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    from voicebot_orchestrator.enhanced_tts_manager import EnhancedTTSManager, TTSEngine
    
    # Initialize manager
    print("⏳ Initializing Enhanced TTS Manager...")
    manager = EnhancedTTSManager()
    
    # Load both engines
    success = await manager.initialize_engines(load_kokoro=True, load_nari=True)
    if not success:
        print("❌ Failed to initialize engines")
        return
    
    # Test phrase
    test_text = "Welcome to our banking service. How may I assist you today?"
    
    print(f"\n🎤 Testing with: \"{test_text}\"")
    
    # Test 1: Kokoro (Fast)
    print(f"\n1️⃣ KOKORO ENGINE TEST")
    print("-" * 30)
    
    try:
        manager.set_engine(TTSEngine.KOKORO)
        audio1, time1, engine1 = await manager.generate_speech(
            test_text, 
            save_path="demo_kokoro.wav"
        )
        print(f"✅ {engine1}: {len(audio1):,} bytes in {time1:.2f}s")
    except Exception as e:
        print(f"❌ Kokoro failed: {e}")
    
    # Test 2: Nari Dia (Quality)
    print(f"\n2️⃣ NARI DIA ENGINE TEST")
    print("-" * 30)
    
    try:
        manager.set_engine(TTSEngine.NARI_DIA)
        audio2, time2, engine2 = await manager.generate_speech(
            test_text,
            save_path="demo_nari_dia.wav"
        )
        print(f"✅ {engine2}: {len(audio2):,} bytes in {time2:.2f}s")
        
        # Performance comparison
        if 'time1' in locals():
            speedup = time2 / time1
            print(f"\n📊 PERFORMANCE COMPARISON:")
            print(f"   Kokoro: {time1:.2f}s")
            print(f"   Nari Dia: {time2:.2f}s")
            print(f"   Nari Dia is {speedup:.1f}x slower than Kokoro")
    except Exception as e:
        print(f"❌ Nari Dia failed: {e}")
    
    # Engine switching demo
    print(f"\n3️⃣ ENGINE SWITCHING DEMO")
    print("-" * 30)
    
    for engine in [TTSEngine.KOKORO, TTSEngine.NARI_DIA, TTSEngine.KOKORO]:
        try:
            manager.set_engine(engine)
            current = manager.get_current_engine()
            print(f"✅ Current engine: {current.value}")
        except Exception as e:
            print(f"❌ Failed to switch to {engine.value}: {e}")
    
    # Summary
    print(f"\n🎉 DEMONSTRATION COMPLETE")
    print("=" * 50)
    
    available = manager.get_available_engines()
    print(f"🎯 Available engines: {[e.value for e in available]}")
    print(f"🔧 Current engine: {manager.get_current_engine().value}")
    
    # Show generated files
    import os
    print(f"\n🔊 Generated audio files:")
    for filename in ["demo_kokoro.wav", "demo_nari_dia.wav"]:
        if os.path.exists(filename):
            size_kb = os.path.getsize(filename) / 1024
            print(f"   • {filename} ({size_kb:.1f} KB)")
    
    # Usage recommendations
    print(f"\n💡 USAGE RECOMMENDATIONS:")
    print(f"   🚀 Kokoro: Real-time conversation, interactive responses")
    print(f"   🎭 Nari Dia: Pre-recorded messages, maximum quality announcements")
    print(f"   🤖 AUTO mode: Smart selection based on context")
    
    # Cleanup
    manager.cleanup()
    print(f"\n✅ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(demo_dual_tts())
