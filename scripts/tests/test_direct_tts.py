#!/usr/bin/env python3
"""
Direct TTS Test
"""
import sys
import os
sys.path.append('.')
sys.path.append('./voicebot_orchestrator')

def test_tts_direct():
    """Test TTS implementation directly"""
    print("🔊 Testing TTS Implementation Directly")
    print("=" * 40)
    
    try:
        from voicebot_orchestrator.tts import KokoroTTS
        
        print("✅ TTS module imported successfully")
        
        # Initialize TTS
        tts = KokoroTTS(voice="af_bella")
        print("✅ TTS initialized")
        
        # Test synthesis
        test_text = "Hello world, this is a test."
        print(f"🎙️ Synthesizing: {test_text}")
        
        import asyncio
        audio_data = asyncio.run(tts.synthesize_speech(test_text))
        
        print(f"✅ Audio generated: {len(audio_data)} bytes")
        
        # Save to file
        with open("test_direct_tts.wav", "wb") as f:
            f.write(audio_data)
        print("💾 Audio saved as: test_direct_tts.wav")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tts_direct()
    if success:
        print("\n🎉 Direct TTS test completed!")
        print("The audio should be more speech-like, not a simple beep.")
    else:
        print("\n💥 Direct TTS test failed!")
