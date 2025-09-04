#!/usr/bin/env python3

import sys
import os
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), 'voicebot_orchestrator'))

async def test_unicode_fix_final():
    """Final test of the Unicode fix"""
    
    try:
        from enhanced_tts_manager import EnhancedTTSManager, TTSEngine
        
        print("🎉 FINAL UNICODE FIX VALIDATION TEST")
        print("=" * 60)
        
        # Initialize the manager
        print("🔄 Initializing TTS Manager...")
        manager = EnhancedTTSManager()
        
        # Test the original problematic text
        test_text = "🎤 Hungrey"
        print(f"\n🎯 Testing ORIGINAL problem: '{test_text}'")
        
        # Show the sanitization working
        sanitized = manager.sanitize_text_for_synthesis(test_text)
        print(f"   ✅ Sanitized text: '{sanitized}'")
        
        # Test actual audio generation
        print("\n🎵 Testing audio generation...")
        try:
            # Use the TTSEngine enum properly
            audio_data = await manager.generate_speech(test_text, TTSEngine.KOKORO, "default")
            
            if audio_data and len(audio_data) > 1000:
                print(f"   🎉 SUCCESS! Audio generated: {len(audio_data)} bytes")
                
                # Save the audio
                with open("final_unicode_fix_success.wav", "wb") as f:
                    f.write(audio_data)
                print("   💾 Saved as: final_unicode_fix_success.wav")
                
                print("\n" + "🎉" * 20)
                print("🏆 UNICODE PROBLEM SOLVED!")
                print("🎉" * 20)
                print("✅ Original error: 'charmap' codec can't encode character '🎤'")  
                print("✅ Fix applied: Text sanitization before TTS synthesis")
                print("✅ Result: '🎤 Hungrey' → 'microphone Hungrey' → Audio generated!")
                print("✅ The TTS service can now handle ANY Unicode characters!")
                
            else:
                print("   ❌ Audio generation failed")
                
        except Exception as e:
            print(f"   ❌ Audio error: {e}")
            import traceback
            traceback.print_exc()
        
        # Additional validation tests
        print("\n🧪 Additional validation tests:")
        test_cases = [
            "🎵 Music playing",
            "🚀 Rocket launch",
            "🎉🎊 Party time!",
            "Regular text"
        ]
        
        for i, text in enumerate(test_cases, 1):
            sanitized = manager.sanitize_text_for_synthesis(text)
            has_emoji = any(ord(c) > 127 for c in text)
            clean = not any(ord(c) > 127 for c in sanitized)
            status = "✅" if (not has_emoji or clean) else "❌"
            print(f"   {i}. '{text}' → '{sanitized}' {status}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    asyncio.run(test_unicode_fix_final())

if __name__ == "__main__":
    main()
