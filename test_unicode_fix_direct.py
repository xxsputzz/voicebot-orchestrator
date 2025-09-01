#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'voicebot_orchestrator'))

def test_unicode_fix_direct():
    """Test the Unicode fix directly in the enhanced TTS manager"""
    
    try:
        from enhanced_tts_manager import EnhancedTTSManager
        
        print("🎯 Testing Unicode Fix in Enhanced TTS Manager")
        print("=" * 55)
        
        # Initialize the manager
        print("🔄 Initializing TTS Manager...")
        manager = EnhancedTTSManager()
        
        # Test the problematic text with emoji
        test_text = "🎤 Hungrey"
        print(f"\n🧪 Testing with original problematic text: '{test_text}'")
        
        # Test our sanitization method directly
        print("\n1️⃣  Testing text sanitization...")
        sanitized = manager.sanitize_text_for_synthesis(test_text)
        print(f"   Input:      '{test_text}'")
        print(f"   Sanitized:  '{sanitized}'")
        
        if "🎤" not in sanitized and "microphone" in sanitized:
            print("   ✅ Sanitization working correctly!")
        else:
            print("   ❌ Sanitization not working!")
        
        # Test various emoji combinations
        test_cases = [
            "🎤 Hungrey",
            "Hello 🎵 world",
            "🚀 Launch time",
            "⏰ Meeting at 3pm",
            "🎉 Celebration 🎊 party",
            "Normal text without emoji"
        ]
        
        print("\n2️⃣  Testing multiple Unicode cases...")
        for i, text in enumerate(test_cases, 1):
            sanitized = manager.sanitize_text_for_synthesis(text)
            emoji_removed = any(ord(c) > 127 for c in text) and not any(ord(c) > 127 for c in sanitized)
            status = "✅" if emoji_removed or not any(ord(c) > 127 for c in text) else "❌"
            print(f"   {i}. '{text}' → '{sanitized}' {status}")
        
        # Test actual audio generation with Kokoro (fastest)
        print("\n3️⃣  Testing actual audio generation...")
        try:
            audio_data = manager.generate_speech(test_text, "KOKORO", "default")
            if audio_data and len(audio_data) > 1000:  # Basic check for valid audio
                print(f"   ✅ Audio generated successfully! Size: {len(audio_data)} bytes")
                
                # Save the audio to verify
                with open("test_unicode_success_direct.wav", "wb") as f:
                    f.write(audio_data)
                print("   💾 Audio saved as test_unicode_success_direct.wav")
                
            else:
                print("   ❌ Failed to generate audio or audio too small")
                
        except Exception as e:
            print(f"   ❌ Audio generation error: {e}")
        
        print("\n🏁 Unicode fix test complete!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_unicode_fix_direct()
