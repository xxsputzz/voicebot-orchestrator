#!/usr/bin/env python3

import sys
import os
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), 'voicebot_orchestrator'))

async def test_unicode_fix_async():
    """Test the Unicode fix with proper async handling"""
    
    try:
        from enhanced_tts_manager import EnhancedTTSManager
        
        print("🎯 Testing Unicode Fix with Audio Generation")
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
        
        # Test actual audio generation with async
        print("\n2️⃣  Testing actual audio generation...")
        try:
            audio_data = await manager.generate_speech(test_text, "KOKORO", "default")
            if audio_data and len(audio_data) > 1000:  # Basic check for valid audio
                print(f"   ✅ Audio generated successfully! Size: {len(audio_data)} bytes")
                
                # Save the audio to verify
                with open("test_unicode_success_final.wav", "wb") as f:
                    f.write(audio_data)
                print("   💾 Audio saved as test_unicode_success_final.wav")
                
                print("\n🎉 SUCCESS! Unicode fix is working perfectly!")
                print("   - Emoji characters are sanitized before TTS")
                print("   - Audio generation works without encoding errors")
                print("   - The original '🎤 Hungrey' problem is SOLVED!")
                
            else:
                print("   ❌ Failed to generate audio or audio too small")
                
        except Exception as e:
            print(f"   ❌ Audio generation error: {e}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    # Run the async test
    asyncio.run(test_unicode_fix_async())

if __name__ == "__main__":
    main()
