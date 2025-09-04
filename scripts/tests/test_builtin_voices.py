#!/usr/bin/env python3
"""
Quick test for Tortoise built-in voices
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_builtin_voices():
    """Test the built-in voices that should work"""
    print("🧪 Testing Tortoise Built-in Voices")
    print("=" * 50)
    
    builtin_voices = ['angie', 'freeman', 'denise', 'pat', 'william', 'tom', 'lj']
    working_voices = []
    
    try:
        from tortoise.utils.audio import load_voices
        
        for voice in builtin_voices:
            print(f"Testing {voice}... ", end="", flush=True)
            try:
                voice_samples, conditioning_latents = load_voices([voice])
                
                if voice_samples is not None and len(voice_samples) > 0:
                    working_voices.append(voice)
                    sample_count = len(voice_samples)
                    print(f"✅ ({sample_count} samples)")
                else:
                    print("❌ (no samples)")
                    
            except Exception as e:
                print(f"❌ (error: {str(e)[:30]}...)")
        
        print(f"\n📊 Results:")
        print(f"   ✅ Working voices ({len(working_voices)}): {', '.join(working_voices)}")
        
        if working_voices:
            print(f"\n🎯 These voices should be available in your interface!")
        else:
            print(f"\n⚠️ No built-in voices found - this indicates a Tortoise TTS installation issue")
            
        return working_voices
        
    except ImportError as e:
        print(f"❌ Cannot import Tortoise TTS: {e}")
        return []
    except Exception as e:
        print(f"❌ Error testing voices: {e}")
        return []

def test_voice_interface():
    """Test that the TTS interface returns the correct voices"""
    print(f"\n🔧 Testing TTS Service Voice List...")
    
    try:
        from tortoise_tts_implementation_real import create_tortoise_tts
        
        print("Creating TTS service... ", end="", flush=True)
        tts_service = create_tortoise_tts(device='cpu')  # Use CPU for quick test
        print("✅")
        
        print("Getting available voices... ", end="", flush=True)
        voices = tts_service.get_available_voices()
        print(f"✅ Found {len(voices)}")
        
        print(f"Available voices: {', '.join(voices)}")
        
        return voices
        
    except Exception as e:
        print(f"❌ Error testing TTS service: {e}")
        return []

if __name__ == "__main__":
    builtin_voices = test_builtin_voices()
    service_voices = test_voice_interface()
    
    if builtin_voices and service_voices:
        print(f"\n✅ Voice system should be working with {len(service_voices)} voices")
        if set(builtin_voices) != set(service_voices):
            print(f"ℹ️ Built-in voices: {builtin_voices}")
            print(f"ℹ️ Service voices: {service_voices}")
    else:
        print(f"\n❌ Voice system has issues - only 3 voices will be available")
