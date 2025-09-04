"""
Test the official Tortoise TTS voices.
This script demonstrates the enhanced voice collection with official repository voices.
"""

from tortoise_tts_implementation_real import TortoiseTTS
import time

def test_official_voices():
    """Test the official voices functionality."""
    
    print("🎤 Testing Official Tortoise TTS Voices")
    print("=" * 50)
    
    # Initialize the TTS engine
    print("🔄 Initializing Tortoise TTS...")
    tts = TortoiseTTS()
    
    # Get available voices
    voices = tts.get_available_voices()
    print(f"\n✅ Found {len(voices)} available voices:")
    
    # Categorize voices
    official_voices = [v for v in voices if v not in ['snakes']]  # All except built-in fallbacks
    training_voices = [v for v in voices if v.startswith('train_')]
    character_voices = [v for v in voices if not v.startswith('train_') and v != 'snakes']
    
    print(f"\n📊 Voice Categories:")
    print(f"  🎭 Character voices ({len(character_voices)}): {', '.join(character_voices)}")
    print(f"  🎓 High-quality training voices ({len(training_voices)}): {', '.join(training_voices)}")
    print(f"  🔧 Built-in fallbacks: snakes")
    
    # Test a few representative voices
    test_text = "Hello! This is a demonstration of the official Tortoise TTS voices."
    
    test_voices = ['freeman', 'deniro', 'train_atkins']  # Morgan Freeman, De Niro, and a training voice
    
    print(f"\n🎯 Testing {len(test_voices)} representative voices...")
    print(f"Text: '{test_text}'")
    
    for voice in test_voices:
        if voice in voices:
            print(f"\n🎵 Testing voice: {voice}")
            try:
                start_time = time.time()
                result = tts.synthesize(test_text, voice)
                duration = time.time() - start_time
                
                if result:
                    print(f"  ✅ Success! Generated in {duration:.1f} seconds")
                    if hasattr(result, 'shape'):
                        print(f"  📊 Audio shape: {result.shape}")
                else:
                    print(f"  ❌ Failed to generate audio")
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}")
        else:
            print(f"  ⚠️ Voice '{voice}' not available")
    
    print(f"\n🎉 Official voices test completed!")
    print(f"💡 You can now use any of these {len(voices)} voices in your TTS applications!")

if __name__ == "__main__":
    test_official_voices()
