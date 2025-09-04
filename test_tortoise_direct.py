#!/usr/bin/env python3
"""
Direct test of Tortoise TTS initialization
"""
import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

print("🔍 Testing Tortoise TTS Direct Import and Initialization")
print("=" * 60)

# Test 1: Import tortoise modules
print("\n1. Testing tortoise module imports...")
try:
    from tortoise.api import TextToSpeech
    print("✅ Successfully imported TextToSpeech")
except Exception as e:
    print(f"❌ Failed to import TextToSpeech: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from tortoise.utils.audio import load_voices, get_voices
    print("✅ Successfully imported audio utilities")
except Exception as e:
    print(f"❌ Failed to import audio utilities: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check voices
print("\n2. Testing voice discovery...")
try:
    voices = get_voices()
    print(f"✅ Found {len(voices)} voices: {list(voices.keys())[:5]}...")
    
    if 'jlaw' in voices:
        print(f"✅ jlaw voice found with files: {voices['jlaw']}")
    else:
        print("❌ jlaw voice not found")
        
except Exception as e:
    print(f"❌ Failed to get voices: {e}")
    traceback.print_exc()

# Test 3: Initialize TTS
print("\n3. Testing TTS initialization...")
try:
    print("⏳ Initializing TextToSpeech (this may take a moment)...")
    tts = TextToSpeech()
    print("✅ Successfully initialized TextToSpeech")
except Exception as e:
    print(f"❌ Failed to initialize TTS: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test voice loading
print("\n4. Testing voice loading...")
try:
    print("⏳ Loading jlaw voice samples...")
    voice_samples, conditioning_latents = load_voices(['jlaw'])
    
    if voice_samples is not None:
        print(f"✅ Loaded voice samples: {len(voice_samples)} clips")
    elif conditioning_latents is not None:
        print(f"✅ Loaded conditioning latents: {conditioning_latents[0].shape if conditioning_latents else 'None'}")
    else:
        print("❌ No voice data loaded")
        
except Exception as e:
    print(f"❌ Failed to load voice: {e}")
    traceback.print_exc()

# Test 5: Simple synthesis test
print("\n5. Testing simple synthesis...")
try:
    print("⏳ Generating short test audio...")
    test_text = "Hello"
    
    # Try with voice samples
    if 'voice_samples' in locals() and voice_samples is not None:
        audio = tts.tts_with_preset(test_text, voice_samples=voice_samples, preset='ultra_fast')
    elif 'conditioning_latents' in locals() and conditioning_latents is not None:
        audio = tts.tts_with_preset(test_text, conditioning_latents=conditioning_latents, preset='ultra_fast')
    else:
        # Fallback to random voice
        audio = tts.tts_with_preset(test_text, preset='ultra_fast')
    
    print(f"✅ Generated audio: shape {audio.shape}, duration ~{audio.shape[0]/22050:.1f}s")
    
except Exception as e:
    print(f"❌ Failed to generate audio: {e}")
    traceback.print_exc()

print("\n🎉 All tests completed!")
