#!/usr/bin/env python3
"""
Test script for real Tortoise TTS functionality
"""
import sys
import os
import torchaudio

# Use conda python path
sys.path.insert(0, os.path.abspath('.'))

try:
    from tortoise.api import TextToSpeech
    print("✅ Successfully imported TextToSpeech from tortoise.api")
    
    # Initialize TTS
    print("🔄 Initializing TextToSpeech...")
    tts = TextToSpeech()
    print("✅ TextToSpeech initialized successfully!")
    
    # Test with a voice sample
    test_text = "Hello, this is a test of the real Tortoise TTS system."
    print(f"🔄 Generating speech for: '{test_text}'")
    
    # Generate speech using built-in voice (by name)
    gen = tts.tts_with_preset(test_text, voice_samples=None, conditioning_latents=None, preset='ultra_fast')
    print("✅ Speech generated successfully!")
    
    # Save the result
    output_path = "test_real_tortoise_output.wav"
    torchaudio.save(output_path, gen.squeeze(0).cpu(), 24000)
    print(f"✅ Audio saved to: {output_path}")
    
    print("\n🎉 Real Tortoise TTS is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
