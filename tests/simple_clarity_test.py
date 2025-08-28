#!/usr/bin/env python3
"""
Generate a short, simple af_bella test for clarity verification
"""
import asyncio
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from voicebot_orchestrator.tts import KokoroTTS

def get_audio_output_path(filename):
    """Get the path for audio output files."""
    # Create audio_samples directory if it doesn't exist
    audio_dir = os.path.join(os.path.dirname(__file__), "audio_samples")
    os.makedirs(audio_dir, exist_ok=True)
    return os.path.join(audio_dir, filename)

async def generate_simple_test():
    """Generate a simple, short test"""
    print("🧪 Simple Clarity Test - af_bella")
    print("=" * 40)
    
    # Initialize TTS
    tts = KokoroTTS(voice="af_bella")
    
    # Very simple, clear text
    test_text = "Hello, this is a test of the Kokoro voice system."
    
    print(f"📝 Text: {test_text}")
    
    try:
        print("🔊 Generating audio...")
        audio_bytes = await tts.synthesize_speech(test_text)
        
        print(f"✅ Generated {len(audio_bytes)} bytes")
        
        # Save with clear name
        filename = "af_bella_simple_test.wav"
        file_path = get_audio_output_path(filename)
        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
        
        print(f"💾 Saved: {filename}")
        print("🎵 This should be crystal clear!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(generate_simple_test())
