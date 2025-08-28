#!/usr/bin/env python3
"""
Simple Kokoro TTS test to verify voice synthesis
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

async def test_simple_kokoro():
    """Test basic Kokoro TTS functionality"""
    print("🧪 Simple Kokoro TTS Test")
    print("=" * 40)
    
    # Initialize TTS
    tts = KokoroTTS(voice="af_bella")
    
    # Test short text
    test_text = "Hello, this is Kokoro TTS speaking."
    print(f"📝 Text: {test_text}")
    
    try:
        # Generate audio
        print("🔊 Generating audio...")
        audio_bytes = await tts.synthesize_speech(test_text)
        
        print(f"✅ Success! Generated {len(audio_bytes)} bytes of audio")
        
        # Save to file in audio_samples directory
        filename = f"simple_kokoro_test.wav"
        file_path = get_audio_output_path(filename)
        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
        
        print(f"💾 Saved to: {file_path}")
        print("🎵 Play the file to hear the Kokoro voice!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_kokoro())
