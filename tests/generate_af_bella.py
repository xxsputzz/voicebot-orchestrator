#!/usr/bin/env python3
"""
Generate a fresh af_bella voice sample
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

async def generate_af_bella_sample():
    """Generate a clear af_bella voice sample"""
    print("🎙️ Generating fresh af_bella voice sample")
    print("=" * 50)
    
    # Initialize TTS with af_bella voice
    tts = KokoroTTS(voice="af_bella")
    
    # Clear, professional banking text
    test_text = "Hello! Welcome to First National Bank. I'm Bella, your AI banking assistant. How can I help you with your account today?"
    
    print(f"📝 Text: {test_text}")
    print(f"🎭 Voice: af_bella (Female American English)")
    
    try:
        # Generate audio
        print("🔊 Generating audio...")
        audio_bytes = await tts.synthesize_speech(test_text)
        
        print(f"✅ Success! Generated {len(audio_bytes)} bytes of audio")
        
        # Save to file with clear name
        filename = "af_bella_banking_sample.wav"
        file_path = get_audio_output_path(filename)
        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
        
        print(f"💾 Saved to: {file_path}")
        print("🎵 This is your fresh af_bella voice sample!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(generate_af_bella_sample())
