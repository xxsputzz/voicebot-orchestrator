#!/usr/bin/env python3
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

async def test():
    tts = KokoroTTS(voice='bm_george')
    audio = await tts.synthesize_speech('Hello from George, a British male voice.')
    print(f'Generated {len(audio)} bytes')
    
    # Save to audio_samples directory
    file_path = get_audio_output_path('george_test.wav')
    with open(file_path, 'wb') as f:
        f.write(audio)
    print(f'✅ George voice test saved to: {file_path}')

asyncio.run(test())
