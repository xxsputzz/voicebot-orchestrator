#!/usr/bin/env python3
"""
STT Debug Test
==============

Test the STT pipeline with better audio handling.
"""

import asyncio
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from faster_whisper_stt import FasterWhisperSTT
from voicebot_orchestrator.tts import KokoroTTS

async def test_stt_with_generated_audio():
    """Test STT using TTS-generated audio."""
    print("🧪 Testing STT with TTS-generated audio...")
    
    # Initialize components
    print("🔧 Initializing TTS...")
    tts = KokoroTTS(voice="af_bella")
    
    print("🔧 Initializing STT...")
    stt = FasterWhisperSTT(model_name="tiny")  # Use tiny for faster testing
    
    # Test phrases
    test_text = "Hello, can you hear me clearly?"
    
    print(f"\n🎵 Generating speech: '{test_text}'")
    audio_data = await tts.synthesize_speech(test_text)
    
    if audio_data is None or len(audio_data) == 0:
        print("❌ TTS failed to generate audio")
        return
    
    print(f"✅ Generated {len(audio_data)} bytes of audio")
    
    # Test STT
    print(f"\n📝 Testing STT transcription...")
    start_time = time.time()
    transcribed_text = await stt.transcribe_audio(audio_data)
    stt_time = time.time() - start_time
    
    print(f"✅ STT completed in {stt_time:.2f}s")
    print(f"Original:    '{test_text}'")
    print(f"Transcribed: '{transcribed_text}'")
    
    # Check if transcription is reasonable
    if transcribed_text and len(transcribed_text) > 5:
        print("✅ STT appears to be working!")
    else:
        print("⚠️ STT may have issues")

if __name__ == "__main__":
    asyncio.run(test_stt_with_generated_audio())
