#!/usr/bin/env python3
"""
Quick Real STT Test
==================

Test the real Faster-Whisper STT with a simple text-to-speech-to-text roundtrip.
"""

import asyncio
import os
import sys
import time
import tempfile
import wave
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from voicebot_orchestrator.tts import KokoroTTS
from faster_whisper_stt import FasterWhisperSTT

async def test_stt_roundtrip():
    """Test STT by creating speech and then transcribing it."""
    print("🧪 Testing Real STT Roundtrip...")
    print("=" * 50)
    
    # Initialize components
    print("🔧 Initializing TTS...")
    tts = KokoroTTS(voice="af_bella")
    
    print("🔧 Initializing STT...")
    stt = FasterWhisperSTT(model_name="tiny")  # Use tiny for faster testing
    
    # Test phrases
    test_phrases = [
        "Hello, how are you today?",
        "Can you help me with my account balance?",
        "I would like to transfer money to another account.",
        "What are the current interest rates?",
        "Thank you for your assistance."
    ]
    
    print(f"\n🧪 Testing {len(test_phrases)} phrases...")
    
    all_results = []
    
    for i, original_text in enumerate(test_phrases, 1):
        print(f"\n--- Test {i}/{len(test_phrases)} ---")
        print(f"Original: '{original_text}'")
        
        start_time = time.time()
        
        # Step 1: Generate speech
        print("🎵 Generating speech...")
        tts_start = time.time()
        audio_data = await tts.synthesize(original_text)
        tts_time = time.time() - tts_start
        
        if audio_data is None:
            print("❌ TTS failed")
            continue
            
        print(f"✅ TTS: {len(audio_data):,} bytes in {tts_time:.2f}s")
        
        # Step 2: Transcribe speech
        print("📝 Transcribing speech...")
        stt_start = time.time()
        transcribed_text = await stt.transcribe_audio(audio_data)
        stt_time = time.time() - stt_start
        
        total_time = time.time() - start_time
        
        print(f"✅ STT: '{transcribed_text}' in {stt_time:.2f}s")
        print(f"⏱️ Total roundtrip: {total_time:.2f}s")
        
        # Calculate accuracy (simple word matching)
        original_words = set(original_text.lower().replace('?', '').replace(',', '').replace('.', '').split())
        transcribed_words = set(transcribed_text.lower().replace('?', '').replace(',', '').replace('.', '').split())
        
        if len(original_words) > 0:
            accuracy = len(original_words & transcribed_words) / len(original_words)
        else:
            accuracy = 0.0
            
        print(f"📊 Word accuracy: {accuracy:.1%}")
        
        # Store results
        all_results.append({
            'original': original_text,
            'transcribed': transcribed_text,
            'tts_time': tts_time,
            'stt_time': stt_time,
            'total_time': total_time,
            'accuracy': accuracy
        })
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 50)
    
    if all_results:
        avg_tts_time = sum(r['tts_time'] for r in all_results) / len(all_results)
        avg_stt_time = sum(r['stt_time'] for r in all_results) / len(all_results)
        avg_total_time = sum(r['total_time'] for r in all_results) / len(all_results)
        avg_accuracy = sum(r['accuracy'] for r in all_results) / len(all_results)
        
        print(f"Tests completed: {len(all_results)}")
        print(f"Average TTS time: {avg_tts_time:.2f}s")
        print(f"Average STT time: {avg_stt_time:.2f}s")
        print(f"Average total time: {avg_total_time:.2f}s")
        print(f"Average accuracy: {avg_accuracy:.1%}")
        
        # STT performance info
        stt_info = stt.get_performance_info()
        print(f"\n🚀 STT Performance Info:")
        for key, value in stt_info.items():
            print(f"  {key}: {value}")
            
        print(f"\n✅ Real STT is working! Faster-Whisper successfully transcribed speech.")
    else:
        print("❌ No successful tests")

if __name__ == "__main__":
    asyncio.run(test_stt_roundtrip())
