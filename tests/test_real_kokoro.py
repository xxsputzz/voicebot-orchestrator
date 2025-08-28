#!/usr/bin/env python3
"""
Real Kokoro TTS Test

Test the actual Kokoro TTS implementation to generate real voice audio.
"""

import asyncio
import sys
import time
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from voicebot_orchestrator.tts import KokoroTTS
from voicebot_orchestrator.datetime_utils import DateTimeFormatter

def get_audio_output_path(filename):
    """Get the path for audio output files."""
    # Create audio_samples directory if it doesn't exist
    audio_dir = os.path.join(os.path.dirname(__file__), "audio_samples")
    os.makedirs(audio_dir, exist_ok=True)
    return os.path.join(audio_dir, filename)

async def test_kokoro_voices():
    """Test different Kokoro voices."""
    
    print("🎙️ REAL KOKORO TTS TEST")
    print("=" * 50)
    
    # Test different voices
    test_voices = [
        "af_bella",    # Female American
        "am_adam",     # Male American  
        "bf_emma",     # Female British
        "bm_george"    # Male British
    ]
    
    test_text = "Hello! I'm your banking assistant powered by Kokoro TTS. How can I help you today?"
    
    for voice in test_voices:
        print(f"\n🎭 Testing voice: {voice}")
        print("-" * 30)
        
        try:
            # Initialize TTS with specific voice
            tts = KokoroTTS(voice=voice, language="en", speed=1.0)
            
            # Generate speech
            start_time = time.time()
            audio_data = await tts.synthesize_speech(test_text)
            generation_time = time.time() - start_time
            
            # Save to file with meaningful name in audio_samples directory
            audio_filename = DateTimeFormatter.get_audio_filename(f"kokoro_test_{voice}")
            audio_path = get_audio_output_path(audio_filename)
            with open(audio_path, "wb") as f:
                f.write(audio_data)
            
            print(f"✅ Voice: {voice}")
            print(f"   Generation time: {generation_time:.2f}s")
            print(f"   Audio file: {audio_filename}")
            print(f"   Audio size: {len(audio_data)} bytes")
            
        except Exception as e:
            print(f"❌ Failed to generate with {voice}: {e}")
    
    print(f"\n🎉 Kokoro TTS test complete!")
    print(f"Check the generated audio files to hear the different voices.")

async def test_banking_conversation():
    """Test a realistic banking conversation with Kokoro."""
    
    print(f"\n💼 BANKING CONVERSATION TEST")
    print("=" * 50)
    
    # Use the best female voice for banking
    tts = KokoroTTS(voice="af_bella", language="en", speed=0.9)
    
    banking_responses = [
        "Hello! Welcome to First National Bank. I'm your AI assistant. How may I help you today?",
        "I can help you check your account balance. Your current balance is two thousand five hundred forty-three dollars and sixty-seven cents.",
        "You have three recent transactions: a deposit of five hundred dollars, a withdrawal of one hundred dollars, and a transfer of fifty dollars.",
        "I can help you transfer two hundred dollars to your savings account. Please confirm this transaction.",
        "Your transfer has been completed successfully. Is there anything else I can help you with today?",
        "Thank you for banking with us. Have a wonderful day!"
    ]
    
    for i, response in enumerate(banking_responses, 1):
        print(f"\n🤖 Banking Response {i}:")
        print(f"Text: {response}")
        
        try:
            start_time = time.time()
            audio_data = await tts.synthesize_speech(response)
            generation_time = time.time() - start_time
            
            # Save banking conversation audio to audio_samples directory
            audio_filename = DateTimeFormatter.get_audio_filename(f"banking_conversation_{i:02d}")
            audio_path = get_audio_output_path(audio_filename)
            with open(audio_path, "wb") as f:
                f.write(audio_data)
            
            print(f"✅ Generated in {generation_time:.2f}s")
            print(f"   File: {audio_filename}")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    print(f"\n💼 Banking conversation test complete!")

async def main():
    """Main test function."""
    try:
        print("🚀 Starting Kokoro TTS Real Voice Tests")
        print("=" * 60)
        
        # Test individual voices
        await test_kokoro_voices()
        
        # Test banking conversation
        await test_banking_conversation()
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"🔊 Play the generated WAV files to hear real Kokoro TTS voices!")
        
    except KeyboardInterrupt:
        print("\n👋 Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
