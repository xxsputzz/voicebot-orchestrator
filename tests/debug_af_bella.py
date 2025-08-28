#!/usr/bin/env python3
"""
Debug and regenerate af_bella banking sample with different approaches
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

async def test_different_texts():
    """Test different text lengths and content to isolate the issue"""
    print("🔍 Debugging af_bella Banking Audio")
    print("=" * 50)
    
    tts = KokoroTTS(voice="af_bella")
    
    # Test different texts progressively
    test_cases = [
        ("short", "Hello, welcome to the bank."),
        ("medium", "Hello, welcome to First National Bank. How can I help you today?"),
        ("long_simple", "Hello and welcome to First National Bank. I am your banking assistant. How may I help you with your account today?"),
        ("original_banking", "Hello! Welcome to First National Bank. I'm Bella, your AI banking assistant. How can I help you with your account today?")
    ]
    
    for test_name, text in test_cases:
        print(f"\n🧪 Testing: {test_name}")
        print(f"📝 Text: {text}")
        print(f"📏 Length: {len(text)} characters")
        
        try:
            audio_bytes = await tts.synthesize_speech(text)
            
            filename = f"af_bella_{test_name}_test.wav"
            file_path = get_audio_output_path(filename)
            with open(file_path, 'wb') as f:
                f.write(audio_bytes)
            
            print(f"✅ Generated: {len(audio_bytes)} bytes → {filename}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print(f"\n🎵 Test all the generated files to see where the quality breaks down!")

if __name__ == "__main__":
    asyncio.run(test_different_texts())
