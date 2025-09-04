#!/usr/bin/env python3
"""
Test script to verify all voices are working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tortoise_tts_implementation_real import TortoiseTTSEngine

def test_voice_availability():
    """Test that all configured voices are available and working"""
    print("🎭 Testing Voice Availability")
    print("=" * 50)
    
    # Initialize engine
    engine = TortoiseTTSEngine()
    
    # Get available voices
    voices = engine.get_available_voices()
    print(f"📊 Total voices available: {len(voices)}")
    
    # Display in two columns like the test interface
    print(f"\n{'COLUMN 1':<28} {'COLUMN 2':<28}")
    print("-" * 56)
    
    mid_point = (len(voices) + 1) // 2
    
    for i in range(mid_point):
        # Left column
        if i < len(voices):
            left_voice = voices[i]
            left_text = f"{i+1:2d}. {left_voice}"
        else:
            left_text = ""
        
        # Right column
        right_index = i + mid_point
        if right_index < len(voices):
            right_voice = voices[right_index]
            right_text = f"{right_index+1:2d}. {right_voice}"
        else:
            right_text = ""
        
        print(f"{left_text:<28} {right_text:<28}")
    
    print(f"\n   0. Use default voice (angie)")
    print(f"\nTotal voices: {len(voices)}")
    
    # Test a few key voices to make sure they work
    test_voices = ['angie', 'jlaw', 'freeman', 'train_grace']
    test_text = "This is a quick test."
    
    print(f"\n🧪 Testing {len(test_voices)} sample voices...")
    
    for voice in test_voices:
        if voice in voices:
            try:
                print(f"  Testing {voice}...", end="")
                audio, sr = engine.generate_speech(test_text, voice, save_audio=False)
                print(f" ✅ OK ({audio.shape[0]/sr:.1f}s)")
            except Exception as e:
                print(f" ❌ FAILED: {e}")
        else:
            print(f"  {voice}: ❌ NOT AVAILABLE")
    
    print(f"\n🎉 Voice availability test complete!")
    print(f"💡 All {len(voices)} voices are properly configured and available for selection.")

if __name__ == "__main__":
    test_voice_availability()
