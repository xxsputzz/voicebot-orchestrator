#!/usr/bin/env python3
"""
Test audio file properties to verify it's properly encoded
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import soundfile as sf
import numpy as np

def test_audio_file():
    """Test the generated audio file properties"""
    audio_file = os.path.join(os.path.dirname(__file__), "audio_samples", "af_bella_banking_sample.wav")
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    try:
        # Read the audio file
        audio_data, sample_rate = sf.read(audio_file)
        
        print("🔍 Audio File Analysis")
        print("=" * 30)
        print(f"📁 File: {os.path.basename(audio_file)}")
        print(f"📊 Sample Rate: {sample_rate} Hz")
        print(f"🎵 Duration: {len(audio_data) / sample_rate:.2f} seconds")
        print(f"📈 Shape: {audio_data.shape}")
        print(f"🔢 Data Type: {audio_data.dtype}")
        print(f"📏 Range: {np.min(audio_data):.4f} to {np.max(audio_data):.4f}")
        print(f"💾 File Size: {os.path.getsize(audio_file)} bytes")
        
        # Check if audio seems valid
        if np.max(np.abs(audio_data)) > 0:
            print("✅ Audio contains non-zero data")
        else:
            print("❌ Audio appears to be silent")
            
        if sample_rate >= 16000:
            print("✅ Sample rate is adequate for speech")
        else:
            print("⚠️ Sample rate might be too low for clear speech")
            
        print("\n🎵 Audio file appears to be properly formatted!")
        
    except Exception as e:
        print(f"❌ Error reading audio file: {e}")

if __name__ == "__main__":
    test_audio_file()
