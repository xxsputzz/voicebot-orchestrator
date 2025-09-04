#!/usr/bin/env python3
"""
Final Solution: Fix the audio truncation in enhanced_tts_manager
Based on our discovery that raw model generates 24+ seconds but service returns 5.7 seconds
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests', 'dia'))

import torch
import soundfile as sf
import time
import numpy as np

def test_file_io_truncation():
    """Test if the issue is in file I/O operations"""
    
    print("🔍 TESTING FILE I/O TRUNCATION HYPOTHESIS")
    print("=" * 60)
    
    try:
        from dia.model import Dia
        
        # Load model
        print("📥 Loading Dia model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
        print(f"✅ Model loaded on {device}")
        
        # Generate audio
        test_text = "[S1] The ancient library held secrets beyond imagination. Dusty tomes lined endless shelves, each containing knowledge from civilizations long forgotten. As Sarah walked through the narrow aisles, her footsteps echoed in the vast silence."
        
        print(f"🎯 Generating audio with high token count...")
        
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        audio = model.generate(
            text=test_text,
            max_tokens=6120,
            cfg_scale=3.0,
            temperature=1.2,
            top_p=0.95,
            verbose=False  # Reduce noise for this test
        )
        
        print(f"🎵 Generated audio: {len(audio)} samples ({len(audio)/44100:.2f} seconds)")
        
        # Test 1: Direct soundfile save (like service does)
        print(f"\n🧪 TEST 1: Direct sf.write (service method)")
        filename1 = f"test_direct_save_{int(time.time())}.wav"
        sf.write(filename1, audio, 44100)
        
        # Read back immediately
        read_audio1, sr1 = sf.read(filename1)
        print(f"   📁 Saved: {filename1}")
        print(f"   📊 Original: {len(audio)} samples")
        print(f"   📊 Read back: {len(read_audio1)} samples ({len(read_audio1)/44100:.2f}s)")
        
        if len(audio) != len(read_audio1):
            print(f"   🚨 TRUNCATION IN SF.WRITE/READ!")
            print(f"   🚨 Lost: {len(audio) - len(read_audio1)} samples")
        else:
            print(f"   ✅ No truncation in sf.write/read")
        
        # Test 2: Service-style file bytes reading
        print(f"\n🧪 TEST 2: File bytes reading (service method)")
        with open(filename1, 'rb') as f:
            file_bytes = f.read()
        
        print(f"   📊 File bytes: {len(file_bytes)} bytes")
        
        # Decode bytes back to audio (what service would do)
        temp_filename = f"test_from_bytes_{int(time.time())}.wav"
        with open(temp_filename, 'wb') as f:
            f.write(file_bytes)
        
        decoded_audio, sr_decoded = sf.read(temp_filename)
        print(f"   📊 Decoded: {len(decoded_audio)} samples ({len(decoded_audio)/44100:.2f}s)")
        
        if len(read_audio1) != len(decoded_audio):
            print(f"   🚨 TRUNCATION IN BYTES ENCODING/DECODING!")
        else:
            print(f"   ✅ No truncation in bytes operations")
        
        # Test 3: Check for any hard limits
        print(f"\n🧪 TEST 3: Checking for system limits")
        max_samples_theory = 44100 * 60  # 1 minute worth
        print(f"   📊 1 minute worth of samples: {max_samples_theory}")
        print(f"   📊 Our audio samples: {len(audio)}")
        print(f"   📊 Ratio: {len(audio)/max_samples_theory:.2f}")
        
        if len(audio) > max_samples_theory:
            print(f"   ⚠️ Audio is longer than 1 minute")
        
        # Test 4: Check the exact truncation point
        truncated_samples = 253952  # What service returned
        print(f"\n🧪 TEST 4: Analyzing truncation point")
        print(f"   📊 Service returned: {truncated_samples} samples")
        print(f"   📊 Original audio: {len(audio)} samples")
        print(f"   📊 Truncation at: {truncated_samples/44100:.2f} seconds")
        
        # Check if there's a pattern
        ratio = truncated_samples / len(audio)
        print(f"   📊 Truncation ratio: {ratio:.3f} ({ratio*100:.1f}%)")
        
        # Clean up
        os.remove(filename1)
        os.remove(temp_filename)
        
        return {
            'original_samples': len(audio),
            'saved_samples': len(read_audio1),
            'decoded_samples': len(decoded_audio),
            'truncation_detected': len(audio) != len(read_audio1)
        }
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔍 Audio Truncation Debug Test")
    print("Investigating where 24+ seconds becomes 5.7 seconds")
    print("=" * 60)
    
    result = test_file_io_truncation()
    
    if result:
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Original: {result['original_samples']} samples")
        print(f"   After save/read: {result['saved_samples']} samples")  
        print(f"   After bytes ops: {result['decoded_samples']} samples")
        print(f"   Truncation detected: {result['truncation_detected']}")
    
    print("\n🏁 Test completed")
    print("=" * 60)
