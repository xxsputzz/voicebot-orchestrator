#!/usr/bin/env python3
"""
Direct Debug: Compare raw Dia model output with service processed output
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests', 'dia'))

import torch
import soundfile as sf
import time
import numpy as np

def test_raw_dia_vs_service_processing():
    """Test raw Dia model output vs how the service processes it"""
    
    print("🔍 DIRECT DIA MODEL DEBUG")
    print("=" * 50)
    
    try:
        from dia.model import Dia
        
        # Load model exactly like the service does
        print("📥 Loading Dia model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
        print(f"✅ Model loaded on {device}")
        
        # Use the same text as our service tests
        test_text = "[S1] The ancient library held secrets beyond imagination. Dusty tomes lined endless shelves, each containing knowledge from civilizations long forgotten. As Sarah walked through the narrow aisles, her footsteps echoed in the vast silence. The smell of old parchment and leather bindings filled the air. Somewhere in these depths lay the answer she sought - a cure for the mysterious plague that had befallen her village. Time was running out, and the keeper of this knowledge was said to be both wise and dangerous."
        
        print(f"📝 Text length: {len(test_text)} characters")
        
        # Test with exact same parameters as service
        max_tokens = 6120  # Same as our service test calculated
        
        print(f"\n🔄 Testing with {max_tokens} max tokens...")
        print("🎯 Using EXACT same parameters as service:")
        print("   - max_tokens: 6120")
        print("   - cfg_scale: 3.0")
        print("   - temperature: 1.2")
        print("   - top_p: 0.95")
        print("   - verbose: True")
        
        # Set seeds exactly like service
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        start_time = time.time()
        
        audio = model.generate(
            text=test_text,
            max_tokens=max_tokens,
            cfg_scale=3.0,    # Exact same as service
            temperature=1.2,  # Exact same as service
            top_p=0.95,       # Exact same as service
            verbose=True      # Exact same as service
        )
        
        generation_time = time.time() - start_time
        
        print(f"\n🔍 RAW MODEL OUTPUT ANALYSIS:")
        print(f"   Type: {type(audio)}")
        print(f"   Shape: {audio.shape if hasattr(audio, 'shape') else 'No shape'}")
        print(f"   Dtype: {audio.dtype if hasattr(audio, 'dtype') else 'No dtype'}")
        
        if hasattr(audio, 'shape') and len(audio.shape) > 0:
            audio_length = len(audio)
            duration_44k = audio_length / 44100  # Service uses 44.1kHz
            
            print(f"   Audio samples: {audio_length}")
            print(f"   Duration @ 44.1kHz: {duration_44k:.2f} seconds")
            print(f"   Min/Max: {audio.min():.6f} / {audio.max():.6f}")
            print(f"   Mean: {audio.mean():.6f}")
            print(f"   Non-zero: {(audio != 0).sum()}/{len(audio)}")
            
            # Save raw audio exactly like service
            print(f"\n💾 SAVING AUDIO LIKE SERVICE:")
            
            # Method 1: Direct save like service
            sample_rate = 44100
            filename1 = f"debug_raw_service_method_{int(time.time())}.wav"
            sf.write(filename1, audio, sample_rate)
            
            # Verify what was actually saved
            saved_audio, saved_sr = sf.read(filename1)
            saved_duration = len(saved_audio) / saved_sr
            
            print(f"   📁 Saved as: {filename1}")
            print(f"   📊 Saved samples: {len(saved_audio)}")
            print(f"   📊 Saved duration: {saved_duration:.2f} seconds")
            print(f"   📊 Sample rate: {saved_sr} Hz")
            
            # Check if there's any truncation
            if abs(audio_length - len(saved_audio)) > 10:  # Allow small differences
                print(f"   ⚠️ TRUNCATION DETECTED!")
                print(f"      Original: {audio_length} samples")
                print(f"      Saved: {len(saved_audio)} samples")
                print(f"      Lost: {audio_length - len(saved_audio)} samples")
            else:
                print(f"   ✅ No truncation detected")
            
            # Calculate file size like service estimates
            with open(filename1, 'rb') as f:
                file_bytes = f.read()
            
            file_size = len(file_bytes)
            service_estimate = file_size / 32000  # Service calculation
            
            print(f"\n🧮 SERVICE ESTIMATION ANALYSIS:")
            print(f"   File size: {file_size} bytes")
            print(f"   Service estimate: {service_estimate:.2f} seconds (file_size/32000)")
            print(f"   Actual duration: {saved_duration:.2f} seconds")
            print(f"   Estimation error: {abs(service_estimate - saved_duration):.2f} seconds")
            
            # Generate comparison summary
            print(f"\n📊 SUMMARY:")
            print(f"   Generation time: {generation_time:.1f} seconds")
            print(f"   Audio generated: {saved_duration:.2f} seconds")
            print(f"   Service would estimate: {service_estimate:.2f} seconds")
            print(f"   Efficiency: {saved_duration/generation_time:.3f} seconds audio per second generation")
            
            return {
                'generation_time': generation_time,
                'audio_duration': saved_duration,
                'audio_samples': len(saved_audio),
                'file_size': file_size,
                'filename': filename1
            }
        else:
            print("   ❌ No audio data generated")
            return None
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔍 Raw Dia Model vs Service Processing Debug")
    print("Testing to identify where audio truncation occurs")
    print("=" * 60)
    
    result = test_raw_dia_vs_service_processing()
    
    if result:
        print(f"\n🏁 Test completed - check {result['filename']} for audio output")
    else:
        print("\n❌ Test failed")
        
    print("=" * 60)
