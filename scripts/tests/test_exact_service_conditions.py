#!/usr/bin/env python3
"""
Reproduce Service Conditions Exactly
Try to understand why service consistently gets 5.76s while direct tests get 14-24s
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests', 'dia'))

import torch
import soundfile as sf
import time
import numpy as np

def test_exact_service_conditions():
    """Test with exact same conditions as service to reproduce the difference"""
    
    print("🔍 EXACT SERVICE CONDITIONS TEST")
    print("=" * 50)
    
    try:
        from dia.model import Dia
        
        # Load model exactly like service
        print("📥 Loading Dia model (service-style)...")
        device = torch.device("cuda")
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
        print(f"✅ Model loaded on {device}")
        
        # Test with exact service text formatting
        base_text = "The ancient library held secrets beyond imagination. Dusty tomes lined endless shelves, each containing knowledge from civilizations long forgotten. As Sarah walked through the narrow aisles, her footsteps echoed in the vast silence. The smell of old parchment and leather bindings filled the air. Somewhere in these depths lay the answer she sought - a cure for the mysterious plague that had befallen her village. Time was running out, and the keeper of this knowledge was said to be both wise and dangerous."
        
        # Format EXACTLY like service does
        formatted_text = f"[S1] {base_text}"
        
        print(f"📝 Text length: {len(base_text)} characters")
        print(f"📝 Formatted: {formatted_text[:100]}...")
        
        # Calculate tokens EXACTLY like service
        estimated_tokens = len(base_text) * 12  # Service's long text calculation
        max_tokens = min(65536, max(4096, estimated_tokens))  # Service's limit
        
        print(f"🧮 Service token calculation:")
        print(f"   Base text: {len(base_text)} chars")
        print(f"   Estimated: {estimated_tokens} tokens")
        print(f"   Final max_tokens: {max_tokens}")
        
        # Test multiple runs with exact same seeds
        results = []
        
        for run in range(3):
            print(f"\n🔄 RUN {run+1}: Exact service parameters")
            
            # Set seeds EXACTLY like service
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
            
            start_time = time.time()
            
            audio = model.generate(
                text=formatted_text,
                max_tokens=max_tokens,
                cfg_scale=3.0,    # Exact service params
                temperature=1.2,  # Exact service params
                top_p=0.95,       # Exact service params
                verbose=True      # Exact service params
            )
            
            generation_time = time.time() - start_time
            
            # Check audio
            if hasattr(audio, 'shape') and len(audio.shape) > 0:
                duration = len(audio) / 44100
                print(f"   🎵 Generated: {len(audio)} samples ({duration:.2f}s)")
                print(f"   ⏱️ Time: {generation_time:.1f}s")
                
                # Save for analysis
                filename = f"exact_service_run_{run+1}_{int(time.time())}.wav"
                sf.write(filename, audio, 44100)
                
                results.append({
                    'run': run+1,
                    'samples': len(audio),
                    'duration': duration,
                    'generation_time': generation_time,
                    'filename': filename
                })
                
                # Check if this matches service output (5.76s = 253,952 samples)
                service_samples = 253952
                if abs(len(audio) - service_samples) < 1000:  # Close match
                    print(f"   🎯 MATCHES SERVICE OUTPUT!")
                elif len(audio) < service_samples:
                    print(f"   📉 Shorter than service output")
                else:
                    print(f"   📈 Longer than service output")
            else:
                print(f"   ❌ No audio generated")
        
        # Analysis
        print(f"\n📊 ANALYSIS OF {len(results)} RUNS:")
        print(f"   Service consistently returns: 253,952 samples (5.76s)")
        
        for result in results:
            ratio = result['samples'] / 253952
            print(f"   Run {result['run']}: {result['samples']} samples ({result['duration']:.2f}s) - {ratio:.2f}x service")
        
        # Check if we can identify the pattern
        if results:
            avg_samples = sum(r['samples'] for r in results) / len(results)
            avg_duration = avg_samples / 44100
            print(f"\n   📊 Average: {avg_samples:.0f} samples ({avg_duration:.2f}s)")
            
            if all(abs(r['samples'] - results[0]['samples']) < 1000 for r in results):
                print(f"   ✅ Consistent generation across runs")
            else:
                print(f"   ⚠️ Inconsistent generation across runs")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_with_service_warmup():
    """Test if service warmup affects generation"""
    
    print(f"\n🧪 SERVICE WARMUP TEST")
    print("=" * 30)
    
    try:
        from dia.model import Dia
        
        device = torch.device("cuda")
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
        
        # Do warmup like service does
        print("🔥 Warming up model (like service)...")
        warmup_audio = model.generate(
            text="[S1] TTS system initialized.",
            max_tokens=256,
            cfg_scale=2.0,
            temperature=1.0,
            top_p=0.9,
            verbose=False
        )
        print(f"   Warmup generated: {len(warmup_audio)} samples")
        
        # Now test with main text
        print("🎯 Testing after warmup...")
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        test_text = "[S1] The ancient library held secrets beyond imagination. Dusty tomes lined endless shelves, each containing knowledge from civilizations long forgotten."
        
        audio = model.generate(
            text=test_text,
            max_tokens=6120,  # Service calculation
            cfg_scale=3.0,
            temperature=1.2,
            top_p=0.95,
            verbose=True
        )
        
        duration = len(audio) / 44100
        print(f"   🎵 After warmup: {len(audio)} samples ({duration:.2f}s)")
        
        # Compare with service result
        if abs(len(audio) - 253952) < 1000:
            print(f"   🎯 MATCHES SERVICE - warmup might be the key!")
        else:
            print(f"   📈 Different from service - warmup not the cause")
            
        return len(audio)
        
    except Exception as e:
        print(f"❌ Warmup test failed: {e}")
        return None

if __name__ == "__main__":
    print("🔍 Exact Service Conditions Reproduction Test")
    print("Attempting to reproduce the exact conditions that lead to 5.76s output")
    print("=" * 70)
    
    # Test 1: Exact service conditions
    results = test_exact_service_conditions()
    
    # Test 2: Service warmup effect
    warmup_result = test_with_service_warmup()
    
    print(f"\n🏁 FINAL ANALYSIS:")
    if results:
        consistent_short = all(r['duration'] < 10 for r in results)
        if consistent_short:
            print("   🔍 Direct tests also produce short audio - issue in model state")
        else:
            print("   🔍 Direct tests produce long audio - issue is in service processing")
    
    print("=" * 70)
