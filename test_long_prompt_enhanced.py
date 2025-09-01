#!/usr/bin/env python3
"""
Long Prompt TTS Test with Enhanced Progress Tracking
Tests the new token estimation, progress tracking, and seed functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests', 'dia'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import soundfile as sf
import time
import numpy as np
import random
from datetime import datetime

def test_long_prompt_with_progress():
    """Test with your specific long prompt using enhanced progress tracking"""
    
    print("🎯 LONG PROMPT TTS TEST WITH PROGRESS TRACKING")
    print("=" * 60)
    
    try:
        from dia.model import Dia
        
        # Load model
        print("📥 Loading Dia model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
        print(f"✅ Model loaded on {device}")
        
        # Your specific test text
        test_text = """Hello, hello! This is Alex calling with Finally Payoff Debt, your pre-qualification specialist. (laughs) I'm so glad you picked up today. (clears throat) I promise this will be quick, and helpful.

Here's the news: We've rolled out new income-based loan options to help people escape high-interest credit cards and personal loans. (sighs) You know the ones—you pay and pay, but the balance never drops.

Now, listen… (gasps) if you've got steady income, we can help you qualify for a personal loan between ten thousand and one hundred thousand dollars. (coughs) That means instead of juggling multiple bills, you could roll them into one easy payment."""
        
        print(f"📝 Test text length: {len(test_text)} characters")
        print(f"📖 Complete text:")
        print("-" * 60)
        print(test_text)
        print("-" * 60)
        
        # Format for Dia
        formatted_text = f"[S1] {test_text}"
        
        # Generate multiple seeds for testing with EOS analysis
        seeds = [12345, 67890, 24680, 13579, 98765]
        token_tests = [2048, 4096, 8192, 16384]  # Different token levels to test EOS behavior
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n🎲 TEST {i}/{len(seeds)} - SEED: {seed}")
            print("-" * 40)
            
            # Test different token levels with this seed
            for j, test_tokens in enumerate(token_tests, 1):
                print(f"\n🔬 SUB-TEST {j}/{len(token_tests)} - {test_tokens} TOKENS")
                
                # Set seed
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                
                print(f"📊 Token Test Analysis:")
                print(f"   Test tokens: {test_tokens}")
                print(f"   Expected duration: ~{test_tokens/1000:.1f} seconds")
                
                print(f"\n🔄 Generating audio with {test_tokens} tokens...")
                start_time = time.time()
                
                audio = model.generate(
                    text=formatted_text,
                    max_tokens=test_tokens,
                    cfg_scale=3.0,
                    temperature=1.2,
                    top_p=0.95,
                    verbose=True  # Will show enhanced progress with EOS tracking
                )
                
                generation_time = time.time() - start_time
                
                # Analyze results with EOS focus
                if hasattr(audio, 'shape') and len(audio.shape) > 0:
                    actual_duration = len(audio) / 44100
                    samples = len(audio)
                    
                    # Format generation time as HH:MM:SS
                    gen_hours = int(generation_time // 3600)
                    gen_minutes = int((generation_time % 3600) // 60)
                    gen_seconds = int(generation_time % 60)
                    gen_time_str = f"{gen_hours:02d}:{gen_minutes:02d}:{gen_seconds:02d}"
                    
                    print(f"\n📊 RESULTS:")
                    print(f"   🎵 Generated duration: {actual_duration:.2f} seconds")
                    print(f"   🔢 Audio samples: {samples:,}")
                    print(f"   ⏱️ Generation time: {gen_time_str}")
                    print(f"   🎯 Tokens requested: {test_tokens}")
                    print(f"   📈 Audio per token: {actual_duration/test_tokens:.4f} sec/token")
                    print(f"   ⚡ Efficiency: {actual_duration/generation_time:.3f} audio_sec/gen_sec")
                    
                    # Save with seed and token count in filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"eos_test_seed_{seed}_tokens_{test_tokens}_{timestamp}.wav"
                    sf.write(filename, audio, 44100)
                    
                    print(f"   💾 Saved: {filename}")
                    
                    # Check audio quality
                    max_amplitude = np.max(np.abs(audio))
                    rms = np.sqrt(np.mean(audio**2))
                    print(f"   🔉 Max amplitude: {max_amplitude:.3f}")
                    print(f"   🔉 RMS level: {rms:.3f}")
                    
                    # EOS Analysis
                    tokens_per_second = test_tokens / actual_duration if actual_duration > 0 else 0
                    print(f"   🔚 EOS Analysis: {tokens_per_second:.0f} tokens produced {actual_duration:.1f}s audio")
                    
                    if actual_duration < (test_tokens / 1000) * 0.5:  # Less than 50% expected
                        print(f"   📉 Early EOS likely - audio much shorter than token count suggests")
                    elif actual_duration > (test_tokens / 1000) * 1.5:  # More than 150% expected
                        print(f"   📈 Efficient EOS - audio longer than expected from token count")
                    else:
                        print(f"   ✅ Normal EOS behavior - audio length matches token expectation")
                    
                    # Store results for comparison
                    result = {
                        'seed': seed,
                        'test_tokens': test_tokens,
                        'actual_duration': actual_duration,
                        'generation_time': generation_time,
                        'samples': samples,
                        'filename': filename,
                        'max_amplitude': max_amplitude,
                        'rms': rms,
                        'tokens_per_second': tokens_per_second
                    }
                    
                    if 'all_results' not in locals():
                        all_results = [result]
                    else:
                        all_results.append(result)
                        
                else:
                    print(f"   ❌ No audio generated")
                
                print()  # Space between sub-tests
        
        # Final analysis
        if 'all_results' in locals():
            print("🎯 COMPREHENSIVE EOS ANALYSIS")
            print("=" * 50)
            
            # Group by token level for analysis
            token_groups = {}
            for result in all_results:
                tokens = result['test_tokens']
                if tokens not in token_groups:
                    token_groups[tokens] = []
                token_groups[tokens].append(result)
            
            print(f"\n🔬 TOKEN LEVEL ANALYSIS:")
            for tokens in sorted(token_groups.keys()):
                group = token_groups[tokens]
                print(f"\n📊 {tokens} Tokens ({len(group)} tests):")
                
                durations = [r['actual_duration'] for r in group]
                gen_times = [r['generation_time'] for r in group]
                efficiencies = [r['actual_duration']/r['generation_time'] for r in group]
                
                avg_duration = sum(durations) / len(durations)
                avg_gen_time = sum(gen_times) / len(gen_times)
                avg_efficiency = sum(efficiencies) / len(efficiencies)
                
                print(f"   🎵 Avg audio duration: {avg_duration:.2f}s")
                print(f"   ⏱️ Avg generation time: {avg_gen_time:.1f}s")
                print(f"   ⚡ Avg efficiency: {avg_efficiency:.3f}")
                print(f"   📈 Expected duration: ~{tokens/1000:.1f}s")
                print(f"   🎯 Accuracy: {avg_duration/(tokens/1000):.2f}x expected")
                
                # EOS efficiency analysis
                if avg_duration < (tokens/1000) * 0.5:
                    print(f"   📉 Early EOS pattern detected")
                elif avg_duration > (tokens/1000) * 1.5:
                    print(f"   📈 Extended generation pattern")
                else:
                    print(f"   ✅ Normal EOS behavior")
            
            print(f"\n🎲 SEED VARIATION ANALYSIS:")
            seeds_used = list(set([r['seed'] for r in all_results]))
            for seed in sorted(seeds_used):
                seed_results = [r for r in all_results if r['seed'] == seed]
                durations = [r['actual_duration'] for r in seed_results]
                avg_duration = sum(durations) / len(durations)
                print(f"   Seed {seed}: {avg_duration:.2f}s avg ({len(seed_results)} tests)")
            
            # Individual results with voice characterization
            print(f"\n📋 DETAILED RESULTS:")
            for result in all_results:
                seed_voice_type = "Unknown"
                if result['rms'] > 0.15:
                    seed_voice_type = "Strong/Confident"
                elif result['rms'] > 0.10:
                    seed_voice_type = "Normal/Balanced"
                else:
                    seed_voice_type = "Soft/Quiet"
                
                # Format generation time as HH:MM:SS
                gen_time = result['generation_time']
                gen_hours = int(gen_time // 3600)
                gen_minutes = int((gen_time % 3600) // 60)
                gen_seconds = int(gen_time % 60)
                gen_time_str = f"{gen_hours:02d}:{gen_minutes:02d}:{gen_seconds:02d}"
                
                print(f"Seed {result['seed']:6d} | {result['test_tokens']:5d}tok | {result['actual_duration']:5.1f}s | {gen_time_str} | {seed_voice_type:15s} | {result['filename']}")
            
            # Overall statistics
            all_durations = [r['actual_duration'] for r in all_results]
            all_gen_times = [r['generation_time'] for r in all_results]
            
            print(f"\n📊 OVERALL STATISTICS:")
            print(f"   Total tests: {len(all_results)}")
            print(f"   Avg audio duration: {sum(all_durations)/len(all_durations):.2f}s")
            print(f"   Avg generation time: {sum(all_gen_times)/len(all_gen_times):.1f}s")
            print(f"   Duration range: {min(all_durations):.2f}s - {max(all_durations):.2f}s")
            print(f"   Generation time range: {min(all_gen_times):.1f}s - {max(all_gen_times):.1f}s")
            
            # Duration variation analysis
            duration_variation = max(all_durations) - min(all_durations)
            print(f"   Duration variation: {duration_variation:.2f}s")
            
            if duration_variation > 2.0:
                print(f"   🎭 High variation - seeds/tokens create different outputs!")
            else:
                print(f"   📊 Low variation - consistent output across tests")
        
        else:
            print("❌ No results to analyze - all tests may have failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎤 Enhanced Dia TTS Test")
    print("Testing long prompt with multiple seeds and progress tracking")
    print("=" * 70)
    
    success = test_long_prompt_with_progress()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("🎵 Check the generated audio files for different seed voices")
    else:
        print("\n❌ Test failed")
        
    print("=" * 70)
