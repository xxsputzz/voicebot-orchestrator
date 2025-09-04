#!/usr/bin/env python3
"""
Interactive TTS Test with User-Defined Seed and Prompt
Allows full customization of text and generation parameters
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

def interactive_tts_test():
    """Interactive test with user-defined seed and prompt"""
    try:
        print("🎯 INTERACTIVE TTS TEST WITH EOS ANALYSIS")
        print("=" * 60)
        
        # Load model
        print("📥 Loading Dia model...")
        os.chdir("tests/dia")
        from dia.model import Dia
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
        print(f"✅ Model loaded on {device}")
        
        # Get user input for prompt text
        print("\n📝 TEXT INPUT")
        print("=" * 30)
        print("Enter your text (type 'DONE' on a new line to finish):")
        print("Or press Enter for default Alex/Payoff Debt text")
        
        user_lines = []
        first_line = input("> ").strip()
        
        if first_line == "":
            # Use default text
            test_text = """Hello, hello! This is Alex calling with Finally Payoff Debt, your pre-qualification specialist. (laughs) I'm so glad you picked up today. (clears throat) I promise this will be quick, and helpful.
Here's the news: We've rolled out new income-based loan options to help people escape high-interest credit cards and personal loans. (sighs) You know the ones—you pay and pay, but the balance never drops.
Now, listen… (gasps) if you've got steady income, we can help you qualify for a personal loan between ten thousand and one hundred thousand dollars. (coughs) That means instead of juggling multiple bills, you could roll them into one easy payment."""
            print("📖 Using default Alex/Payoff Debt text")
        elif first_line.upper() == "DONE":
            print("❌ No text entered!")
            return False
        else:
            user_lines.append(first_line)
            
            while True:
                line = input("> ").strip()
                if line.upper() == "DONE":
                    break
                user_lines.append(line)
            
            test_text = "\n".join(user_lines)
            print(f"📖 Custom text entered ({len(user_lines)} lines)")
        
        # Format and show text
        formatted_text = f"[S1] {test_text.replace('→', '->')}"  # Unicode fix + Dia format
        print(f"\n📊 Text Analysis:")
        print(f"   Characters: {len(test_text)}")
        print(f"   Lines: {test_text.count(chr(10)) + 1}")
        print(f"   Words (approx): {len(test_text.split())}")
        
        print(f"\n📖 Complete text:")
        print("-" * 60)
        print(test_text)
        print("-" * 60)
        
        # Get user input for seed
        print("\n🎲 SEED SELECTION")
        print("=" * 30)
        seed_input = input("Enter seed number (or press Enter for random): ").strip()
        
        if seed_input == "":
            user_seed = random.randint(1000, 99999)
            print(f"🎲 Random seed generated: {user_seed}")
        else:
            try:
                user_seed = int(seed_input)
                print(f"🎲 Using seed: {user_seed}")
            except ValueError:
                print("❌ Invalid seed, using random")
                user_seed = random.randint(1000, 99999)
                print(f"🎲 Random seed generated: {user_seed}")
        
        # Get token count preference
        print("\n🔧 TOKEN CONFIGURATION")
        print("=" * 30)
        print("Available options:")
        print("1. Quick test (2048 tokens ~ 2s audio)")
        print("2. Medium test (8192 tokens ~ 8s audio)")
        print("3. Long test (16384 tokens ~ 16s audio)")
        print("4. Maximum test (32768 tokens ~ 32s audio)")
        print("5. Custom amount")
        print("6. Multi-level analysis (test multiple token levels)")
        
        choice = input("Choose option (1-6, default=2): ").strip()
        
        if choice == "1":
            token_tests = [2048]
            print("🚀 Quick test selected")
        elif choice == "3":
            token_tests = [16384]
            print("🎯 Long test selected")
        elif choice == "4":
            token_tests = [32768]
            print("🔥 Maximum test selected")
        elif choice == "5":
            custom_tokens = input("Enter custom token count: ").strip()
            try:
                custom_count = int(custom_tokens)
                if custom_count < 512:
                    print("⚠️ Minimum 512 tokens recommended")
                    custom_count = 512
                elif custom_count > 65536:
                    print("⚠️ Maximum 65536 tokens (memory limit)")
                    custom_count = 65536
                token_tests = [custom_count]
                print(f"🎨 Custom test: {custom_count} tokens")
            except ValueError:
                print("❌ Invalid token count, using 8192")
                token_tests = [8192]
        elif choice == "6":
            token_tests = [2048, 4096, 8192, 16384]
            print("📊 Multi-level analysis selected")
        else:  # Default choice == "2" or invalid
            token_tests = [8192]
            print("⚖️ Medium test selected (default)")
        
        # Estimate total time based on actual performance data
        total_tests = len(token_tests)
        
        # Updated estimates based on actual test results:
        # Seed 16969: 2048 tokens took 11:55 (715s) for 15.12s audio
        # Actual performance: ~350s per 1000 tokens (much slower than expected)
        
        max_tokens = max(token_tests)
        if max_tokens <= 2048:
            est_time_per_test = 12 * 60  # 12 minutes for 2048 tokens
        elif max_tokens <= 4096:
            est_time_per_test = 24 * 60  # 24 minutes for 4096 tokens  
        elif max_tokens <= 8192:
            est_time_per_test = 48 * 60  # 48 minutes for 8192 tokens
        elif max_tokens <= 16384:
            est_time_per_test = 96 * 60  # 96 minutes for 16384 tokens
        else:
            est_time_per_test = 180 * 60  # 3+ hours for 32768 tokens
        
        total_est_time = total_tests * est_time_per_test
        
        print(f"\n⏱️ REALISTIC TIME ESTIMATES (Based on Actual Performance)")
        print(f"   Tests to run: {total_tests}")
        print(f"   Est. time per test: ~{est_time_per_test//60:.0f} minutes")
        print(f"   Total estimated time: ~{total_est_time//60:.0f} minutes ({total_est_time//3600:.1f} hours)")
        print(f"   📊 Performance note: GPU generation is ~350s per 1000 tokens")
        
        # Confirm before running
        confirm = input(f"\n🚀 Ready to run {total_tests} test(s)? (Y/n): ").strip().lower()
        if confirm and confirm != 'y' and confirm != 'yes':
            print("❌ Test cancelled by user")
            return False
        
        # Run the tests
        print(f"\n🔄 STARTING TESTS")
        print("=" * 50)
        
        all_results = []
        
        for i, test_tokens in enumerate(token_tests, 1):
            print(f"\n🔬 TEST {i}/{len(token_tests)} - {test_tokens} TOKENS")
            print("-" * 40)
            
            # Set seed
            torch.manual_seed(user_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(user_seed)
            
            print(f"📊 Configuration:")
            print(f"   Seed: {user_seed}")
            print(f"   Tokens: {test_tokens}")
            print(f"   Expected duration: ~{test_tokens/1000:.1f} seconds")
            
            print(f"\n🔄 Generating audio...")
            start_time = time.time()
            
            try:
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
                    print(f"   🎯 Tokens used: {test_tokens}")
                    print(f"   📈 Audio per token: {actual_duration/test_tokens:.4f} sec/token")
                    print(f"   ⚡ Efficiency: {actual_duration/generation_time:.3f} audio_sec/gen_sec")
                    
                    # Save with seed and token count in filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"interactive_test_seed_{user_seed}_tokens_{test_tokens}_{timestamp}.wav"
                    sf.write(filename, audio, 44100)
                    
                    print(f"   💾 Saved: {filename}")
                    
                    # Check audio quality
                    max_amplitude = np.max(np.abs(audio))
                    rms = np.sqrt(np.mean(audio**2))
                    print(f"   🔉 Max amplitude: {max_amplitude:.3f}")
                    print(f"   🔉 RMS level: {rms:.3f}")
                    
                    # Voice characterization
                    if rms > 0.15:
                        voice_type = "Strong/Confident"
                    elif rms > 0.10:
                        voice_type = "Normal/Balanced"
                    else:
                        voice_type = "Soft/Quiet"
                    print(f"   🎭 Voice character: {voice_type}")
                    
                    # EOS Analysis
                    tokens_per_second = test_tokens / actual_duration if actual_duration > 0 else 0
                    expected_duration = test_tokens / 1000
                    efficiency_ratio = actual_duration / expected_duration if expected_duration > 0 else 0
                    
                    print(f"   🔚 EOS Analysis:")
                    print(f"      - Token efficiency: {tokens_per_second:.0f} tokens → {actual_duration:.1f}s audio")
                    print(f"      - Efficiency ratio: {efficiency_ratio:.2f}x expected")
                    
                    if efficiency_ratio < 0.5:
                        print(f"      - 📉 Early EOS likely - audio much shorter than expected")
                    elif efficiency_ratio > 1.5:
                        print(f"      - 📈 Extended generation - audio longer than expected")
                    else:
                        print(f"      - ✅ Normal EOS behavior - audio length as expected")
                    
                    # Store results
                    result = {
                        'seed': user_seed,
                        'test_tokens': test_tokens,
                        'actual_duration': actual_duration,
                        'generation_time': generation_time,
                        'samples': samples,
                        'filename': filename,
                        'max_amplitude': max_amplitude,
                        'rms': rms,
                        'voice_type': voice_type,
                        'tokens_per_second': tokens_per_second,
                        'efficiency_ratio': efficiency_ratio
                    }
                    all_results.append(result)
                    
                else:
                    print(f"   ❌ No audio generated")
            
            except KeyboardInterrupt:
                print(f"\n⚠️ Test interrupted by user")
                break
            except Exception as e:
                print(f"   ❌ Generation failed: {e}")
                continue
        
        # Final analysis
        if all_results:
            print("\n" + "="*60)
            print("📈 COMPREHENSIVE ANALYSIS")
            print("="*60)
            
            if len(all_results) > 1:
                print(f"\n🔬 TOKEN LEVEL COMPARISON:")
                for result in all_results:
                    print(f"   {result['test_tokens']:5d} tokens → {result['actual_duration']:5.1f}s | {result['voice_type']:15s} | {result['efficiency_ratio']:.2f}x efficiency")
                
                # Statistics
                durations = [r['actual_duration'] for r in all_results]
                gen_times = [r['generation_time'] for r in all_results]
                
                print(f"\n📊 OVERALL STATISTICS:")
                print(f"   Tests completed: {len(all_results)}")
                print(f"   Seed used: {user_seed}")
                print(f"   Avg audio duration: {sum(durations)/len(durations):.2f}s")
                print(f"   Avg generation time: {sum(gen_times)/len(gen_times):.1f}s")
                print(f"   Duration range: {min(durations):.2f}s - {max(durations):.2f}s")
                
                # Best result
                best_result = max(all_results, key=lambda x: x['actual_duration'])
                print(f"\n🏆 BEST RESULT:")
                print(f"   Tokens: {best_result['test_tokens']}")
                print(f"   Duration: {best_result['actual_duration']:.2f}s")
                print(f"   File: {best_result['filename']}")
            else:
                result = all_results[0]
                print(f"\n📋 SINGLE TEST RESULTS:")
                print(f"   Seed: {result['seed']}")
                print(f"   Tokens: {result['test_tokens']}")
                print(f"   Duration: {result['actual_duration']:.2f}s")
                print(f"   Voice: {result['voice_type']}")
                print(f"   Efficiency: {result['efficiency_ratio']:.2f}x expected")
                print(f"   File: {result['filename']}")
        
        else:
            print("❌ No successful tests completed")
            
        print("\n✅ Interactive test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = interactive_tts_test()
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n💥 Tests failed or were interrupted")
