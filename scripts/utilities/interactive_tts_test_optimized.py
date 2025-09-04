#!/usr/bin/env python3
"""
GPU-Optimized Interactive TTS Test
Includes performance optimizations and realistic time estimates
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
import gc

def optimize_gpu_performance():
    """Apply GPU optimizations for faster generation"""
    print("🚀 APPLYING GPU OPTIMIZATIONS")
    print("=" * 40)
    
    optimizations_applied = []
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        optimizations_applied.append("✅ GPU cache cleared")
    
    # Set optimal GPU settings
    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        optimizations_applied.append("✅ CUDNN optimizations enabled")
        
        # Set memory management (reduced allocation to avoid OOM)
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of GPU memory instead of 90%
        optimizations_applied.append("✅ GPU memory allocation optimized (70%)")
        
        # Set memory allocation config
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        optimizations_applied.append("✅ CUDA memory fragmentation fix applied")
    
    # Force garbage collection
    gc.collect()
    optimizations_applied.append("✅ Memory garbage collection")
    
    for opt in optimizations_applied:
        print(f"   {opt}")
    
    print(f"\n💡 ADDITIONAL OPTIMIZATION TIPS:")
    print(f"   • Close browser tabs using GPU (Chrome, Edge)")
    print(f"   • Close Discord, Steam, other GPU apps")
    print(f"   • Disable Windows GPU scheduling temporarily")
    print(f"   • Use smaller batch sizes for faster iteration")
    
    return len(optimizations_applied)

def get_realistic_estimates(token_tests):
    """Get realistic time estimates based on actual performance data"""
    
    # Performance data from actual tests:
    # - 2048 tokens: 11m55s (715s) generation time
    # - Generated 15.12s audio with 64.3% token efficiency
    # - Rate: ~350 seconds per 1000 tokens
    
    estimates = {}
    total_time = 0
    
    for tokens in token_tests:
        # Base rate: 350 seconds per 1000 tokens
        base_time = (tokens / 1000) * 350
        
        # Add overhead for model warmup (first test is slower)
        if len(estimates) == 0:
            estimated_time = base_time * 1.2  # 20% overhead for first test
        else:
            estimated_time = base_time
        
        estimates[tokens] = estimated_time
        total_time += estimated_time
    
    return estimates, total_time

def interactive_tts_test_optimized():
    """GPU-optimized interactive test with realistic estimates"""
    try:
        print("🎯 GPU-OPTIMIZED INTERACTIVE TTS TEST")
        print("=" * 60)
        
        # Apply optimizations first
        opts_applied = optimize_gpu_performance()
        
        # Additional memory clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Load model
        print("\n📥 Loading Dia model...")
        os.chdir("tests/dia")
        from dia.model import Dia
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load with optimizations and memory management
        torch.cuda.empty_cache()  # Clear before loading
        with torch.no_grad():
            model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
        
        print(f"✅ Model loaded on {device}")
        
        # Show GPU status
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"📊 GPU Memory: {gpu_allocated:.1f}GB allocated, {gpu_reserved:.1f}GB reserved of {gpu_memory:.1f}GB total")
        
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
        
        # Get token count preference with performance warnings
        print("\n🔧 TOKEN CONFIGURATION")
        print("=" * 30)
        print("Available options (with realistic time estimates):")
        print("1. 🚀 Quick test (2048 tokens ~ 12 minutes)")
        print("2. ⚖️ Medium test (8192 tokens ~ 48 minutes)")
        print("3. 🎯 Long test (16384 tokens ~ 96 minutes)")
        print("4. 🔥 Maximum test (32768 tokens ~ 3+ hours)")
        print("5. 🎨 Custom amount")
        print("6. 📊 Multi-level analysis (2048→8192, very long!)")
        print("7. 💡 Optimize settings (reduce tokens for speed)")
        
        choice = input("Choose option (1-7, default=1 for speed): ").strip()
        
        if choice == "1" or choice == "":
            token_tests = [2048]
            print("🚀 Quick test selected - reasonable 12 minute duration")
        elif choice == "2":
            token_tests = [8192]
            print("⚖️ Medium test selected - expect ~48 minutes")
        elif choice == "3":
            confirm = input("⚠️ Long test takes ~96 minutes. Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                token_tests = [16384]
                print("🎯 Long test confirmed")
            else:
                token_tests = [2048]
                print("🚀 Switched to quick test")
        elif choice == "4":
            confirm = input("⚠️ Maximum test takes 3+ HOURS. Are you sure? (y/N): ").strip().lower()
            if confirm == 'y':
                token_tests = [32768]
                print("🔥 Maximum test confirmed - this will take a very long time!")
            else:
                token_tests = [2048]
                print("🚀 Switched to quick test for sanity")
        elif choice == "5":
            custom_tokens = input("Enter custom token count (512-65536): ").strip()
            try:
                custom_count = int(custom_tokens)
                if custom_count < 512:
                    print("⚠️ Minimum 512 tokens")
                    custom_count = 512
                elif custom_count > 65536:
                    print("⚠️ Maximum 65536 tokens")
                    custom_count = 65536
                
                est_time = (custom_count / 1000) * 350 / 60  # minutes
                if est_time > 60:
                    confirm = input(f"⚠️ Custom test will take ~{est_time:.0f} minutes. Continue? (y/N): ").strip().lower()
                    if confirm != 'y':
                        custom_count = 2048
                        print("🚀 Switched to quick test")
                
                token_tests = [custom_count]
                print(f"🎨 Custom test: {custom_count} tokens (~{est_time:.0f} minutes)")
            except ValueError:
                print("❌ Invalid token count, using quick test")
                token_tests = [2048]
        elif choice == "6":
            confirm = input("⚠️ Multi-level analysis takes 2+ HOURS total. Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                token_tests = [2048, 4096, 8192]
                print("📊 Multi-level analysis confirmed")
            else:
                token_tests = [2048]
                print("🚀 Switched to quick test")
        elif choice == "7":
            token_tests = [1024]
            print("💡 Speed-optimized test: 1024 tokens (~6 minutes)")
        else:
            token_tests = [2048]
            print("🚀 Default quick test selected")
        
        # Get realistic time estimates
        estimates, total_time = get_realistic_estimates(token_tests)
        
        print(f"\n⏱️ REALISTIC TIME ESTIMATES")
        print("=" * 40)
        for i, tokens in enumerate(token_tests, 1):
            est_minutes = estimates[tokens] / 60
            print(f"   Test {i}: {tokens} tokens → ~{est_minutes:.0f} minutes")
        
        total_hours = total_time / 3600
        total_minutes = total_time / 60
        
        if total_time > 3600:
            print(f"   📊 Total time: ~{total_hours:.1f} hours")
        else:
            print(f"   📊 Total time: ~{total_minutes:.0f} minutes")
        
        print(f"\n💡 Performance based on actual test data:")
        print(f"   Rate: ~350 seconds per 1000 tokens")
        print(f"   GPU: RTX 4060 with optimizations applied")
        
        # Final confirmation for long tests
        if total_time > 1800:  # More than 30 minutes
            confirm = input(f"\n⚠️ This will take {total_minutes:.0f} minutes. Continue? (y/N): ").strip().lower()
            if confirm != 'y':
                print("❌ Test cancelled by user")
                return False
        
        # Run the tests
        print(f"\n🔄 STARTING OPTIMIZED TESTS")
        print("=" * 50)
        
        all_results = []
        
        for i, test_tokens in enumerate(token_tests, 1):
            print(f"\n🔬 TEST {i}/{len(token_tests)} - {test_tokens} TOKENS")
            print("-" * 40)
            
            # Clear cache between tests
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Set seed
            torch.manual_seed(user_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(user_seed)
            
            est_minutes = estimates[test_tokens] / 60
            print(f"📊 Configuration:")
            print(f"   Seed: {user_seed}")
            print(f"   Tokens: {test_tokens}")
            print(f"   Estimated time: ~{est_minutes:.0f} minutes")
            
            print(f"\n🔄 Generating audio with GPU optimizations...")
            start_time = time.time()
            
            try:
                with torch.no_grad():  # Save memory
                    audio = model.generate(
                        text=formatted_text,
                        max_tokens=test_tokens,
                        cfg_scale=3.0,
                        temperature=1.2,
                        top_p=0.95,
                        verbose=True
                    )
                
                generation_time = time.time() - start_time
                
                # Clear cache after generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Analyze results
                if hasattr(audio, 'shape') and len(audio.shape) > 0:
                    actual_duration = len(audio) / 44100
                    samples = len(audio)
                    
                    # Format times
                    gen_hours = int(generation_time // 3600)
                    gen_minutes = int((generation_time % 3600) // 60)
                    gen_seconds = int(generation_time % 60)
                    gen_time_str = f"{gen_hours:02d}:{gen_minutes:02d}:{gen_seconds:02d}"
                    
                    print(f"\n📊 RESULTS:")
                    print(f"   🎵 Generated duration: {actual_duration:.2f} seconds")
                    print(f"   🔢 Audio samples: {samples:,}")
                    print(f"   ⏱️ Generation time: {gen_time_str}")
                    print(f"   🎯 Tokens used: {test_tokens}")
                    print(f"   📈 Performance: {test_tokens/generation_time:.1f} tokens/sec")
                    print(f"   ⚡ Efficiency: {actual_duration/generation_time:.3f} audio_sec/gen_sec")
                    
                    # Save file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"optimized_test_seed_{user_seed}_tokens_{test_tokens}_{timestamp}.wav"
                    sf.write(filename, audio, 44100)
                    print(f"   💾 Saved: {filename}")
                    
                    # Quality analysis
                    max_amplitude = np.max(np.abs(audio))
                    rms = np.sqrt(np.mean(audio**2))
                    
                    if rms > 0.15:
                        voice_type = "Strong/Confident"
                    elif rms > 0.10:
                        voice_type = "Normal/Balanced"
                    else:
                        voice_type = "Soft/Quiet"
                    
                    print(f"   🎭 Voice character: {voice_type}")
                    print(f"   🔉 Audio quality: Max={max_amplitude:.3f}, RMS={rms:.3f}")
                    
                    # Performance comparison
                    expected_time = estimates[test_tokens]
                    time_ratio = generation_time / expected_time
                    
                    if time_ratio < 0.8:
                        print(f"   🚀 Faster than expected! ({time_ratio:.2f}x)")
                    elif time_ratio > 1.2:
                        print(f"   🐌 Slower than expected ({time_ratio:.2f}x)")
                    else:
                        print(f"   ✅ Performance as expected ({time_ratio:.2f}x)")
                    
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
                        'performance_ratio': time_ratio
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
            print("📈 PERFORMANCE ANALYSIS")
            print("="*60)
            
            for result in all_results:
                perf_status = "🚀" if result['performance_ratio'] < 1.0 else "🐌" if result['performance_ratio'] > 1.2 else "✅"
                print(f"{perf_status} {result['test_tokens']:5d} tokens → {result['actual_duration']:5.1f}s | {result['generation_time']/60:.1f}min | {result['voice_type']}")
            
            # Overall statistics
            avg_perf = sum(r['performance_ratio'] for r in all_results) / len(all_results)
            total_gen_time = sum(r['generation_time'] for r in all_results)
            
            print(f"\n📊 SUMMARY:")
            print(f"   Tests completed: {len(all_results)}")
            print(f"   Average performance: {avg_perf:.2f}x expected")
            print(f"   Total generation time: {total_gen_time/60:.1f} minutes")
            print(f"   Optimizations applied: {opts_applied}")
        
        print("\n✅ Optimized test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = interactive_tts_test_optimized()
    if success:
        print("\n🎉 All optimized tests completed!")
    else:
        print("\n💥 Tests failed or were interrupted")
