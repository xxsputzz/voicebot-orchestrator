"""
Focused test of Nari Dia-1.6B with persistent loading
Get real performance numbers for comparison with Kokoro
"""
import asyncio
import time
import torch
import gc
import sys
import os
import tempfile
import soundfile as sf
from pathlib import Path

# Add Nari Dia to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests', 'dia'))

class NariDiaPersistentTest:
    """Test Nari Dia with persistent model loading"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.loaded = False
    
    def load_model(self):
        """Load and warm up Nari Dia model"""
        print("🚀 Loading Nari Dia-1.6B for persistent testing...")
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available - Nari Dia requires GPU")
            return False
        
        try:
            from dia.model import Dia
            
            self.device = torch.device("cuda")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"📊 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
            start_time = time.time()
            
            # Load model
            print("📦 Loading model from HuggingFace...")
            self.model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=self.device)
            
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f}s")
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"📊 GPU memory: {memory_used:.2f}GB used, {memory_reserved:.2f}GB reserved")
            
            # Warm up with a test generation
            print("🔥 Warming up model...")
            warmup_start = time.time()
            
            warmup_audio = self.model.generate(
                text="[S1] System initialization test.",
                max_tokens=256,
                cfg_scale=2.0,
                temperature=1.0,
                top_p=0.9,
                verbose=False
            )
            
            warmup_time = time.time() - warmup_start
            print(f"✅ Warmup completed in {warmup_time:.2f}s")
            
            self.loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Nari Dia: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_speech(self, text: str, max_tokens: int = 1024) -> tuple[float, str]:
        """Generate speech and return (generation_time, audio_file_path)"""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Format text properly for dialogue
            if not text.startswith("[S1]"):
                formatted_text = f"[S1] {text}"
            else:
                formatted_text = text
            
            print(f"🎤 Generating: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            
            start_time = time.time()
            
            # Generate audio
            audio = self.model.generate(
                text=formatted_text,
                max_tokens=max_tokens,
                cfg_scale=2.5,
                temperature=1.1,
                top_p=0.92,
                verbose=False  # Reduce output for cleaner timing
            )
            
            generation_time = time.time() - start_time
            
            # Save to unique file to avoid conflicts
            timestamp = int(time.time() * 1000)
            output_path = f"nari_test_{timestamp}.wav"
            
            # Save audio with proper cleanup
            sample_rate = 44100
            sf.write(output_path, audio, sample_rate)
            
            # Check file was created successfully
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024
                duration = len(audio) / sample_rate
                print(f"   ⏱️  Generation: {generation_time:.3f}s")
                print(f"   🎵 Duration: {duration:.1f}s")
                print(f"   📁 File: {file_size:.1f}KB")
                print(f"   ⚡ Realtime factor: {generation_time/duration:.2f}x")
                
                return generation_time, output_path
            else:
                raise RuntimeError("Audio file was not created")
                
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            raise e
    
    def benchmark_performance(self):
        """Benchmark Nari Dia performance with various text lengths"""
        print("\n🏁 NARI DIA PERSISTENT PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        test_cases = [
            {
                "name": "Short Response",
                "text": "Hello, how can I help you today?",
                "max_tokens": 512
            },
            {
                "name": "Medium Response", 
                "text": "I understand your concern about your account balance. Let me check that information for you right away.",
                "max_tokens": 768
            },
            {
                "name": "Long Response",
                "text": "Thank you for calling our banking services. I've reviewed your account and can confirm your current balance is two thousand four hundred and fifty-seven dollars and thirty-two cents. Your last transaction was a deposit of three hundred dollars on Monday. Is there anything else I can help you with today?",
                "max_tokens": 1024
            },
            {
                "name": "Complex Banking",
                "text": "I've processed your loan application and I'm pleased to inform you that you've been pre-approved for a personal loan of up to fifteen thousand dollars at an interest rate of four point nine percent. We can discuss the terms and monthly payment options that would work best for your budget.",
                "max_tokens": 1024
            }
        ]
        
        results = []
        total_generation_time = 0
        total_audio_duration = 0
        generated_files = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['name']}:")
            
            try:
                gen_time, audio_file = self.generate_speech(
                    test_case['text'], 
                    test_case['max_tokens']
                )
                
                # Calculate audio duration
                audio_data, sample_rate = sf.read(audio_file)
                duration = len(audio_data) / sample_rate
                
                results.append({
                    'name': test_case['name'],
                    'generation_time': gen_time,
                    'audio_duration': duration,
                    'realtime_factor': gen_time / duration,
                    'text_length': len(test_case['text'])
                })
                
                total_generation_time += gen_time
                total_audio_duration += duration
                generated_files.append(audio_file)
                
                # Status assessment
                if gen_time < 1.0:
                    status = "🚀 REAL-TIME"
                elif gen_time < 2.0:
                    status = "⚡ FAST"
                elif gen_time < 4.0:
                    status = "🔄 ACCEPTABLE"
                else:
                    status = "⏳ SLOW"
                
                print(f"   {status} - {gen_time:.3f}s generation for {duration:.1f}s audio")
                
            except Exception as e:
                print(f"   ❌ FAILED: {e}")
                results.append({
                    'name': test_case['name'],
                    'generation_time': None,
                    'error': str(e)
                })
        
        # Summary
        print(f"\n📊 NARI DIA PERFORMANCE SUMMARY")
        print("=" * 50)
        
        successful_results = [r for r in results if r.get('generation_time') is not None]
        
        if successful_results:
            avg_generation = total_generation_time / len(successful_results)
            avg_audio = total_audio_duration / len(successful_results)
            avg_realtime = avg_generation / avg_audio
            
            print(f"🎯 Average generation time: {avg_generation:.3f}s")
            print(f"🎵 Average audio duration: {avg_audio:.1f}s")
            print(f"⚡ Average realtime factor: {avg_realtime:.2f}x")
            
            # Performance categories
            real_time = sum(1 for r in successful_results if r['generation_time'] < 1.0)
            fast = sum(1 for r in successful_results if 1.0 <= r['generation_time'] < 2.0)
            acceptable = sum(1 for r in successful_results if 2.0 <= r['generation_time'] < 4.0)
            slow = sum(1 for r in successful_results if r['generation_time'] >= 4.0)
            
            print(f"\n🏅 Performance Distribution:")
            print(f"   🚀 Real-time (<1s): {real_time}/{len(successful_results)}")
            print(f"   ⚡ Fast (1-2s): {fast}/{len(successful_results)}")
            print(f"   🔄 Acceptable (2-4s): {acceptable}/{len(successful_results)}")
            print(f"   ⏳ Slow (>4s): {slow}/{len(successful_results)}")
            
            # Real-time capability assessment
            real_time_capable = avg_generation < 1.5
            print(f"\n🎯 Real-time conversation capable: {'✅ YES' if real_time_capable else '❌ NO'}")
            
            if not real_time_capable:
                target_speedup = 1.0 / avg_generation
                print(f"💡 Need {1/target_speedup:.1f}x speedup for real-time")
        
        print(f"\n🔊 Generated audio files:")
        for file_path in generated_files:
            if os.path.exists(file_path):
                print(f"   • {file_path}")
        
        return results
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        if self.model:
            del self.model
            torch.cuda.empty_cache()
            gc.collect()
            print("✅ GPU memory cleared")

async def test_nari_persistent():
    """Main test function"""
    tester = NariDiaPersistentTest()
    
    try:
        # Load model
        if not tester.load_model():
            print("❌ Cannot proceed without model")
            return
        
        # Run benchmark
        results = tester.benchmark_performance()
        
        # Compare with Kokoro (from previous test)
        print(f"\n🏆 COMPARISON WITH KOKORO")
        print("=" * 40)
        print("Kokoro TTS (persistent):")
        print("   • Average: 0.787s")
        print("   • Range: 0.503-0.973s")
        print("   • Real-time capable: ✅")
        print()
        
        successful_nari = [r for r in results if r.get('generation_time') is not None]
        if successful_nari:
            avg_nari = sum(r['generation_time'] for r in successful_nari) / len(successful_nari)
            print("Nari Dia (persistent):")
            print(f"   • Average: {avg_nari:.3f}s")
            min_time = min(r['generation_time'] for r in successful_nari)
            max_time = max(r['generation_time'] for r in successful_nari)
            print(f"   • Range: {min_time:.3f}-{max_time:.3f}s")
            print(f"   • Real-time capable: {'✅' if avg_nari < 1.5 else '❌'}")
            
            if avg_nari > 0.787:
                slower_factor = avg_nari / 0.787
                print(f"   • {slower_factor:.1f}x slower than Kokoro")
            else:
                faster_factor = 0.787 / avg_nari
                print(f"   • {faster_factor:.1f}x faster than Kokoro")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tester.cleanup()

if __name__ == "__main__":
    asyncio.run(test_nari_persistent())
