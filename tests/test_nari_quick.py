"""
Quick test of Nari Dia-1.6B TTS with CUDA acceleration
"""
import sys
import os
import gc
import torch
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests', 'dia'))

def test_nari_quick():
    """Quick test of Nari Dia with CUDA acceleration"""
    print("🚀 Quick test of Nari Dia-1.6B with CUDA...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU memory: {gpu_memory:.2f} GB")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n📦 Loading Nari Dia model...")
    try:
        # Import Nari Dia classes
        from dia.model import Dia
        import soundfile as sf
        
        # Initialize with CUDA device
        device = torch.device("cuda")
        print(f"Using device: {device}")
        
        start_time = time.time()
        
        # Load model from Hugging Face
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Check GPU memory usage
        memory_used = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU memory used: {memory_used:.2f} GB")
        print(f"GPU memory reserved: {memory_reserved:.2f} GB")
        
        # Test synthesis with shorter text and fewer tokens
        print("\n🎤 Testing quick voice synthesis...")
        test_text = "Hello world!"
        
        synthesis_start = time.time()
        
        # Generate audio with fewer tokens for speed
        audio = model.generate(
            text=test_text,
            max_tokens=512,  # Much shorter for quick test
            cfg_scale=2.0,   # Lower for speed
            temperature=1.0, # Lower for speed
            top_p=0.9,       # Lower for speed
            verbose=True     # Show progress
        )
        
        synthesis_time = time.time() - synthesis_start
        print(f"✅ Quick synthesis completed in {synthesis_time:.2f} seconds")
        
        # Save audio
        output_path = "nari_quick_test.wav"
        sample_rate = 44100
        sf.write(output_path, audio, sample_rate)
        print(f"✅ Audio saved to {output_path}")
        
        # Check file size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024
            duration = len(audio) / sample_rate
            print(f"Audio file size: {file_size:.1f} KB")
            print(f"Audio duration: {duration:.1f} seconds")
        
        # Final memory check
        final_memory = torch.cuda.memory_allocated(0) / 1024**3
        print(f"Final GPU memory used: {final_memory:.2f} GB")
        
        print("🎉 Nari Dia quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Clean up GPU memory
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    success = test_nari_quick()
    print(f"\nTest result: {'✅ SUCCESS' if success else '❌ FAILED'}")
