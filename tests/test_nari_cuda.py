"""
Test Nari Dia-1.6B TTS with CUDA acceleration
"""
import sys
import os
import gc
import torch
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'dia'))

def test_nari_cuda():
    """Test Nari Dia with CUDA acceleration"""
    print("🚀 Testing Nari Dia-1.6B with CUDA...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test GPU acceleration")
        return False
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU memory: {gpu_memory:.2f} GB")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n📦 Loading Nari Dia model...")
    try:
        # Debug: Show the path being added
        dia_path = os.path.join(os.path.dirname(__file__), 'dia')
        print(f"Adding path: {dia_path}")
        print(f"Directory exists: {os.path.exists(dia_path)}")
        print(f"Contents: {os.listdir(dia_path) if os.path.exists(dia_path) else 'Path not found'}")
        
        # Import Nari Dia classes
        from dia.model import Dia
        import soundfile as sf
        
        # Initialize with CUDA device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        start_time = time.time()
        
        # Load model from Hugging Face
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU memory used: {memory_used:.2f} GB")
            print(f"GPU memory reserved: {memory_reserved:.2f} GB")
        
        # Test synthesis
        print("\n🎤 Testing voice synthesis...")
        test_text = "Hello! I'm testing the Nari Dia text-to-speech system with GPU acceleration. How does this sound?"
        
        synthesis_start = time.time()
        
        # Generate audio
        audio = model.generate(
            text=test_text,
            max_tokens=3072,  # Use default max tokens
            cfg_scale=3.0,
            temperature=1.3,
            top_p=0.95
        )
        
        synthesis_time = time.time() - synthesis_start
        print(f"✅ Synthesis completed in {synthesis_time:.2f} seconds")
        
        # Save audio
        output_path = "nari_cuda_test_output.wav"
        sample_rate = 44100  # Standard sample rate
        sf.write(output_path, audio, sample_rate)
        print(f"✅ Audio saved to {output_path}")
        
        # Check file size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024
            print(f"Audio file size: {file_size:.1f} KB")
        
        # Final memory check
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f"Final GPU memory used: {final_memory:.2f} GB")
        
        print("🎉 Nari Dia CUDA test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure Nari Dia is properly installed")
        return False
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❌ GPU out of memory: {e}")
            print("The model requires more VRAM than available (8GB)")
            print("Recommended: 10GB+ VRAM")
            return False
        else:
            print(f"❌ Runtime error: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    finally:
        # Clean up GPU memory
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    success = test_nari_cuda()
    print(f"\nTest result: {'✅ SUCCESS' if success else '❌ FAILED'}")
