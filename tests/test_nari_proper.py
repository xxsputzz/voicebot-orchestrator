"""
Test Nari Dia-1.6B TTS with proper speaker tags and female voice
"""
import sys
import os
import gc
import torch
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests', 'dia'))

def test_nari_proper_female():
    """Test Nari Dia with proper speaker tags for female voice"""
    print("🚀 Testing Nari Dia-1.6B with proper speaker tags for female voice...")
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
        
        # Test different female voice scenarios
        test_scenarios = [
            {
                "name": "Banking Assistant (Female)",
                "text": "[S1] Hello! Welcome to our banking services. I'm Sarah, your personal banking assistant. How may I help you today? [S2] Hi Sarah, I'd like to check my account balance please. [S1] Of course! I'll be happy to help you with that. Let me access your account information right away.",
                "filename": "nari_banking_female.wav",
                "seed": 42  # Fixed seed for consistency
            },
            {
                "name": "Friendly Conversation (Female)",
                "text": "[S1] Hi there! How are you doing today? I hope you're having a wonderful afternoon. [S2] I'm doing great, thank you for asking! [S1] That's fantastic to hear. Is there anything I can help you with today?",
                "filename": "nari_friendly_female.wav", 
                "seed": 123
            },
            {
                "name": "Professional Service (Female)",
                "text": "[S1] Thank you for calling our customer service line. My name is Emma, and I'll be assisting you today. [S2] Hello Emma, I need help with my recent transaction. [S1] I understand your concern. Let me look into that for you right away.",
                "filename": "nari_professional_female.wav",
                "seed": 456
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n🎤 Test {i}: {scenario['name']}")
            print(f"Text: {scenario['text']}")
            
            synthesis_start = time.time()
            
            # Set seed for consistent voice
            torch.manual_seed(scenario['seed'])
            if torch.cuda.is_available():
                torch.cuda.manual_seed(scenario['seed'])
            
            # Generate audio with proper parameters
            audio = model.generate(
                text=scenario['text'],
                max_tokens=2048,  # Moderate length for natural speech
                cfg_scale=3.0,    # Standard guidance
                temperature=1.2,  # Slight randomness for natural speech
                top_p=0.95,       # Good nucleus sampling
                verbose=True      # Show progress
            )
            
            synthesis_time = time.time() - synthesis_start
            print(f"✅ Synthesis completed in {synthesis_time:.2f} seconds")
            
            # Save audio
            output_path = scenario['filename']
            sample_rate = 44100
            sf.write(output_path, audio, sample_rate)
            print(f"✅ Audio saved to {output_path}")
            
            # Check file info
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024
                duration = len(audio) / sample_rate
                print(f"📊 File size: {file_size:.1f} KB")
                print(f"📊 Duration: {duration:.1f} seconds")
            
            # Brief pause between generations
            time.sleep(1)
        
        # Final memory check
        final_memory = torch.cuda.memory_allocated(0) / 1024**3
        print(f"\n💾 Final GPU memory used: {final_memory:.2f} GB")
        
        print("\n🎉 Nari Dia female voice test completed successfully!")
        print("\n🎧 **Generated Audio Files:**")
        for scenario in test_scenarios:
            if os.path.exists(scenario['filename']):
                print(f"   - {scenario['name']}: {scenario['filename']}")
        
        print(f"\n💡 **Tips for best results:**")
        print(f"   - Each audio uses a different seed for voice variety")
        print(f"   - The [S1] and [S2] tags create dialogue format")
        print(f"   - Moderate text length produces most natural speech")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up GPU memory
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    success = test_nari_proper_female()
    print(f"\nTest result: {'✅ SUCCESS' if success else '❌ FAILED'}")
