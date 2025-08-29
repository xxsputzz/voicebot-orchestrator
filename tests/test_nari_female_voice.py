"""
Test Nari Dia with proper female voice generation
"""
import sys
import os
import gc
import torch
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests', 'dia'))

def test_nari_female_voice():
    """Test Nari Dia with proper formatting for female voice"""
    print("🎤 Testing Nari Dia with proper female voice generation...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        from dia.model import Dia
        import soundfile as sf
        
        # Load model
        device = torch.device("cuda")
        print(f"Loading model on {device}...")
        
        start_time = time.time()
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Test 1: Simple female dialogue with proper speaker tags
        print("\n🎭 Test 1: Simple female banking assistant")
        text1 = "[S1] Hello! Welcome to our banking services. How may I assist you today? I'm here to help with all your financial needs."
        
        audio1 = model.generate(
            text=text1,
            max_tokens=1024,
            cfg_scale=3.0,
            temperature=1.2,
            top_p=0.95
        )
        
        sf.write("nari_female_simple.wav", audio1, 44100)
        print(f"✅ Generated: nari_female_simple.wav ({len(audio1)/44100:.1f}s)")
        
        # Test 2: Dialogue with emotional elements
        print("\n🎭 Test 2: Female voice with emotions")
        text2 = "[S1] (clears throat) Good afternoon! I'm so glad you called today. (laughs) We have some exciting new banking products that might interest you."
        
        audio2 = model.generate(
            text=text2,
            max_tokens=1024,
            cfg_scale=3.0,
            temperature=1.3,
            top_p=0.95
        )
        
        sf.write("nari_female_emotion.wav", audio2, 44100)
        print(f"✅ Generated: nari_female_emotion.wav ({len(audio2)/44100:.1f}s)")
        
        # Test 3: Longer banking conversation
        print("\n🎭 Test 3: Banking conversation")
        text3 = "[S1] Thank you for calling First National Bank. I understand you're interested in opening a new savings account. [S2] Yes, that's right. What are your current interest rates? [S1] Our premium savings account offers two point five percent annual percentage yield. Would you like me to explain the details?"
        
        audio3 = model.generate(
            text=text3,
            max_tokens=1536,  # Longer for conversation
            cfg_scale=2.8,
            temperature=1.1,
            top_p=0.92
        )
        
        sf.write("nari_banking_conversation.wav", audio3, 44100)
        print(f"✅ Generated: nari_banking_conversation.wav ({len(audio3)/44100:.1f}s)")
        
        # Memory usage
        memory_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"\n📊 GPU memory used: {memory_used:.2f} GB")
        
        print(f"\n🎧 **Generated Audio Files:**")
        print(f"   - nari_female_simple.wav: Simple banking greeting")
        print(f"   - nari_female_emotion.wav: Emotional banking assistant") 
        print(f"   - nari_banking_conversation.wav: Full banking conversation")
        print(f"\n✨ All files should contain clear female speech!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Clean up
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    success = test_nari_female_voice()
    print(f"\nTest result: {'✅ SUCCESS' if success else '❌ FAILED'}")
