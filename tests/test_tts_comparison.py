"""
Compare Kokoro vs Nari Dia TTS for female voice quality
"""
import sys
import os
import time
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests', 'dia'))

def test_comparison():
    """Compare Kokoro and Nari Dia TTS"""
    print("🎭 Comparing Kokoro vs Nari Dia TTS for female voice")
    
    test_text = "Hello! Welcome to our banking services. How may I assist you today?"
    
    print(f"Test text: {test_text}")
    print()
    
    # Test 1: Kokoro TTS
    print("1️⃣ Testing Kokoro TTS...")
    try:
        from voicebot_orchestrator.tts import KokoroTTS
        
        kokoro_start = time.time()
        kokoro_tts = KokoroTTS()
        
        # Use af_bella female voice
        kokoro_audio_path = kokoro_tts.generate_speech(test_text, voice_name="af_bella")
        kokoro_time = time.time() - kokoro_start
        
        print(f"✅ Kokoro generation completed in {kokoro_time:.2f} seconds")
        print(f"✅ Kokoro audio saved to: {kokoro_audio_path}")
        
        if os.path.exists(kokoro_audio_path):
            kokoro_size = os.path.getsize(kokoro_audio_path) / 1024
            print(f"📊 Kokoro file size: {kokoro_size:.1f} KB")
        
    except Exception as e:
        print(f"❌ Kokoro failed: {e}")
        kokoro_time = None
        kokoro_audio_path = None
    
    print()
    
    # Test 2: Nari Dia TTS
    print("2️⃣ Testing Nari Dia TTS...")
    try:
        from dia.model import Dia
        import soundfile as sf
        
        nari_start = time.time()
        
        # Load model (cached from previous test)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
        
        # Generate audio with optimized settings
        audio = model.generate(
            text=test_text,
            max_tokens=1024,  # Moderate length
            cfg_scale=2.5,
            temperature=1.1,
            top_p=0.92
        )
        
        # Save audio
        nari_audio_path = "nari_comparison_test.wav"
        sample_rate = 44100
        sf.write(nari_audio_path, audio, sample_rate)
        
        nari_time = time.time() - nari_start
        
        print(f"✅ Nari Dia generation completed in {nari_time:.2f} seconds")
        print(f"✅ Nari Dia audio saved to: {nari_audio_path}")
        
        if os.path.exists(nari_audio_path):
            nari_size = os.path.getsize(nari_audio_path) / 1024
            nari_duration = len(audio) / sample_rate
            print(f"📊 Nari Dia file size: {nari_size:.1f} KB")
            print(f"📊 Nari Dia duration: {nari_duration:.1f} seconds")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Nari Dia failed: {e}")
        nari_time = None
        nari_audio_path = None
    
    print()
    
    # Comparison summary
    print("📊 **COMPARISON SUMMARY**")
    print("=" * 50)
    
    if kokoro_time and nari_time:
        print(f"⏱️  Speed:")
        print(f"   - Kokoro: {kokoro_time:.2f} seconds")
        print(f"   - Nari Dia: {nari_time:.2f} seconds")
        if kokoro_time < nari_time:
            print(f"   🏆 Kokoro is {nari_time/kokoro_time:.1f}x faster")
        else:
            print(f"   🏆 Nari Dia is {kokoro_time/nari_time:.1f}x faster")
    
    print(f"🎯 Quality Factors:")
    print(f"   - Kokoro: Real-time model, optimized for speed")
    print(f"   - Nari Dia: Dialogue-focused, designed for conversations")
    
    print(f"💾 Memory Usage:")
    print(f"   - Kokoro: ~500MB (ONNX model)")
    print(f"   - Nari Dia: ~6.3GB GPU memory (1.6B parameters)")
    
    print(f"🎤 Voice Characteristics:")
    print(f"   - Kokoro af_bella: African female voice")
    print(f"   - Nari Dia: Adaptive dialogue-focused synthesis")
    
    print(f"\n🎧 **Listen to both files to compare quality!**")
    if kokoro_audio_path and os.path.exists(kokoro_audio_path):
        print(f"   - Kokoro: {kokoro_audio_path}")
    if nari_audio_path and os.path.exists(nari_audio_path):
        print(f"   - Nari Dia: {nari_audio_path}")

if __name__ == "__main__":
    test_comparison()
