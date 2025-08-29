#!/usr/bin/env python3
"""
Nari Dia-1.6B TTS Test
=====================

Test Nari Dia-1.6B TTS model with female voice generation.
Compare it against Kokoro TTS.
"""

import asyncio
import sys
import os
import time
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import both TTS systems
from voicebot_orchestrator.tts import KokoroTTS

# Import Nari Dia
try:
    from dia.model import Dia
    import soundfile as sf
    import numpy as np
    NARI_AVAILABLE = True
    print("✅ Nari Dia-1.6B available")
except ImportError as e:
    NARI_AVAILABLE = False
    print(f"❌ Nari Dia-1.6B not available: {e}")

class NariDiaTTS:
    """Nari Dia-1.6B TTS wrapper with female voice focus."""
    
    def __init__(self, seed: int = 42):
        """Initialize Nari Dia TTS.
        
        Args:
            seed: Random seed for consistent voice generation
        """
        self.seed = seed
        self._model = None
        print(f"🎤 Initializing Nari Dia-1.6B TTS (seed: {seed})")
        
    def _load_model(self):
        """Load the Nari Dia model."""
        if self._model is None and NARI_AVAILABLE:
            try:
                print("🔄 Loading Nari Dia-1.6B model...")
                self._model = Dia.from_pretrained("nari-labs/Dia-1.6B")
                print("✅ Nari Dia-1.6B model loaded successfully")
                return True
            except Exception as e:
                print(f"❌ Failed to load Nari Dia model: {e}")
                return False
        return self._model is not None
        
    async def synthesize_speech(self, text: str, speaker_tag: str = "[S1]") -> bytes:
        """
        Synthesize speech using Nari Dia-1.6B.
        
        Args:
            text: Text to synthesize
            speaker_tag: Speaker tag for voice control ([S1] or [S2])
            
        Returns:
            Audio data as bytes
        """
        if not NARI_AVAILABLE:
            print("⚠️ Nari Dia not available, returning empty audio")
            return b""
            
        if not self._load_model():
            print("⚠️ Could not load Nari model, returning empty audio")
            return b""
            
        try:
            # Set seed for consistent voice generation
            import torch
            torch.manual_seed(self.seed)
            
            # Format text with speaker tag for female voice
            # We'll use techniques to encourage a female-sounding voice:
            # 1. Use [S1] tag (often sounds more feminine)
            # 2. Add emotional context that tends to produce softer tones
            formatted_text = f"{speaker_tag} {text}"
            
            print(f"🎵 Generating with Nari Dia: '{formatted_text}'")
            
            # Generate audio
            start_time = time.time()
            audio_array = self._model.generate(formatted_text)
            generation_time = time.time() - start_time
            
            print(f"✅ Nari Dia generation completed in {generation_time:.2f}s")
            print(f"📊 Audio shape: {audio_array.shape if hasattr(audio_array, 'shape') else 'unknown'}")
            
            # Convert numpy array to bytes
            if isinstance(audio_array, np.ndarray):
                # Nari outputs at 44.1kHz by default
                sample_rate = 44100
                
                # Save to temporary WAV file and read as bytes
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                    
                try:
                    # Write WAV file
                    sf.write(temp_path, audio_array, sample_rate)
                    
                    # Read back as bytes
                    with open(temp_path, 'rb') as f:
                        audio_bytes = f.read()
                    
                    print(f"🔊 Generated {len(audio_bytes)} bytes of audio")
                    return audio_bytes
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            else:
                print("❌ Unexpected audio format from Nari Dia")
                return b""
                
        except Exception as e:
            print(f"❌ Nari Dia synthesis failed: {e}")
            return b""

async def test_tts_comparison():
    """Test and compare Kokoro vs Nari Dia TTS."""
    print("🧪 TTS Comparison Test: Kokoro vs Nari Dia-1.6B")
    print("=" * 60)
    
    # Test phrases with different scenarios
    test_phrases = [
        "Hello, how can I help you with your banking needs today?",
        "Your account balance is $1,234.56. Would you like to hear your recent transactions?",
        "I understand you'd like to transfer money. Let me help you with that process.",
        "Thank you for banking with us. Have a wonderful day!"
    ]
    
    # Initialize TTS systems
    print("\n🔧 Initializing TTS systems...")
    
    # Kokoro TTS (current system)
    print("🎙️ Setting up Kokoro TTS...")
    kokoro_tts = KokoroTTS(voice="af_bella")  # Female voice
    
    # Nari Dia TTS (new system to test)
    print("🎤 Setting up Nari Dia TTS...")
    nari_tts = NariDiaTTS(seed=42)  # Fixed seed for consistent voice
    
    print(f"\n🧪 Testing {len(test_phrases)} phrases...")
    
    results = []
    
    for i, text in enumerate(test_phrases, 1):
        print(f"\n--- Test {i}/{len(test_phrases)} ---")
        print(f"Text: \"{text}\"")
        
        result = {
            'text': text,
            'kokoro_time': 0,
            'kokoro_size': 0,
            'nari_time': 0,
            'nari_size': 0,
            'kokoro_success': False,
            'nari_success': False
        }
        
        # Test Kokoro TTS
        print("\n🎵 Testing Kokoro TTS...")
        try:
            kokoro_start = time.time()
            kokoro_audio = await kokoro_tts.synthesize_speech(text)
            kokoro_time = time.time() - kokoro_start
            
            if kokoro_audio and len(kokoro_audio) > 0:
                result['kokoro_time'] = kokoro_time
                result['kokoro_size'] = len(kokoro_audio)
                result['kokoro_success'] = True
                print(f"✅ Kokoro: {len(kokoro_audio):,} bytes in {kokoro_time:.2f}s")
            else:
                print("❌ Kokoro: Failed to generate audio")
                
        except Exception as e:
            print(f"❌ Kokoro error: {e}")
        
        # Test Nari Dia TTS
        print("\n🎤 Testing Nari Dia TTS...")
        try:
            nari_start = time.time()
            nari_audio = await nari_tts.synthesize_speech(text)
            nari_time = time.time() - nari_start
            
            if nari_audio and len(nari_audio) > 0:
                result['nari_time'] = nari_time
                result['nari_size'] = len(nari_audio)
                result['nari_success'] = True
                print(f"✅ Nari Dia: {len(nari_audio):,} bytes in {nari_time:.2f}s")
            else:
                print("❌ Nari Dia: Failed to generate audio")
                
        except Exception as e:
            print(f"❌ Nari Dia error: {e}")
        
        results.append(result)
    
    # Summary
    print(f"\n📊 Comparison Summary")
    print("=" * 60)
    
    kokoro_success = sum(1 for r in results if r['kokoro_success'])
    nari_success = sum(1 for r in results if r['nari_success'])
    
    print(f"Tests completed: {len(results)}")
    print(f"Kokoro successes: {kokoro_success}/{len(results)}")
    print(f"Nari Dia successes: {nari_success}/{len(results)}")
    
    if kokoro_success > 0:
        avg_kokoro_time = sum(r['kokoro_time'] for r in results if r['kokoro_success']) / kokoro_success
        avg_kokoro_size = sum(r['kokoro_size'] for r in results if r['kokoro_success']) / kokoro_success
        print(f"Kokoro avg time: {avg_kokoro_time:.2f}s")
        print(f"Kokoro avg size: {avg_kokoro_size:,.0f} bytes")
    
    if nari_success > 0:
        avg_nari_time = sum(r['nari_time'] for r in results if r['nari_success']) / nari_success
        avg_nari_size = sum(r['nari_size'] for r in results if r['nari_success']) / nari_success
        print(f"Nari Dia avg time: {avg_nari_time:.2f}s")
        print(f"Nari Dia avg size: {avg_nari_size:,.0f} bytes")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if nari_success == len(results) and kokoro_success == len(results):
        print("✅ Both TTS systems working well!")
        if nari_success > 0 and kokoro_success > 0:
            nari_avg = sum(r['nari_time'] for r in results if r['nari_success']) / nari_success
            kokoro_avg = sum(r['kokoro_time'] for r in results if r['kokoro_success']) / kokoro_success
            if nari_avg < kokoro_avg:
                print("🚀 Nari Dia appears faster")
            elif kokoro_avg < nari_avg:
                print("🚀 Kokoro appears faster")
            else:
                print("⚖️ Both systems have similar speed")
    elif nari_success > kokoro_success:
        print("🎯 Nari Dia appears more reliable")
    elif kokoro_success > nari_success:
        print("🎯 Kokoro appears more reliable")
    else:
        print("⚠️ Both systems had issues - check configuration")

if __name__ == "__main__":
    asyncio.run(test_tts_comparison())
