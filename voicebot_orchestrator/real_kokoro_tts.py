"""
Real Kokoro TTS Implementation
Uses the actual Kokoro ONNX model to generate real speech
"""
import asyncio
import tempfile
import os
from pathlib import Path
from typing import Optional
import logging

class RealKokoroTTS:
    """Real Kokoro TTS using ONNX model"""
    
    def __init__(self, voice: str = "af_bella", language: str = "en", speed: float = 1.0):
        self.voice = voice
        self.language = language
        self.speed = speed
        self._kokoro_engine = None
        
        # Model paths
        self.project_root = Path(__file__).parent.parent
        self.model_path = self.project_root / "kokoro-v1.0.onnx"
        self.voices_path = self.project_root / "voices-v1.0.bin"
        
        print("[INIT] Initializing Real KokoroTTS")
        print(f"  Model: {self.model_path}")
        print(f"  Voices: {self.voices_path}")
        print(f"  Voice: {voice}")
    
    def _load_kokoro_engine(self) -> bool:
        """Load real Kokoro TTS engine"""
        if self._kokoro_engine is not None:
            return True
            
        try:
            # Check if model files exist
            if not self.model_path.exists():
                print(f"[ERROR] Model file not found: {self.model_path}")
                return False
                
            if not self.voices_path.exists():
                print(f"[ERROR] Voices file not found: {self.voices_path}")
                return False
            
            # Try to import and initialize Kokoro
            try:
                from kokoro_onnx import Kokoro
                print("[LOADING] Loading Kokoro ONNX engine...")
                
                self._kokoro_engine = Kokoro(
                    model_path=str(self.model_path),
                    voices_path=str(self.voices_path)
                )
                
                print("[OK] Real Kokoro TTS engine loaded successfully!")
                return True
                
            except ImportError as e:
                print(f"[ERROR] kokoro_onnx package not available: {e}")
                print("   Install with: pip install kokoro-onnx")
                return False
                
            except Exception as e:
                print(f"[ERROR] Failed to load Kokoro engine: {e}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error loading Kokoro: {e}")
            return False
    
    async def synthesize_speech(self, text: str, format: str = "wav") -> bytes:
        """
        Synthesize real speech from text using Kokoro ONNX
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
            
        # Try to load real Kokoro engine
        if not self._load_kokoro_engine():
            print("[WARNING] Real Kokoro not available, falling back to speech-like patterns")
            return await self._fallback_synthesis(text)
        
        try:
            print(f"[SYNTH] Synthesizing with real Kokoro: '{text[:50]}...'")
            
            # Generate speech with real Kokoro - returns (audio_array, sample_rate)
            audio_result = await asyncio.to_thread(
                self._kokoro_engine.create,
                text=text,
                voice=self.voice,
                speed=self.speed
            )
            
            # Extract audio array and sample rate
            audio_array, sample_rate = audio_result
            print(f"[OK] Real speech generated: {len(audio_array)} samples at {sample_rate}Hz")
            
            # Convert numpy array to WAV bytes
            wav_bytes = self._numpy_to_wav(audio_array, sample_rate)
            print(f"[OK] WAV file created: {len(wav_bytes)} bytes")
            return wav_bytes
            
        except Exception as e:
            print(f"[ERROR] Real Kokoro synthesis failed: {e}")
            print("[WARNING] Falling back to speech-like patterns")
            return await self._fallback_synthesis(text)
    
    def _numpy_to_wav(self, audio_array, sample_rate: int) -> bytes:
        """Convert numpy audio array to WAV bytes"""
        import wave
        import io
        import numpy as np
        
        # Ensure audio is in the right format
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
            
        # Normalize and convert to 16-bit PCM
        audio_array = np.clip(audio_array, -1.0, 1.0)  # Clip to valid range
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    async def _fallback_synthesis(self, text: str) -> bytes:
        """Fallback to speech-like patterns if real TTS fails"""
        # Import the existing speech-like implementation
        from .tts import KokoroTTS as FallbackTTS
        
        fallback = FallbackTTS(voice=self.voice, language=self.language, speed=self.speed)
        return await fallback.synthesize_speech(text)
    
    async def synthesize_to_file(self, text: str, filepath: str, format: str = "wav"):
        """Synthesize speech and save to file"""
        audio_data = await self.synthesize_speech(text, format)
        
        with open(filepath, 'wb') as f:
            f.write(audio_data)
    
    def set_voice_parameters(self, voice: Optional[str] = None, language: Optional[str] = None, speed: Optional[float] = None):
        """Set voice parameters"""
        if voice is not None:
            self.voice = voice
        if language is not None:
            self.language = language
        if speed is not None:
            if not 0.5 <= speed <= 2.0:
                raise ValueError("Speed must be between 0.5 and 2.0")
            self.speed = speed
    
    def get_supported_formats(self):
        """Get supported audio formats"""
        return ["wav", "mp3"]
    
    def get_available_voices(self):
        """Get available voice profiles"""
        return ["af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]

# Alias for compatibility
KokoroTTS = RealKokoroTTS
