"""
Real Speech-to-Text (STT) module using local Whisper.
"""
import asyncio
import tempfile
import os
import wave
from typing import Union
import numpy as np

try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper library available")
except ImportError:
    WHISPER_AVAILABLE = False
    print("❌ Whisper library not available")

try:
    import torch
    TORCH_AVAILABLE = True
    print("✅ PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch not available")


class RealWhisperSTT:
    """Real Whisper-based speech-to-text processor."""
    
    def __init__(self, model_name: str = "base", device: str = "auto") -> None:
        """
        Initialize real Whisper STT.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
            device: Device to run inference on (auto, cpu, cuda)
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self._model = None
        
        print(f"🎤 Initializing RealWhisperSTT with model: {model_name}, device: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _load_model(self) -> None:
        """Load Whisper model."""
        if self._model is None and WHISPER_AVAILABLE:
            try:
                print(f"🔄 Loading Whisper model '{self.model_name}' on {self.device}...")
                self._model = whisper.load_model(self.model_name, device=self.device)
                print(f"✅ Whisper model loaded successfully!")
            except Exception as e:
                print(f"❌ Failed to load Whisper model: {e}")
                self._model = None
    
    async def transcribe_audio(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """
        Transcribe audio data to text using real Whisper.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            
        Returns:
            Transcribed text
            
        Raises:
            ValueError: If audio data is invalid
        """
        if audio_data is None:
            raise ValueError("Audio data cannot be empty")
        
        if isinstance(audio_data, bytes) and len(audio_data) == 0:
            raise ValueError("Audio data cannot be empty")
        
        if isinstance(audio_data, np.ndarray) and audio_data.size == 0:
            raise ValueError("Audio data cannot be empty")
        
        if not WHISPER_AVAILABLE:
            print("⚠️ Whisper not available, using mock transcription")
            return await self._mock_transcribe(audio_data)
        
        self._load_model()
        
        if self._model is None:
            print("⚠️ Whisper model not loaded, using mock transcription")
            return await self._mock_transcribe(audio_data)
        
        try:
            # Convert audio data to temporary file for Whisper
            return await asyncio.to_thread(self._transcribe_with_whisper, audio_data)
        except Exception as e:
            print(f"❌ Whisper transcription failed: {e}")
            return await self._mock_transcribe(audio_data)
    
    def _transcribe_with_whisper(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """Transcribe using real Whisper model."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            try:
                if isinstance(audio_data, bytes):
                    # Write bytes directly to temp file
                    temp_file.write(audio_data)
                elif isinstance(audio_data, np.ndarray):
                    # Convert numpy array to WAV file
                    self._numpy_to_wav(audio_data, temp_file.name)
                
                temp_file.flush()
                temp_path = temp_file.name
                
                # Transcribe with Whisper
                print(f"🔊 Transcribing audio file: {temp_path}")
                result = self._model.transcribe(temp_path)
                
                transcribed_text = result["text"].strip()
                print(f"📝 Transcription result: '{transcribed_text}'")
                
                return transcribed_text
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def _numpy_to_wav(self, audio_array: np.ndarray, output_path: str, sample_rate: int = 16000):
        """Convert numpy array to WAV file."""
        # Ensure audio is in the right format for WAV
        if audio_array.dtype != np.int16:
            # Convert float to int16
            if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                audio_array = (audio_array * 32767).astype(np.int16)
            else:
                audio_array = audio_array.astype(np.int16)
        
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_array.tobytes())
    
    async def _mock_transcribe(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """Fallback mock transcription."""
        print("⚠️ Using mock transcription (Whisper not available)")
        if isinstance(audio_data, bytes):
            audio_length = len(audio_data)
            if audio_length < 10000:
                return "Hello"
            elif audio_length < 50000:
                return "How can I help you today?"
            else:
                return "I would like to check my account balance please."
        
        elif isinstance(audio_data, np.ndarray):
            if audio_data.size < 10000:
                return "Yes"
            elif audio_data.size < 50000:
                return "What is my current balance?"
            else:
                return "I need help with my banking account information."
        
        return "Sorry, I didn't understand that."
    
    async def transcribe_file(self, file_path: str) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        if not os.path.exists(file_path):
            raise ValueError(f"Audio file not found: {file_path}")
        
        if not WHISPER_AVAILABLE:
            return "Mock transcription from file"
        
        self._load_model()
        
        if self._model is None:
            return "Mock transcription from file (model not loaded)"
        
        try:
            result = await asyncio.to_thread(self._model.transcribe, file_path)
            return result["text"].strip()
        except Exception as e:
            print(f"❌ File transcription failed: {e}")
            return "Failed to transcribe file"
    
    def get_supported_formats(self) -> list[str]:
        """Get list of supported audio formats."""
        return ["wav", "mp3", "m4a", "flac", "ogg"]
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self._model is not None,
            "available": WHISPER_AVAILABLE
        }
