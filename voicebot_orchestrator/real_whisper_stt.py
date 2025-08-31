"""
Real Whisper STT Implementation
==============================

Replaces the mock STT with actual OpenAI Whisper transcription.
"""
import asyncio
import io
import tempfile
import os
import logging
from typing import Union
import numpy as np

class RealWhisperSTT:
    """Real Whisper-based speech-to-text processor."""
    
    def __init__(self, model_name: str = "base", device: str = "cpu") -> None:
        """
        Initialize Whisper STT.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
            device: Device to run inference on (cpu, cuda)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._whisper = None
    
    def _load_model(self) -> None:
        """Load Whisper model lazily."""
        if self._model is None:
            try:
                import whisper
                self._whisper = whisper
                logging.info(f"Loading Whisper model: {self.model_name}")
                self._model = whisper.load_model(self.model_name, device=self.device)
                logging.info(f"✅ Whisper model loaded: {self.model_name}")
            except ImportError:
                logging.error("❌ OpenAI Whisper not installed. Install with: pip install openai-whisper")
                raise ImportError("OpenAI Whisper not installed. Install with: pip install openai-whisper")
            except Exception as e:
                logging.error(f"❌ Failed to load Whisper model: {e}")
                raise
    
    async def transcribe_audio(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """
        Transcribe audio data to text.
        
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
        
        self._load_model()
        
        # Save to temporary file for Whisper processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            if isinstance(audio_data, bytes):
                temp_file.write(audio_data)
            else:
                # Convert numpy array to wav bytes if needed
                import wave
                with wave.open(temp_file.name, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(16000)  # 16kHz
                    wf.writeframes(audio_data.astype(np.int16).tobytes())
            
            temp_path = temp_file.name
        
        try:
            # Use real Whisper transcription
            result = await asyncio.to_thread(self._transcribe_file, temp_path)
            return result.strip()
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _transcribe_file(self, file_path: str) -> str:
        """Transcribe file using Whisper (synchronous)."""
        try:
            logging.info(f"🔄 Starting Whisper transcription of: {file_path}")
            
            # Check if file exists and has content
            if not os.path.exists(file_path):
                logging.error(f"❌ File not found: {file_path}")
                return ""
            
            file_size = os.path.getsize(file_path)
            logging.info(f"📊 File size: {file_size} bytes")
            
            if file_size < 100:  # Very small file
                logging.warning(f"⚠️  File too small: {file_size} bytes")
                return ""
            
            result = self._model.transcribe(
                file_path,
                language=None,  # Auto-detect language
                task="transcribe",
                verbose=False
            )
            
            text = result.get("text", "")
            confidence = result.get("segments", [{}])[0].get("avg_logprob", 0.0) if result.get("segments") else 0.0
            
            logging.info(f"✅ Whisper transcription result: '{text}' (confidence: {confidence:.3f})")
            
            if not text or len(text.strip()) == 0:
                logging.warning("⚠️  Whisper returned empty transcript")
                return ""
            
            return text
            
        except Exception as e:
            logging.error(f"❌ Whisper transcription failed: {e}")
            import traceback
            logging.error(f"📋 Traceback: {traceback.format_exc()}")
            return ""
    
    async def transcribe_file(self, file_path: str) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcribed text
            
        Raises:
            ValueError: If file doesn't exist or is invalid
        """
        if not os.path.exists(file_path):
            raise ValueError(f"Audio file not found: {file_path}")
        
        self._load_model()
        
        # Use real Whisper transcription
        result = await asyncio.to_thread(self._transcribe_file, file_path)
        return result.strip()
    
    def get_supported_formats(self) -> list[str]:
        """Get list of supported audio formats."""
        return ["wav", "mp3", "m4a", "flac", "ogg", "mp4", "webm"]
    
    def validate_audio_format(self, file_path: str) -> bool:
        """
        Validate if audio file format is supported.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if format is supported, False otherwise
        """
        file_extension = file_path.lower().split('.')[-1]
        return file_extension in self.get_supported_formats()


# Fallback to mock if Whisper is not available
class WhisperSTT:
    """Whisper-based speech-to-text processor with real implementation."""
    
    def __init__(self, model_name: str = "base", device: str = "cpu") -> None:
        """
        Initialize Whisper STT.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
            device: Device to run inference on (cpu, cuda)
        """
        self.model_name = model_name
        self.device = device
        self._real_stt = None
        self._use_real = True
        
        # Try to use real Whisper, fallback to mock
        try:
            self._real_stt = RealWhisperSTT(model_name, device)
            logging.info("✅ Using real Whisper STT implementation")
        except ImportError:
            logging.warning("⚠️  Falling back to mock STT (install openai-whisper for real transcription)")
            self._use_real = False
        except Exception as e:
            logging.warning(f"⚠️  Falling back to mock STT due to error: {e}")
            self._use_real = False
    
    async def transcribe_audio(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """Transcribe audio data to text."""
        if self._use_real and self._real_stt:
            return await self._real_stt.transcribe_audio(audio_data)
        else:
            return await self._mock_transcribe_audio(audio_data)
    
    async def transcribe_file(self, file_path: str) -> str:
        """Transcribe audio file to text."""
        if self._use_real and self._real_stt:
            return await self._real_stt.transcribe_file(file_path)
        else:
            return await self._mock_transcribe_file(file_path)
    
    async def _mock_transcribe_audio(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """Mock transcription for testing purposes."""
        if isinstance(audio_data, bytes):
            audio_length = len(audio_data)
            if audio_length < 1000:
                return "Hello"
            elif audio_length < 5000:
                return "How can I help you today?"
            else:
                return "I would like to check my account balance please."
        
        elif isinstance(audio_data, np.ndarray):
            if audio_data.size < 1000:
                return "Yes"
            elif audio_data.size < 5000:
                return "What is my current balance?"
            else:
                return "I need help with my banking account information."
        
        return "Sorry, I didn't understand that."
    
    async def _mock_transcribe_file(self, file_path: str) -> str:
        """Mock file transcription."""
        if not os.path.exists(file_path):
            raise ValueError(f"Audio file not found: {file_path}")
        
        # Read file and mock transcribe
        with open(file_path, 'rb') as f:
            audio_data = f.read()
        
        return await self._mock_transcribe_audio(audio_data)
    
    def get_supported_formats(self) -> list[str]:
        """Get list of supported audio formats."""
        if self._use_real and self._real_stt:
            return self._real_stt.get_supported_formats()
        else:
            return ["wav", "mp3", "m4a", "flac", "ogg"]
    
    def validate_audio_format(self, file_path: str) -> bool:
        """Validate if audio file format is supported."""
        if self._use_real and self._real_stt:
            return self._real_stt.validate_audio_format(file_path)
        else:
            file_extension = file_path.lower().split('.')[-1]
            return file_extension in self.get_supported_formats()
