"""
Speech-to-Text (STT) module using Whisper.
"""
import asyncio
import io
import tempfile
from typing import Union
import numpy as np


class WhisperSTT:
    """Whisper-based speech-to-text processor."""
    
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
    
    def _load_model(self) -> None:
        """Load Whisper model lazily."""
        if self._model is None:
            # Mock model loading - in real implementation would use:
            # import whisper
            # self._model = whisper.load_model(self.model_name, device=self.device)
            self._model = f"whisper_{self.model_name}_{self.device}"
    
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
        
        # Mock transcription - in real implementation would use:
        # result = self._model.transcribe(audio_path)
        # return result["text"].strip()
        
        # For now, return a mock transcription
        return await asyncio.to_thread(self._mock_transcribe, audio_data)
    
    def _mock_transcribe(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """Mock transcription for testing purposes."""
        if isinstance(audio_data, bytes):
            # Simulate processing audio bytes
            audio_length = len(audio_data)
            if audio_length < 1000:
                return "Hello"
            elif audio_length < 5000:
                return "How can I help you today?"
            else:
                return "I would like to check my account balance please."
        
        elif isinstance(audio_data, np.ndarray):
            # Simulate processing numpy array
            if audio_data.size < 1000:
                return "Yes"
            elif audio_data.size < 5000:
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
            
        Raises:
            ValueError: If file doesn't exist or is invalid
        """
        import os
        
        if not os.path.exists(file_path):
            raise ValueError(f"Audio file not found: {file_path}")
        
        # Read file and transcribe
        with open(file_path, 'rb') as f:
            audio_data = f.read()
        
        return await self.transcribe_audio(audio_data)
    
    def get_supported_formats(self) -> list[str]:
        """Get list of supported audio formats."""
        return ["wav", "mp3", "m4a", "flac", "ogg"]
    
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
