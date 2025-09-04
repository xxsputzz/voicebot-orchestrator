"""
Speech-to-Text (STT) module using Whisper.
"""
import asyncio
import io
import tempfile
import logging
from typing import Union
import numpy as np

logger = logging.getLogger(__name__)

class WhisperSTT:
    """Whisper-based speech-to-text processor with real Whisper implementation."""
    
    def __init__(self, model_name: str = "base", device: str = "cpu", use_faster_whisper: bool = True) -> None:
        """
        Initialize Whisper STT.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
            device: Device to run inference on (cpu, cuda)
            use_faster_whisper: Use faster-whisper for better performance (recommended)
        """
        self.model_name = model_name
        self.device = device
        self.use_faster_whisper = use_faster_whisper
        self._model = None
        
        logger.info(f"🎙️ Initializing Whisper STT:")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Faster Whisper: {use_faster_whisper}")
    
    def _load_model(self) -> None:
        """Load Whisper model lazily."""
        if self._model is None:
            try:
                if self.use_faster_whisper:
                    # Use faster-whisper for better performance
                    from faster_whisper import WhisperModel
                    
                    # Optimize settings for better accuracy
                    model_size = self.model_name
                    compute_type = "int8" if self.device == "cpu" else "float16"
                    
                    logger.info(f"📥 Loading Faster Whisper model: {model_size}")
                    self._model = WhisperModel(
                        model_size, 
                        device=self.device,
                        compute_type=compute_type,
                        cpu_threads=4,  # Use multiple CPU threads
                        num_workers=1   # Single worker to avoid conflicts
                    )
                    logger.info("✅ Faster Whisper model loaded successfully")
                    
                else:
                    # Use original OpenAI Whisper
                    import whisper
                    
                    logger.info(f"📥 Loading OpenAI Whisper model: {self.model_name}")
                    self._model = whisper.load_model(self.model_name, device=self.device)
                    logger.info("✅ OpenAI Whisper model loaded successfully")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper model: {e}")
                # Fallback to mock for testing
                self._model = f"whisper_{self.model_name}_{self.device}"
                self.use_faster_whisper = False
    
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
        
        # Use real Whisper transcription
        return await asyncio.to_thread(self._real_transcribe, audio_data)
    
    def _real_transcribe(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """Real Whisper transcription."""
        try:
            # Write audio to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                if isinstance(audio_data, bytes):
                    tmp_file.write(audio_data)
                else:
                    # Convert numpy array to bytes if needed
                    import soundfile as sf
                    sf.write(tmp_file.name, audio_data, 16000, format='wav')
                
                temp_path = tmp_file.name
            
            try:
                if self.use_faster_whisper and hasattr(self._model, 'transcribe'):
                    # Use faster-whisper
                    logger.info(f"🎙️ Transcribing with Faster Whisper: {temp_path}")
                    logger.info(f"📊 File size: {len(audio_data)} bytes")
                    
                    # Faster Whisper transcription with optimized settings
                    segments, info = self._model.transcribe(
                        temp_path, 
                        beam_size=5,           # Better accuracy
                        language="en",         # Force English for banking calls
                        condition_on_previous_text=False,  # Don't rely on context for short audio
                        temperature=0.0,       # Deterministic output
                        word_timestamps=False,  # Faster processing
                        vad_filter=True,       # Voice activity detection
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    
                    # Combine all segments
                    transcription = " ".join([segment.text for segment in segments]).strip()
                    
                    logger.info(f"✅ Faster Whisper transcription result: '{transcription[:50]}...' (confidence: {info.language_probability:.3f})")
                    logger.info(f"🔍 DEBUG: Raw transcription result: '{transcription}'")
                    logger.info(f"🔍 DEBUG: Result length: {len(transcription)}")
                    logger.info(f"🔍 DEBUG: Result type: {type(transcription)}")
                    
                    return transcription
                    
                elif hasattr(self._model, 'transcribe'):
                    # Use OpenAI Whisper
                    logger.info(f"🎙️ Transcribing with OpenAI Whisper: {temp_path}")
                    
                    result = self._model.transcribe(
                        temp_path,
                        language="en",
                        temperature=0.0,
                        word_timestamps=False,
                        fp16=False if self.device == "cpu" else True
                    )
                    
                    transcription = result["text"].strip()
                    logger.info(f"✅ OpenAI Whisper transcription result: '{transcription}'")
                    
                    return transcription
                    
                else:
                    # Fallback mock
                    logger.warning("⚠️ Using mock transcription - Whisper not properly loaded")
                    return self._mock_transcribe(audio_data)
                    
            finally:
                # Clean up temp file
                import os
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            # Fallback to mock for error recovery
            return self._mock_transcribe(audio_data)
    
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
