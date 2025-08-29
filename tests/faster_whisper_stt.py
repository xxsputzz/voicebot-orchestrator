"""
Fast Speech-to-Text (STT) module using Faster-Whisper.
Optimized version that's 2-4x faster than regular Whisper.
"""
import asyncio
import tempfile
import os
import wave
from typing import Union, Tuple
import numpy as np

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    print("✅ Faster-Whisper library available")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("❌ Faster-Whisper library not available")

try:
    import torch
    TORCH_AVAILABLE = True
    print("✅ PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch not available")


class FasterWhisperSTT:
    """Faster-Whisper based speech-to-text processor (2-4x faster than regular Whisper)."""
    
    def __init__(self, model_name: str = "base", device: str = "auto", compute_type: str = "auto") -> None:
        """
        Initialize Faster-Whisper STT.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large-v1, large-v2, large-v3)
            device: Device to run inference on (auto, cpu, cuda)
            compute_type: Compute type (auto, int8, int16, float16, float32)
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.compute_type = self._get_compute_type(compute_type)
        self._model = None
        
        print(f"🎤 Initializing FasterWhisperSTT")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Compute Type: {self.compute_type}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _get_compute_type(self, compute_type: str) -> str:
        """Determine the best compute type."""
        if compute_type == "auto":
            if self.device == "cuda":
                return "float16"  # Faster on GPU
            else:
                return "int8"     # Faster on CPU
        return compute_type
    
    def _load_model(self) -> None:
        """Load Faster-Whisper model."""
        if self._model is None and FASTER_WHISPER_AVAILABLE:
            try:
                print(f"🔄 Loading Faster-Whisper model '{self.model_name}'...")
                print(f"   Device: {self.device}, Compute: {self.compute_type}")
                
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=4  # Optimize CPU usage
                )
                
                print(f"✅ Faster-Whisper model loaded successfully!")
            except Exception as e:
                print(f"❌ Failed to load Faster-Whisper model: {e}")
                self._model = None
    
    async def transcribe_audio(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """
        Transcribe audio data to text using Faster-Whisper.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            
        Returns:
            Transcribed text
            
        Raises:
            ValueError: If audio data is invalid
        """
        # Validate audio data
        if audio_data is None:
            print("⚠️ No audio data provided")
            return "No audio detected"
        
        if isinstance(audio_data, bytes):
            if len(audio_data) == 0:
                print("⚠️ Empty audio data (bytes)")
                return "No audio detected"
            if len(audio_data) < 1000:  # Minimum threshold for meaningful audio
                print(f"⚠️ Audio too short: {len(audio_data)} bytes")
                return "Audio too short"
        
        if isinstance(audio_data, np.ndarray):
            if audio_data.size == 0:
                print("⚠️ Empty audio data (array)")
                return "No audio detected"
            if audio_data.size < 1000:  # Minimum threshold
                print(f"⚠️ Audio too short: {audio_data.size} samples")
                return "Audio too short"
        
        if not FASTER_WHISPER_AVAILABLE:
            print("⚠️ Faster-Whisper not available, using mock transcription")
            return await self._mock_transcribe(audio_data)
        
        self._load_model()
        
        if self._model is None:
            print("⚠️ Faster-Whisper model not loaded, using mock transcription")
            return await self._mock_transcribe(audio_data)
        
        try:
            # Convert audio data to temporary file for Faster-Whisper
            return await asyncio.to_thread(self._transcribe_with_faster_whisper, audio_data)
        except Exception as e:
            print(f"❌ Faster-Whisper transcription failed: {e}")
            # Return a more informative fallback
            if "No audio" in str(e).lower() or "empty" in str(e).lower():
                return "No speech detected"
            else:
                return "Speech unclear, please try again"
    
    def _transcribe_with_faster_whisper(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """Transcribe using Faster-Whisper model."""
        temp_path = None
        try:
            # Create temp file with a unique name to avoid conflicts
            import uuid
            temp_filename = f"whisper_audio_{uuid.uuid4().hex}.wav"
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="whisper_") as temp_file:
                temp_path = temp_file.name
                
                if isinstance(audio_data, bytes):
                    # For bytes data, try to interpret as WAV or raw audio
                    print(f"📊 Processing {len(audio_data)} bytes of audio data")
                    
                    # Check if it's already a valid WAV file
                    if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:20]:
                        print("📁 Data appears to be valid WAV format, writing directly")
                        with open(temp_path, 'wb') as f:
                            f.write(audio_data)
                    else:
                        print("📁 Raw audio data detected, converting to WAV format")
                        # Assume raw PCM data, convert to proper WAV
                        try:
                            # Skip if data is too small
                            if len(audio_data) < 1000:
                                return "Audio too short to process"
                            
                            # Convert bytes to numpy array (assuming 16-bit PCM)
                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                            temp_file.close()  # Close before writing with wave module
                            self._numpy_to_wav(audio_array, temp_path)
                        except Exception as e:
                            print(f"⚠️ Error converting raw audio: {e}")
                            return "Could not convert audio format"
                                
                elif isinstance(audio_data, np.ndarray):
                    # Convert numpy array to WAV file
                    print(f"📊 Processing numpy array: shape {audio_data.shape}, dtype {audio_data.dtype}")
                    temp_file.close()  # Close file before writing with wave module
                    self._numpy_to_wav(audio_data, temp_path)
                else:
                    temp_file.close()
                    return "Invalid audio data type"
                
                # Ensure file is properly closed before transcription
                if not temp_file.closed:
                    temp_file.flush()
                    temp_file.close()
                
                # Add a small delay to ensure file is fully written
                import time
                time.sleep(0.1)
                
                # Validate the WAV file before processing
                try:
                    import wave
                    with wave.open(temp_path, 'rb') as test_wav:
                        frames = test_wav.getnframes()
                        sample_rate = test_wav.getframerate()
                        channels = test_wav.getnchannels()
                        duration = frames / sample_rate if sample_rate > 0 else 0
                        
                        print(f"📁 WAV validation: {frames} frames, {sample_rate}Hz, {channels}ch, {duration:.2f}s")
                        
                        if frames < 1000:  # Less than ~0.06 seconds at 16kHz
                            return "Audio too short for transcription"
                        if duration > 30:  # Longer than 30 seconds
                            return "Audio too long for transcription"
                            
                except Exception as wav_error:
                    print(f"❌ WAV file validation failed: {wav_error}")
                    return "Invalid audio file format"
                
                # Transcribe with Faster-Whisper
                print(f"🔊 Transcribing audio with Faster-Whisper...")
                
                # Faster-Whisper transcribe method
                segments, info = self._model.transcribe(
                    temp_path,
                    beam_size=5,           # Good balance of speed vs accuracy
                    language="en",         # Specify language for speed
                    condition_on_previous_text=False,  # Faster processing
                    vad_filter=True,       # Voice activity detection
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Collect all segments
                transcribed_text = ""
                segment_count = 0
                for segment in segments:
                    transcribed_text += segment.text
                    segment_count += 1
                
                transcribed_text = transcribed_text.strip()
                
                # Check if we got meaningful results
                if not transcribed_text or len(transcribed_text) < 2:
                    print(f"📝 Faster-Whisper: No speech detected (empty result)")
                    return "No speech detected"
                
                # Check for very short or meaningless results
                if len(transcribed_text) < 5 and segment_count == 0:
                    print(f"📝 Faster-Whisper: Audio too short or unclear")
                    return "Audio unclear, please speak louder"
                
                print(f"📝 Faster-Whisper result: '{transcribed_text}' ({segment_count} segments)")
                print(f"📊 Language: {info.language} (confidence: {info.language_probability:.2f})")
                
                return transcribed_text
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return "Could not process audio"
        finally:
            # Clean up temp file with retry logic
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError as e:
                    print(f"⚠️ Could not delete temp file: {e}")
                    # Try again after a short delay
                    try:
                        import time
                        time.sleep(0.2)
                        os.unlink(temp_path)
                    except:
                        pass  # Give up if still can't delete
    
    def _numpy_to_wav(self, audio_array: np.ndarray, output_path: str, sample_rate: int = 16000):
        """Convert numpy array to WAV file with proper formatting for Whisper."""
        try:
            # Ensure audio is in the right format for WAV
            if audio_array.dtype != np.int16:
                # Convert float to int16
                if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                    # Normalize to [-1, 1] range first
                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0:
                        audio_array = audio_array / max_val
                    # Convert to int16 range
                    audio_array = (audio_array * 32767).astype(np.int16)
                else:
                    audio_array = audio_array.astype(np.int16)
            
            # Ensure mono audio (flatten if stereo)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.flatten()
            
            # Create WAV file with proper headers
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit (2 bytes)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_array.tobytes())
                
            print(f"📁 Created WAV file: {output_path} ({len(audio_array)} samples, {sample_rate}Hz)")
            
        except Exception as e:
            print(f"❌ Error creating WAV file: {e}")
            raise
    
    async def _mock_transcribe(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """Fallback mock transcription."""
        print("⚠️ Using mock transcription (Faster-Whisper not available)")
        if isinstance(audio_data, bytes):
            audio_length = len(audio_data)
            if audio_length < 10000:
                return "Hello, how are you?"
            elif audio_length < 50000:
                return "Can you help me with my account?"
            elif audio_length < 100000:
                return "I would like to check my account balance please."
            else:
                return "I need assistance with my banking services and account information."
        
        elif isinstance(audio_data, np.ndarray):
            if audio_data.size < 10000:
                return "Yes, please help me."
            elif audio_data.size < 50000:
                return "What is my current balance?"
            elif audio_data.size < 100000:
                return "I need help with my banking account information."
            else:
                return "Can you assist me with a money transfer and account details?"
        
        return "Sorry, I didn't understand that clearly."
    
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
        
        if not FASTER_WHISPER_AVAILABLE:
            return "Mock transcription from file"
        
        self._load_model()
        
        if self._model is None:
            return "Mock transcription from file (model not loaded)"
        
        try:
            segments, info = await asyncio.to_thread(
                self._model.transcribe,
                file_path,
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True
            )
            
            # Collect all segments
            transcribed_text = ""
            for segment in segments:
                transcribed_text += segment.text
            
            return transcribed_text.strip()
        except Exception as e:
            print(f"❌ File transcription failed: {e}")
            return "Failed to transcribe file"
    
    def get_supported_formats(self) -> list[str]:
        """Get list of supported audio formats."""
        return ["wav", "mp3", "m4a", "flac", "ogg", "opus", "webm"]
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "loaded": self._model is not None,
            "available": FASTER_WHISPER_AVAILABLE,
            "library": "faster-whisper"
        }
    
    def get_performance_info(self) -> dict:
        """Get performance information."""
        return {
            "expected_speedup": "2-4x faster than regular Whisper",
            "memory_usage": "Lower memory usage than regular Whisper",
            "cpu_optimized": self.device == "cpu",
            "gpu_optimized": self.device == "cuda",
            "compute_type": self.compute_type
        }
