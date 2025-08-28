"""
Tests for Speech-to-Text (STT) functionality.
"""
import asyncio
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebot_orchestrator.stt import WhisperSTT
import numpy as np


class TestWhisperSTT:
    """Test cases for WhisperSTT."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stt = WhisperSTT(model_name="base", device="cpu")
    
    async def test_transcribe_audio_bytes(self):
        """Test transcribing audio from bytes."""
        # Small audio data
        audio_data = b"small audio data"
        result = await self.stt.transcribe_audio(audio_data)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == "Hello"  # Expected mock result for small data
    
    async def test_transcribe_audio_numpy(self):
        """Test transcribing audio from numpy array."""
        # Create numpy audio data
        audio_data = np.random.random(2000).astype(np.float32)
        result = await self.stt.transcribe_audio(audio_data)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    async def test_transcribe_empty_audio(self):
        """Test transcribing empty audio raises error."""
        try:
            await self.stt.transcribe_audio(b"")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "cannot be empty" in str(e)
    
    async def test_transcribe_none_audio(self):
        """Test transcribing None audio raises error."""
        try:
            await self.stt.transcribe_audio(None)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "cannot be empty" in str(e)
    
    async def test_transcribe_file(self):
        """Test transcribing audio file."""
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"mock audio file content")
            temp_path = f.name
        
        try:
            result = await self.stt.transcribe_file(temp_path)
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            os.unlink(temp_path)
    
    async def test_transcribe_nonexistent_file(self):
        """Test transcribing non-existent file raises error."""
        try:
            await self.stt.transcribe_file("nonexistent_file.wav")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found" in str(e)
    
    def test_get_supported_formats(self):
        """Test getting supported audio formats."""
        formats = self.stt.get_supported_formats()
        
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert "wav" in formats
        assert "mp3" in formats
    
    def test_validate_audio_format(self):
        """Test audio format validation."""
        # Valid formats
        assert self.stt.validate_audio_format("test.wav") is True
        assert self.stt.validate_audio_format("test.mp3") is True
        assert self.stt.validate_audio_format("TEST.WAV") is True  # Case insensitive
        
        # Invalid formats
        assert self.stt.validate_audio_format("test.txt") is False
        assert self.stt.validate_audio_format("test.jpg") is False
        assert self.stt.validate_audio_format("test") is False
    
    def test_model_loading(self):
        """Test model loading functionality."""
        # Model should not be loaded initially
        assert self.stt._model is None
        
        # Trigger model loading
        self.stt._load_model()
        
        # Model should now be loaded (mock)
        assert self.stt._model is not None
        assert isinstance(self.stt._model, str)
    
    async def test_different_audio_sizes(self):
        """Test transcription with different audio sizes."""
        # Test with different sizes to trigger different mock responses
        small_audio = b"x" * 500
        medium_audio = b"x" * 2000
        large_audio = b"x" * 10000
        
        small_result = await self.stt.transcribe_audio(small_audio)
        medium_result = await self.stt.transcribe_audio(medium_audio)
        large_result = await self.stt.transcribe_audio(large_audio)
        
        # All should return valid strings
        assert isinstance(small_result, str)
        assert isinstance(medium_result, str)
        assert isinstance(large_result, str)
        
        # Results should be different based on size
        assert small_result == "Hello"
        assert medium_result == "How can I help you today?"
        assert large_result == "I would like to check my account balance please."


# Test runner for pytest compatibility
async def run_tests():
    """Run all tests."""
    test_class = TestWhisperSTT()
    
    test_methods = [
        test_class.test_transcribe_audio_bytes,
        test_class.test_transcribe_audio_numpy,
        test_class.test_transcribe_empty_audio,
        test_class.test_transcribe_none_audio,
        test_class.test_transcribe_file,
        test_class.test_transcribe_nonexistent_file,
        test_class.test_different_audio_sizes,
    ]
    
    sync_test_methods = [
        test_class.test_get_supported_formats,
        test_class.test_validate_audio_format,
        test_class.test_model_loading,
    ]
    
    passed = 0
    failed = 0
    
    # Run async tests
    for test_method in test_methods:
        test_class.setup_method()
        try:
            await test_method()
            print(f"✓ {test_method.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_method.__name__}: {e}")
            failed += 1
    
    # Run sync tests
    for test_method in sync_test_methods:
        test_class.setup_method()
        try:
            test_method()
            print(f"✓ {test_method.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_method.__name__}: {e}")
            failed += 1
    
    print(f"\nSTT Tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
