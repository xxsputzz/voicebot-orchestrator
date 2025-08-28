"""
Tests for Text-to-Speech (TTS) functionality.
"""
import asyncio
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebot_orchestrator.tts import KokoroTTS
import numpy as np


class TestKokoroTTS:
    """Test cases for KokoroTTS."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tts = KokoroTTS(voice="default", language="en", speed=1.0)
    
    async def test_synthesize_speech_simple(self):
        """Test synthesizing speech from simple text."""
        text = "Hello world"
        audio_data = await self.tts.synthesize_speech(text)
        
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0
    
    async def test_synthesize_speech_different_formats(self):
        """Test synthesizing speech in different formats."""
        text = "Test audio"
        
        # Test WAV format
        wav_data = await self.tts.synthesize_speech(text, "wav")
        assert isinstance(wav_data, bytes)
        assert len(wav_data) > 0
        assert wav_data.startswith(b'RIFF')  # WAV header
        
        # Test MP3 format
        mp3_data = await self.tts.synthesize_speech(text, "mp3")
        assert isinstance(mp3_data, bytes)
        assert len(mp3_data) > 0
    
    async def test_synthesize_speech_empty_text(self):
        """Test synthesizing speech from empty text raises error."""
        try:
            await self.tts.synthesize_speech("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "cannot be empty" in str(e)
    
    async def test_synthesize_speech_whitespace_text(self):
        """Test synthesizing speech from whitespace-only text raises error."""
        try:
            await self.tts.synthesize_speech("   \n\t   ")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "cannot be empty" in str(e)
    
    async def test_synthesize_speech_long_text(self):
        """Test synthesizing speech from very long text raises error."""
        long_text = "x" * 5001  # Exceeds limit
        try:
            await self.tts.synthesize_speech(long_text)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "too long" in str(e)
    
    async def test_synthesize_speech_unsupported_format(self):
        """Test synthesizing speech with unsupported format raises error."""
        try:
            await self.tts.synthesize_speech("Hello", "flac")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unsupported format" in str(e)
    
    async def test_synthesize_to_file(self):
        """Test synthesizing speech to file."""
        text = "Hello world"
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            await self.tts.synthesize_to_file(text, temp_path, "wav")
            
            # Check file exists and has content
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
            
            # Check file content
            with open(temp_path, 'rb') as f:
                content = f.read()
                assert content.startswith(b'RIFF')  # WAV header
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_supported_formats(self):
        """Test getting supported audio formats."""
        formats = self.tts.get_supported_formats()
        
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert "wav" in formats
        assert "mp3" in formats
    
    def test_get_available_voices(self):
        """Test getting available voice profiles."""
        voices = self.tts.get_available_voices()
        
        assert isinstance(voices, list)
        assert len(voices) > 0
        assert "default" in voices
        assert "male" in voices
        assert "female" in voices
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = self.tts.get_supported_languages()
        
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages
    
    async def test_validate_text_valid(self):
        """Test text validation with valid input."""
        valid_texts = [
            "Hello world",
            "What is your account balance?",
            "Thank you for your assistance.",
            "The quick brown fox jumps over the lazy dog.",
            "123-456-7890"
        ]
        
        for text in valid_texts:
            is_valid = await self.tts.validate_text(text)
            assert is_valid is True
    
    async def test_validate_text_invalid(self):
        """Test text validation with invalid input."""
        invalid_texts = [
            "",  # Empty
            "   ",  # Whitespace only
            "x" * 5001,  # Too long
            "Hello 世界",  # Non-ASCII characters
            "Text with emoji 😀",  # Emoji
        ]
        
        for text in invalid_texts:
            is_valid = await self.tts.validate_text(text)
            assert is_valid is False
    
    def test_set_voice_parameters(self):
        """Test setting voice parameters."""
        # Test setting voice
        self.tts.set_voice_parameters(voice="male")
        assert self.tts.voice == "male"
        
        # Test setting language
        self.tts.set_voice_parameters(language="es")
        assert self.tts.language == "es"
        
        # Test setting speed
        self.tts.set_voice_parameters(speed=1.5)
        assert self.tts.speed == 1.5
        
        # Test setting all parameters
        self.tts.set_voice_parameters(voice="female", language="fr", speed=0.8)
        assert self.tts.voice == "female"
        assert self.tts.language == "fr"
        assert self.tts.speed == 0.8
    
    def test_set_voice_parameters_invalid_speed(self):
        """Test setting invalid speed raises error."""
        try:
            self.tts.set_voice_parameters(speed=0.1)  # Too slow
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Speed must be between" in str(e)
        
        try:
            self.tts.set_voice_parameters(speed=3.0)  # Too fast
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Speed must be between" in str(e)
    
    def test_create_wav_bytes(self):
        """Test WAV byte creation."""
        # Create test audio data
        audio_data = np.random.randint(-32768, 32767, 1000, dtype=np.int16)
        sample_rate = 16000
        
        wav_bytes = self.tts._create_wav_bytes(audio_data, sample_rate)
        
        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > len(audio_data) * 2  # Should include header
        assert wav_bytes.startswith(b'RIFF')
        assert b'WAVE' in wav_bytes[:20]
    
    async def test_synthesize_different_text_lengths(self):
        """Test synthesis with different text lengths."""
        texts = [
            "Hi",  # Very short
            "Hello there",  # Short
            "This is a medium length sentence for testing.",  # Medium
            "This is a much longer sentence that should produce a longer audio output for comprehensive testing of the text-to-speech functionality."  # Long
        ]
        
        audio_lengths = []
        
        for text in texts:
            audio_data = await self.tts.synthesize_speech(text)
            audio_lengths.append(len(audio_data))
        
        # Longer texts should generally produce longer audio
        assert audio_lengths[0] < audio_lengths[-1]
    
    def test_engine_loading(self):
        """Test TTS engine loading."""
        # Engine should not be loaded initially
        assert self.tts._engine is None
        
        # Trigger engine loading
        self.tts._load_engine()
        
        # Engine should now be loaded (mock)
        assert self.tts._engine is not None
        assert isinstance(self.tts._engine, str)


# Test runner for pytest compatibility
async def run_tests():
    """Run all tests."""
    test_class = TestKokoroTTS()
    
    async_test_methods = [
        test_class.test_synthesize_speech_simple,
        test_class.test_synthesize_speech_different_formats,
        test_class.test_synthesize_speech_empty_text,
        test_class.test_synthesize_speech_whitespace_text,
        test_class.test_synthesize_speech_long_text,
        test_class.test_synthesize_speech_unsupported_format,
        test_class.test_synthesize_to_file,
        test_class.test_validate_text_valid,
        test_class.test_validate_text_invalid,
        test_class.test_synthesize_different_text_lengths,
    ]
    
    sync_test_methods = [
        test_class.test_get_supported_formats,
        test_class.test_get_available_voices,
        test_class.test_get_supported_languages,
        test_class.test_set_voice_parameters,
        test_class.test_set_voice_parameters_invalid_speed,
        test_class.test_create_wav_bytes,
        test_class.test_engine_loading,
    ]
    
    passed = 0
    failed = 0
    
    # Run async tests
    for test_method in async_test_methods:
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
    
    print(f"\nTTS Tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
