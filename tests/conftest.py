"""
Test configuration and fixtures.
"""
import pytest
import asyncio
import tempfile
import os
from pathlib import Path

# Set test environment variables
os.environ.update({
    "WHISPER_MODEL": "base",
    "WHISPER_DEVICE": "cpu",
    "MISTRAL_MODEL_PATH": "./tests/mock_models/mistral",
    "MISTRAL_MAX_TOKENS": "256",
    "MISTRAL_TEMPERATURE": "0.5",
    "KOKORO_VOICE": "test",
    "KOKORO_LANGUAGE": "en",
    "KOKORO_SPEED": "1.0",
    "HOST": "localhost",
    "PORT": "8001",
    "LOG_LEVEL": "DEBUG",
    "SESSION_TIMEOUT": "300",
    "MAX_CONCURRENT_SESSIONS": "5"
})


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_audio_file():
    """Create a temporary mock audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Write minimal WAV header
        f.write(b'RIFF')
        f.write((1000).to_bytes(4, 'little'))  # File size
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write((16).to_bytes(4, 'little'))    # Format chunk size
        f.write((1).to_bytes(2, 'little'))     # PCM format
        f.write((1).to_bytes(2, 'little'))     # Mono
        f.write((16000).to_bytes(4, 'little')) # Sample rate
        f.write((32000).to_bytes(4, 'little')) # Byte rate
        f.write((2).to_bytes(2, 'little'))     # Block align
        f.write((16).to_bytes(2, 'little'))    # Bits per sample
        f.write(b'data')
        f.write((500).to_bytes(4, 'little'))   # Data size
        f.write(b'\x00' * 500)                 # Silence data
        
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_model_directory():
    """Create a temporary mock model directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "mistral"
        model_path.mkdir()
        
        # Create mock model files
        (model_path / "config.json").write_text('{"model_type": "mistral"}')
        (model_path / "tokenizer.json").write_text('{}')
        
        yield str(model_path)


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Hello, how can I help you today?",
        "What is my account balance?",
        "I need help with my banking account.",
        "Thank you for your assistance.",
        ""  # Empty text for edge case testing
    ]


@pytest.fixture
def sample_audio_data():
    """Sample audio data for testing."""
    import numpy as np
    
    # Generate simple sine wave audio data
    sample_rate = 16000
    duration = 1.0  # 1 second
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_signal = np.sin(2 * np.pi * frequency * t) * 0.1
    audio_int16 = (audio_signal * 32767).astype(np.int16)
    
    return audio_int16.tobytes()
