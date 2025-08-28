"""
Configuration settings for the voicebot orchestrator.
"""
from typing import Optional
import os
from pathlib import Path


class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self) -> None:
        """Initialize settings from environment variables."""
        # Whisper STT Configuration
        self.whisper_model: str = os.getenv("WHISPER_MODEL", "base")
        self.whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
        
        # Mistral LLM Configuration
        self.mistral_model_path: str = os.getenv(
            "MISTRAL_MODEL_PATH", 
            "./models/mistral-7b-instruct"
        )
        self.mistral_max_tokens: int = int(os.getenv("MISTRAL_MAX_TOKENS", "512"))
        self.mistral_temperature: float = float(os.getenv("MISTRAL_TEMPERATURE", "0.7"))
        
        # Kokoro TTS Configuration
        self.kokoro_voice: str = os.getenv("KOKORO_VOICE", "default")
        self.kokoro_language: str = os.getenv("KOKORO_LANGUAGE", "en")
        self.kokoro_speed: float = float(os.getenv("KOKORO_SPEED", "1.0"))
        
        # Server Configuration
        self.host: str = os.getenv("HOST", "localhost")
        self.port: int = int(os.getenv("PORT", "8000"))
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        
        # Session Configuration
        self.session_timeout: int = int(os.getenv("SESSION_TIMEOUT", "3600"))
        self.max_concurrent_sessions: int = int(os.getenv("MAX_CONCURRENT_SESSIONS", "10"))
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not os.path.exists(self.mistral_model_path):
            raise ValueError(f"Mistral model path does not exist: {self.mistral_model_path}")
        
        if self.mistral_max_tokens <= 0:
            raise ValueError("Mistral max tokens must be positive")
        
        if not 0.0 <= self.mistral_temperature <= 2.0:
            raise ValueError("Mistral temperature must be between 0.0 and 2.0")
        
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Port must be between 1 and 65535")


# Global settings instance
settings = Settings()
