"""
Audio Output Utilities
Centralized audio file management for all TTS engines
"""
import os
from datetime import datetime
from typing import Optional

class AudioOutputManager:
    """Manages centralized audio output for all TTS engines"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize audio output manager
        
        Args:
            base_dir: Base directory for audio files. If None, uses tests/audio_samples
        """
        if base_dir is None:
            # Use tests/audio_samples as default
            project_root = os.path.dirname(os.path.dirname(__file__))
            self.audio_dir = os.path.join(project_root, "tests", "audio_samples")
        else:
            self.audio_dir = base_dir
            
        # Ensure directory exists
        os.makedirs(self.audio_dir, exist_ok=True)
        
    def get_audio_path(self, filename: str, engine: str = "tts") -> str:
        """Get full path for audio file
        
        Args:
            filename: Name of the audio file
            engine: TTS engine name (kokoro, nari_dia, etc.)
            
        Returns:
            Full path to audio file in centralized directory
        """
        # Add engine prefix if not already present
        if not filename.startswith(engine):
            base_name = os.path.splitext(filename)[0]
            extension = os.path.splitext(filename)[1] or ".wav"
            filename = f"{engine}_{base_name}{extension}"
            
        return os.path.join(self.audio_dir, filename)
    
    def get_timestamped_path(self, prefix: str, engine: str = "tts") -> str:
        """Get timestamped audio path
        
        Args:
            prefix: Prefix for the filename (e.g., "output", "demo", "test")
            engine: TTS engine name
            
        Returns:
            Full path with timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{engine}_{prefix}_{timestamp}.wav"
        return os.path.join(self.audio_dir, filename)
    
    def get_engine_path(self, engine: str, description: str = "output") -> str:
        """Get path for specific engine output
        
        Args:
            engine: Engine name (kokoro, nari_dia)
            description: Description of the audio (output, demo, test, etc.)
            
        Returns:
            Full path for engine-specific audio file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{engine}_{description}_{timestamp}.wav"
        return os.path.join(self.audio_dir, filename)
    
    def cleanup_old_files(self, max_files: int = 50, pattern: str = "*.wav"):
        """Clean up old audio files to prevent directory bloat
        
        Args:
            max_files: Maximum number of files to keep
            pattern: File pattern to match for cleanup
        """
        import glob
        
        pattern_path = os.path.join(self.audio_dir, pattern)
        files = glob.glob(pattern_path)
        
        if len(files) > max_files:
            # Sort by modification time (oldest first)
            files.sort(key=os.path.getmtime)
            
            # Remove oldest files
            for file_path in files[:-max_files]:
                try:
                    os.remove(file_path)
                    print(f"🗑️ Cleaned up old audio file: {os.path.basename(file_path)}")
                except OSError:
                    pass
    
    def list_audio_files(self, engine: Optional[str] = None) -> list:
        """List audio files in the output directory
        
        Args:
            engine: Filter by engine name (optional)
            
        Returns:
            List of audio file paths
        """
        import glob
        
        if engine:
            pattern = os.path.join(self.audio_dir, f"{engine}_*.wav")
        else:
            pattern = os.path.join(self.audio_dir, "*.wav")
            
        return glob.glob(pattern)
    
    def get_latest_file(self, engine: Optional[str] = None) -> Optional[str]:
        """Get the most recently created audio file
        
        Args:
            engine: Filter by engine name (optional)
            
        Returns:
            Path to latest audio file or None
        """
        files = self.list_audio_files(engine)
        if not files:
            return None
            
        # Sort by creation time (newest first)
        files.sort(key=os.path.getctime, reverse=True)
        return files[0]

# Global instance for easy access
audio_manager = AudioOutputManager()

def get_audio_output_path(filename: str, engine: str = "tts") -> str:
    """Convenience function to get audio output path
    
    Args:
        filename: Name of the audio file
        engine: TTS engine name
        
    Returns:
        Full path to audio file in centralized directory
    """
    return audio_manager.get_audio_path(filename, engine)

def get_timestamped_audio_path(prefix: str, engine: str = "tts") -> str:
    """Convenience function to get timestamped audio path
    
    Args:
        prefix: Prefix for the filename
        engine: TTS engine name
        
    Returns:
        Full path with timestamp
    """
    return audio_manager.get_timestamped_path(prefix, engine)
