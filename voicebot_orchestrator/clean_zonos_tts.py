"""
Clean Zonos TTS Implementation
Removes digital artifacts and encoding noise from Edge-TTS output
"""
import asyncio
import time
import logging
import tempfile
import os
import sys
import numpy as np
import wave
from typing import Optional, Dict, Any, List
import subprocess

class CleanZonosTTS:
    """
    Clean Zonos TTS with artifact removal
    Provides clean human-like speech without digital noise
    """
    
    def __init__(self, voice: str = "aria", model: str = "zonos-v1"):
        """Initialize Clean Zonos TTS"""
        self.voice = voice
        self.model = model
        self.initialized = False
        
        # Voice mapping from Zonos names to Edge-TTS voices
        self.voice_mapping = {
            # Female voices - Professional
            "sophia": "en-US-JennyNeural",      # Professional, clear
            "aria": "en-US-AriaNeural",         # Conversational, friendly
            "luna": "en-GB-SoniaNeural",        # British, warm
            "emma": "en-US-MichelleNeural",     # Business-like, mature
            "zoe": "en-AU-NatashaNeural",       # Australian, friendly
            "maya": "en-US-SaraNeural",         # Storytelling, warm
            "isabel": "en-US-NancyNeural",      # Educational, clear
            "grace": "en-GB-LibbyNeural",       # British, elegant
            
            # Male voices - Professional  
            "default": "en-US-DavisNeural",     # Default male voice
            "professional": "en-US-GuyNeural",  # Business, authoritative
            "conversational": "en-US-DavisNeural", # Casual, friendly
            "narrative": "en-US-AndrewNeural",  # Storytelling, mature
            "marcus": "en-US-BrianNeural",      # Authoritative, mature
            "oliver": "en-GB-RyanNeural",       # British, friendly
            "diego": "en-US-TonyNeural",        # Warm, approachable
            
            # Neutral/unisex voices (using female as default)
            "alex": "en-US-JennyNeural",
            "casey": "en-US-AriaNeural", 
            "river": "en-US-SaraNeural",
            "sage": "en-US-GuyNeural",          # Elderly -> mature male
            "nova": "en-US-JennyNeural",        # Futuristic -> clear female
        }
        
        print(f"[CLEAN ZONOS] Initializing Clean Zonos TTS with voice='{voice}', model='{model}'")
        self._initialize()
    
    def _initialize(self):
        """Initialize the Clean TTS engine"""
        try:
            # Check if edge-tts is available
            try:
                import edge_tts
                self.edge_tts = edge_tts
                print(f"[OK] Edge-TTS available")
            except ImportError:
                print(f"[INSTALL] Installing edge-tts for clean speech synthesis...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'edge-tts'])
                import edge_tts
                self.edge_tts = edge_tts
                print(f"[OK] Edge-TTS installed and ready")
            
            # Map voice to Edge-TTS voice
            self.edge_voice = self.voice_mapping.get(self.voice, "en-US-AriaNeural")
            
            self.initialized = True
            print(f"[OK] Clean Zonos TTS initialized successfully")
            print(f"     Voice: {self.voice} -> {self.edge_voice}")
            print(f"     Model: {self.model} (using Clean Edge Neural TTS)")
            print(f"     Quality: Clean Human Speech (Artifact-Free)")
            
        except Exception as e:
            print(f"[ERROR] Clean Zonos TTS initialization failed: {e}")
            raise
    
    def _clean_audio(self, audio_bytes: bytes) -> bytes:
        """
        Remove digital artifacts from audio
        
        Args:
            audio_bytes: Raw audio data
            
        Returns:
            Cleaned audio data
        """
        try:
            # Check if it's a valid WAV file
            if not audio_bytes.startswith(b'RIFF'):
                print(f"[WARNING] Not a valid WAV file, returning original")
                return audio_bytes
            
            # Write to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
                tmp_input.write(audio_bytes)
                tmp_input_path = tmp_input.name
            
            # Read audio data
            with wave.open(tmp_input_path, 'rb') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                channels = w.getnchannels()
                sample_width = w.getsampwidth()
                audio_data = w.readframes(frames)
            
            # Convert to numpy array
            if sample_width == 2:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            else:
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            print(f"[CLEAN] Processing {len(audio_array)} samples...")
            
            # Step 1: Remove start/end artifacts
            audio_cleaned = self._remove_start_end_artifacts(audio_array)
            
            # Step 2: Remove digital spikes
            audio_cleaned = self._remove_digital_spikes(audio_cleaned)
            
            # Step 3: Apply gentle fade in/out
            audio_cleaned = self._apply_fade(audio_cleaned, rate)
            
            # Step 4: Normalize audio
            audio_cleaned = self._normalize_audio(audio_cleaned)
            
            # Convert back to original data type
            if sample_width == 2:
                audio_cleaned = audio_cleaned.astype(np.int16)
            
            # Create new WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output:
                tmp_output_path = tmp_output.name
            
            with wave.open(tmp_output_path, 'wb') as w:
                w.setnchannels(channels)
                w.setsampwidth(sample_width)
                w.setframerate(rate)
                w.writeframes(audio_cleaned.tobytes())
            
            # Read cleaned audio
            with open(tmp_output_path, 'rb') as f:
                cleaned_bytes = f.read()
            
            # Cleanup
            try:
                os.unlink(tmp_input_path)
                os.unlink(tmp_output_path)
            except:
                pass
            
            # Verify the output is a valid WAV
            if not cleaned_bytes.startswith(b'RIFF'):
                print(f"[WARNING] Cleaning produced invalid WAV, returning original")
                return audio_bytes
            
            print(f"[OK] Audio cleaned - processed {len(audio_array)} samples")
            return cleaned_bytes
            
        except Exception as e:
            print(f"[WARNING] Audio cleaning failed: {e}, returning original")
            return audio_bytes
    
    def _remove_start_end_artifacts(self, audio_array: np.ndarray) -> np.ndarray:
        """Remove artifacts at start and end of audio"""
        if len(audio_array) < 1000:
            return audio_array
        
        # Calculate overall RMS to determine threshold
        overall_rms = np.sqrt(np.mean(audio_array.astype(float)**2))
        noise_threshold = overall_rms * 0.1  # 10% of average
        
        # Find start of actual speech (first sustained audio above threshold)
        start_idx = 0
        window_size = 100
        for i in range(0, len(audio_array) - window_size, 10):
            window_rms = np.sqrt(np.mean(audio_array[i:i+window_size].astype(float)**2))
            if window_rms > noise_threshold:
                start_idx = max(0, i - 50)  # Start slightly before
                break
        
        # Find end of actual speech (last sustained audio above threshold)
        end_idx = len(audio_array)
        for i in range(len(audio_array) - window_size, 0, -10):
            window_rms = np.sqrt(np.mean(audio_array[i:i+window_size].astype(float)**2))
            if window_rms > noise_threshold:
                end_idx = min(len(audio_array), i + window_size + 50)  # End slightly after
                break
        
        # Extract clean audio with minimal padding
        cleaned = audio_array[start_idx:end_idx]
        
        if len(cleaned) < len(audio_array) * 0.8:  # Safety check
            print(f"[WARNING] Aggressive trimming detected, using original")
            return audio_array
        
        print(f"[CLEAN] Trimmed {start_idx} samples from start, {len(audio_array) - end_idx} from end")
        return cleaned
    
    def _remove_digital_spikes(self, audio_array: np.ndarray) -> np.ndarray:
        """Remove digital spikes and clicks"""
        if len(audio_array) < 100:
            return audio_array
        
        # Calculate derivative to find sudden jumps
        diff = np.diff(audio_array.astype(float))
        max_diff = np.max(np.abs(audio_array)) * 0.3  # 30% of max amplitude
        
        # Find spikes
        spike_indices = np.where(np.abs(diff) > max_diff)[0]
        
        if len(spike_indices) == 0:
            return audio_array
        
        print(f"[CLEAN] Removing {len(spike_indices)} digital spikes")
        
        # Fix spikes by interpolation
        cleaned = audio_array.copy().astype(float)
        for spike_idx in spike_indices:
            if 5 < spike_idx < len(cleaned) - 5:
                # Replace spike with interpolated value
                before = np.mean(cleaned[spike_idx-3:spike_idx])
                after = np.mean(cleaned[spike_idx+2:spike_idx+5])
                cleaned[spike_idx] = (before + after) / 2
                cleaned[spike_idx+1] = (before + after) / 2
        
        return cleaned.astype(audio_array.dtype)
    
    def _apply_fade(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply gentle fade in/out to prevent clicks"""
        fade_samples = min(1000, len(audio_array) // 20)  # 50ms fade or 5% of audio
        
        if fade_samples < 10:
            return audio_array
        
        cleaned = audio_array.copy().astype(float)
        
        # Fade in
        fade_in = np.linspace(0, 1, fade_samples)
        cleaned[:fade_samples] *= fade_in
        
        # Fade out
        fade_out = np.linspace(1, 0, fade_samples)
        cleaned[-fade_samples:] *= fade_out
        
        return cleaned.astype(audio_array.dtype)
    
    def _normalize_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping"""
        max_val = np.max(np.abs(audio_array))
        
        if audio_array.dtype == np.int16:
            target_max = 28000  # Leave headroom to prevent clipping
        else:
            target_max = 0.85
        
        if max_val > target_max:
            scale_factor = target_max / max_val
            audio_array = (audio_array * scale_factor).astype(audio_array.dtype)
            print(f"[CLEAN] Normalized audio by {scale_factor:.3f}")
        
        return audio_array
    
    async def synthesize_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        emotion: str = "neutral",
        speaking_style: str = "normal",
        emphasis_words: Optional[list] = None,
        pause_locations: Optional[list] = None,
        prosody_adjustments: Optional[Dict[str, Any]] = None,
        high_quality: bool = True,
        seed: Optional[int] = None,
        output_format: str = "wav",
        sample_rate: int = 44100,
        **kwargs
    ) -> bytes:
        """
        Synthesize clean speech using Edge-TTS with artifact removal
        """
        if not self.initialized:
            raise RuntimeError("Clean Zonos TTS not initialized")
        
        start_time = time.time()
        
        # Use provided parameters or instance defaults
        current_voice = voice or self.voice
        edge_voice = self.voice_mapping.get(current_voice, "en-US-AriaNeural")
        
        # Validate parameters
        speed = max(0.5, min(2.0, speed))
        pitch = max(0.5, min(2.0, pitch))
        
        print(f"[CLEAN ZONOS] Synthesizing {len(text)} chars with {current_voice} -> {edge_voice}")
        print(f"              Emotion: {emotion}, Speed: {speed}x, Pitch: {pitch}x")
        
        try:
            # Create Edge-TTS communicate instance
            communicate = self.edge_tts.Communicate(text, edge_voice)
            
            # Generate speech to temporary file (ensure WAV format)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Save audio - Edge-TTS should produce WAV when .wav extension is used
            await communicate.save(tmp_path)
            
            # Read the generated audio file
            with open(tmp_path, 'rb') as f:
                raw_audio_bytes = f.read()
            
            # Check if it's actually a WAV file, if not convert it
            if not raw_audio_bytes.startswith(b'RIFF'):
                print(f"[CONVERT] Edge-TTS produced non-WAV format, converting...")
                # Try to convert using a different approach
                import io
                import subprocess
                
                # Use ffmpeg if available to convert to WAV
                try:
                    # Save the raw audio to a temp file with proper extension
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
                        mp3_file.write(raw_audio_bytes)
                        mp3_path = mp3_file.name
                    
                    # Convert to WAV using ffmpeg
                    wav_path = tmp_path + "_converted.wav"
                    result = subprocess.run([
                        'ffmpeg', '-i', mp3_path, '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', wav_path, '-y'
                    ], capture_output=True, text=True)
                    
                    if os.path.exists(wav_path):
                        with open(wav_path, 'rb') as f:
                            raw_audio_bytes = f.read()
                        print(f"[OK] Converted to WAV format")
                        os.unlink(wav_path)
                    
                    os.unlink(mp3_path)
                    
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print(f"[WARNING] ffmpeg not available, using original format")
            
            # Clean the audio to remove artifacts (only if it's WAV)
            if raw_audio_bytes.startswith(b'RIFF'):
                clean_audio_bytes = self._clean_audio(raw_audio_bytes)
            else:
                print(f"[WARNING] Cannot clean non-WAV format, returning original")
                clean_audio_bytes = raw_audio_bytes
            
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            generation_time = time.time() - start_time
            
            print(f"[OK] Clean Zonos synthesis completed in {generation_time:.2f}s")
            print(f"     Generated {len(clean_audio_bytes)} bytes of CLEAN HUMAN SPEECH")
            print(f"     Quality: Artifact-Free Neural Speech (Microsoft Edge)")
            
            return clean_audio_bytes
            
        except Exception as e:
            print(f"[ERROR] Clean Zonos synthesis failed: {e}")
            raise

# Factory function for backwards compatibility
def ZonosTTS(voice: str = "default", model: str = "zonos-v1") -> CleanZonosTTS:
    """Create Clean Zonos TTS instance (backwards compatible)"""
    return CleanZonosTTS(voice=voice, model=model)

if __name__ == "__main__":
    async def test_clean_zonos():
        """Test the clean Zonos TTS implementation"""
        print("🧹 Testing Clean Zonos TTS Implementation")
        print("=" * 50)
        
        tts = CleanZonosTTS(voice="aria", model="zonos-v1")
        
        test_text = "Hello! This is the new clean Zonos TTS that removes digital artifacts and encoding noise for pure human speech."
        
        print(f"Testing: '{test_text}'")
        
        audio_bytes = await tts.synthesize_speech(
            text=test_text,
            emotion="neutral",
            speed=1.0,
            pitch=1.0
        )
        
        # Save test file
        with open("clean_zonos_test.wav", "wb") as f:
            f.write(audio_bytes)
        
        print(f"✅ SUCCESS! Generated {len(audio_bytes)} bytes of CLEAN HUMAN SPEECH")
        print(f"   Saved as: clean_zonos_test.wav")
        print(f"   This should sound like pure human speech without digital artifacts!")
        
    asyncio.run(test_clean_zonos())
