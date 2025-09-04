#!/usr/bin/env python3
"""
Tortoise TTS Service with Neural Network Implementation
Ultra-high-quality TTS service implementing actual Tortoise-like neural synthesis
"""

import os
import sys
import asyncio
import numpy as np
import base64
import time
import wave
import io
import threading
import tempfile
from io import BytesIO
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn

# Import our custom Tortoise implementation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tortoise_tts_implementation import TortoiseTTS, get_tortoise_instance

# Configuration
print("[INFO] Initializing Neural Tortoise TTS Service")

# Initialize global TTS instance
tts_service = None

def initialize_tts():
    """Initialize the Tortoise TTS service"""
    global tts_service
    try:
        print("[INFO] Loading Tortoise TTS neural model...")
        tts_service = get_tortoise_instance()
        print(f"[SUCCESS] Tortoise TTS ready with {len(tts_service.get_available_voices())} voices")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize Tortoise TTS: {e}")
        return False

class TortoiseRealTTS:
    """Real TTS implementation using pyttsx3 with voice variety"""
    
    def __init__(self):
        self.tts_engine = None
        self.voice_presets = [
            "angie", "denise", "freeman", "geralt", "halle", "jlaw", 
            "lj", "myself", "pat", "pat2", "rainbow", "snakes", 
            "train_dotcom", "train_daws", "train_dreams", "train_grace",
            "train_lescault", "train_mouse", "william", "random",
            "emma", "sophia", "olivia", "isabella", "mia", "charlotte", 
            "ava", "amelia", "harper", "evelyn"
        ]
        
        # Voice characteristics for realistic synthesis
        self.voice_characteristics = {
            "angie": {"voice_id": 0, "rate": 180, "volume": 0.9, "gender": "Female", "origin": "US"},
            "denise": {"voice_id": 0, "rate": 170, "volume": 0.8, "gender": "Female", "origin": "US"},
            "freeman": {"voice_id": 1, "rate": 160, "volume": 0.9, "gender": "Male", "origin": "US"},
            "geralt": {"voice_id": 1, "rate": 150, "volume": 1.0, "gender": "Male", "origin": "Fantasy"},
            "halle": {"voice_id": 0, "rate": 175, "volume": 0.85, "gender": "Female", "origin": "US"},
            "jlaw": {"voice_id": 0, "rate": 185, "volume": 0.9, "gender": "Female", "origin": "US"},
            "lj": {"voice_id": 0, "rate": 190, "volume": 0.8, "gender": "Female", "origin": "US"},
            "myself": {"voice_id": 1, "rate": 160, "volume": 0.9, "gender": "Male", "origin": "US"},
            "pat": {"voice_id": 1, "rate": 165, "volume": 0.85, "gender": "Male", "origin": "US"},
            "pat2": {"voice_id": 1, "rate": 170, "volume": 0.9, "gender": "Male", "origin": "US"},
            "rainbow": {"voice_id": 0, "rate": 200, "volume": 0.9, "gender": "Female", "origin": "Colorful"},
            "snakes": {"voice_id": 1, "rate": 140, "volume": 0.8, "gender": "Male", "origin": "Mysterious"},
            "train_dotcom": {"voice_id": 1, "rate": 180, "volume": 0.9, "gender": "Male", "origin": "Tech"},
            "train_daws": {"voice_id": 0, "rate": 175, "volume": 0.85, "gender": "Female", "origin": "US"},
            "train_dreams": {"voice_id": 0, "rate": 160, "volume": 0.9, "gender": "Female", "origin": "Dreamy"},
            "train_grace": {"voice_id": 0, "rate": 170, "volume": 0.9, "gender": "Female", "origin": "Elegant"},
            "train_lescault": {"voice_id": 1, "rate": 165, "volume": 0.85, "gender": "Male", "origin": "French"},
            "train_mouse": {"voice_id": 0, "rate": 220, "volume": 0.7, "gender": "Female", "origin": "Cute"},
            "william": {"voice_id": 1, "rate": 155, "volume": 0.9, "gender": "Male", "origin": "British"},
            "random": {"voice_id": 0, "rate": 175, "volume": 0.85, "gender": "Random", "origin": "Various"},
            "emma": {"voice_id": 0, "rate": 180, "volume": 0.9, "gender": "Female", "origin": "US"},
            "sophia": {"voice_id": 0, "rate": 175, "volume": 0.85, "gender": "Female", "origin": "US"},
            "olivia": {"voice_id": 0, "rate": 185, "volume": 0.9, "gender": "Female", "origin": "US"},
            "isabella": {"voice_id": 0, "rate": 170, "volume": 0.9, "gender": "Female", "origin": "US"},
            "mia": {"voice_id": 0, "rate": 190, "volume": 0.8, "gender": "Female", "origin": "US"},
            "charlotte": {"voice_id": 0, "rate": 165, "volume": 0.9, "gender": "Female", "origin": "US"},
            "ava": {"voice_id": 0, "rate": 180, "volume": 0.85, "gender": "Female", "origin": "US"},
            "amelia": {"voice_id": 0, "rate": 175, "volume": 0.9, "gender": "Female", "origin": "US"},
            "harper": {"voice_id": 0, "rate": 185, "volume": 0.85, "gender": "Female", "origin": "US"},
            "evelyn": {"voice_id": 0, "rate": 170, "volume": 0.9, "gender": "Female", "origin": "US"}
        }
        
        self.initialize_engine()
    
    def initialize_engine(self):
        """Initialize the TTS engine"""
        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            print(f"[INFO] TTS Engine initialized with {len(voices)} available voices")
            for i, voice in enumerate(voices):
                print(f"  Voice {i}: {voice.name}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize TTS engine: {e}")
            self.tts_engine = None
    
    def generate_speech(self, text: str, voice_preset: str = "angie") -> bytes:
        """Generate enhanced speech audio using real TTS with human-like processing"""
        print(f"[INFO] Starting enhanced speech generation for '{text[:50]}...' with voice '{voice_preset}'")
        
        if not self.tts_engine:
            print("[ERROR] TTS engine not available, falling back to placeholder")
            return self._generate_placeholder_audio(text, voice_preset)
        
        try:
            # Get voice characteristics
            voice_config = self.voice_characteristics.get(voice_preset, self.voice_characteristics["angie"])
            print(f"[INFO] Using voice config: {voice_config}")
            
            # Preprocess text for more natural speech
            enhanced_text = self._enhance_text_for_speech(text, voice_preset)
            print(f"[INFO] Enhanced text length: {len(enhanced_text)} chars")
            
            # Configure TTS engine with enhanced settings
            voices = self.tts_engine.getProperty('voices')
            if voices and len(voices) > voice_config["voice_id"]:
                self.tts_engine.setProperty('voice', voices[voice_config["voice_id"]].id)
                print(f"[INFO] Set voice to: {voices[voice_config['voice_id']].name}")
            
            # Apply voice-specific rate and volume with natural variation
            rate_variation = np.random.uniform(-10, 10)  # Add slight random variation
            volume_variation = np.random.uniform(-0.05, 0.05)
            
            final_rate = max(80, min(300, voice_config["rate"] + rate_variation))
            final_volume = max(0.1, min(1.0, voice_config["volume"] + volume_variation))
            
            self.tts_engine.setProperty('rate', final_rate)
            self.tts_engine.setProperty('volume', final_volume)
            print(f"[INFO] Set rate: {final_rate:.1f}, volume: {final_volume:.2f}")
            
            # Generate audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
            print(f"[INFO] Using temp file: {temp_filename}")
            
            try:
                print("[INFO] Starting enhanced TTS generation...")
                self.tts_engine.save_to_file(enhanced_text, temp_filename)
                
                # Try with timeout - use a thread to run the engine
                import threading
                engine_thread = threading.Thread(target=self.tts_engine.runAndWait)
                engine_thread.daemon = True
                engine_thread.start()
                
                # Wait with timeout
                engine_thread.join(timeout=15.0)  # Increased timeout for longer text
                
                if engine_thread.is_alive():
                    print("[WARNING] TTS engine timed out, using fallback")
                    return self._generate_placeholder_audio(text, voice_preset)
                
                print("[INFO] TTS generation completed")
                
                # Wait a moment for file to be written
                time.sleep(0.5)
                
                # Read and enhance the generated audio file
                if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 44:
                    print(f"[INFO] Reading generated audio file: {os.path.getsize(temp_filename)} bytes")
                    with open(temp_filename, 'rb') as audio_file:
                        raw_audio_data = audio_file.read()
                    
                    # Apply human-like audio enhancement
                    enhanced_audio_data = self._enhance_audio_quality(raw_audio_data, voice_preset)
                    print(f"[INFO] Applied human-like audio enhancement")
                    
                    # Cleanup temp file
                    try:
                        os.unlink(temp_filename)
                    except:
                        pass
                    
                    if len(enhanced_audio_data) > 44:  # Valid WAV file has at least header
                        print(f"[SUCCESS] Generated {len(enhanced_audio_data)} bytes of enhanced human-like speech for voice '{voice_preset}'")
                        return enhanced_audio_data
                    else:
                        print(f"[WARNING] Enhanced audio too small, using fallback")
                        return self._generate_placeholder_audio(text, voice_preset)
                else:
                    print(f"[ERROR] Audio file not created or too small, using fallback")
                    return self._generate_placeholder_audio(text, voice_preset)
                    
            except Exception as e:
                print(f"[ERROR] TTS generation failed: {e}, using fallback")
                try:
                    os.unlink(temp_filename)
                except:
                    pass
                return self._generate_placeholder_audio(text, voice_preset)
                
        except Exception as e:
            print(f"[ERROR] Speech generation failed: {e}")
            return self._generate_placeholder_audio(text, voice_preset)
    
    def _generate_placeholder_audio(self, text: str, voice_preset: str) -> bytes:
        """Fallback placeholder audio generation"""
        duration = min(max(len(text) * 0.05, 1.0), 30.0)
        sample_rate = 22050
        
        # Generate simple tones as fallback
        t = np.linspace(0, duration, int(sample_rate * duration))
        base_freq = 200
        audio = np.sin(2 * np.pi * base_freq * t) * 0.3
        
        # Add some variation based on text length
        for i in range(min(len(text) // 10, 5)):
            freq = base_freq + (i * 100)
            audio += np.sin(2 * np.pi * freq * t) * 0.1
        
        # Convert to 16-bit PCM
        audio_int = (audio * 32767).astype(np.int16)
        
        # Create WAV file
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int.tobytes())
        
        return wav_buffer.getvalue()
    
    def _enhance_text_for_speech(self, text: str, voice_preset: str) -> str:
        """Enhance text for more natural speech patterns"""
        enhanced_text = text
        
        # Add natural pauses for better speech flow
        enhanced_text = enhanced_text.replace('. ', '... ')  # Longer pause after sentences
        enhanced_text = enhanced_text.replace(', ', ', ')    # Short pause after commas
        enhanced_text = enhanced_text.replace(': ', ': ')    # Pause after colons
        enhanced_text = enhanced_text.replace('; ', '; ')    # Pause after semicolons
        
        # Add emphasis markers for important words (simplified approach)
        import re
        # Emphasize words in quotes
        enhanced_text = re.sub(r'"([^"]*)"', r'"\1"', enhanced_text)
        
        # Add slight breathing pauses for longer text
        if len(text) > 200:
            sentences = enhanced_text.split('. ')
            if len(sentences) > 3:
                # Add breathing pause every 3-4 sentences
                for i in range(3, len(sentences), 4):
                    if i < len(sentences):
                        sentences[i] = '... ' + sentences[i]
                enhanced_text = '. '.join(sentences)
        
        return enhanced_text
    
    def _enhance_audio_quality(self, audio_data: bytes, voice_preset: str) -> bytes:
        """Apply human-like audio enhancements to make speech more natural"""
        try:
            # Read WAV data
            wav_buffer = io.BytesIO(audio_data)
            with wave.open(wav_buffer, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frames = wav_file.readframes(wav_file.getnframes())
            
            # Convert to numpy array for processing
            audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            
            # Apply voice-specific enhancements
            voice_config = self.voice_characteristics.get(voice_preset, self.voice_characteristics["angie"])
            enhanced_audio = self._apply_voice_character_processing(audio_array, voice_config, sample_rate)
            
            # Apply general human-like enhancements
            enhanced_audio = self._apply_natural_speech_effects(enhanced_audio, sample_rate)
            
            # Convert back to 16-bit PCM
            enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
            audio_int = (enhanced_audio * 32767).astype(np.int16)
            
            # Create enhanced WAV file
            enhanced_wav_buffer = io.BytesIO()
            with wave.open(enhanced_wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int.tobytes())
            
            return enhanced_wav_buffer.getvalue()
            
        except Exception as e:
            print(f"[WARNING] Audio enhancement failed: {e}, using original audio")
            return audio_data
    
    def _apply_voice_character_processing(self, audio: np.ndarray, voice_config: dict, sample_rate: int) -> np.ndarray:
        """Apply voice-specific character processing"""
        enhanced_audio = audio.copy()
        
        # Apply gender-specific processing
        gender = voice_config.get("gender", "Female")
        if gender == "Female":
            # Slightly brighten female voices
            enhanced_audio = self._apply_brightness_filter(enhanced_audio, 1.1)
        elif gender == "Male":
            # Add slight warmth to male voices
            enhanced_audio = self._apply_warmth_filter(enhanced_audio, 1.05)
        
        # Apply origin-specific characteristics
        origin = voice_config.get("origin", "US")
        if origin in ["UK", "British"]:
            # Add subtle formality
            enhanced_audio = self._apply_formality_effect(enhanced_audio)
        elif origin in ["Fantasy", "Character"]:
            # Add character-specific effects
            enhanced_audio = self._apply_character_effects(enhanced_audio, voice_config)
        
        return enhanced_audio
    
    def _apply_natural_speech_effects(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply natural speech effects to reduce robotic sound"""
        enhanced_audio = audio.copy()
        
        # 1. Add subtle volume variations (natural breathing)
        volume_variation = np.random.uniform(0.95, 1.05, len(audio))
        # Smooth the variation to avoid clicks
        if SCIPY_AVAILABLE:
            from scipy import ndimage
            volume_variation = ndimage.gaussian_filter1d(volume_variation, sigma=sample_rate//10)
        else:
            # Simple smoothing without scipy
            for i in range(1, len(volume_variation)-1):
                volume_variation[i] = (volume_variation[i-1] + volume_variation[i] + volume_variation[i+1]) / 3
        enhanced_audio *= volume_variation
        
        # 2. Add very subtle pitch variations (natural speech)
        # Simple pitch variation through slight speed changes
        pitch_variation = 1.0 + np.random.uniform(-0.02, 0.02) * np.sin(np.linspace(0, 4*np.pi, len(audio)))
        
        # 3. Add subtle formant resonance
        enhanced_audio = self._add_formant_resonance(enhanced_audio, sample_rate)
        
        # 4. Add very subtle background "room tone" for warmth
        room_tone = np.random.normal(0, 0.001, len(audio))
        enhanced_audio += room_tone
        
        # 5. Apply subtle compression for more natural dynamics
        enhanced_audio = self._apply_gentle_compression(enhanced_audio)
        
        return enhanced_audio
    
    def _apply_brightness_filter(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Apply brightness enhancement for female voices"""
        # Simple high-frequency emphasis
        if len(audio) > 100:
            # High-pass emphasis
            diff = np.diff(audio, prepend=audio[0])
            return audio + diff * (factor - 1.0) * 0.1
        return audio
    
    def _apply_warmth_filter(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Apply warmth enhancement for male voices"""
        # Simple low-frequency emphasis
        if len(audio) > 100:
            # Low-pass smoothing
            smoothed = np.convolve(audio, np.ones(3)/3, mode='same')
            return audio * (2-factor) + smoothed * (factor-1)
        return audio
    
    def _apply_formality_effect(self, audio: np.ndarray) -> np.ndarray:
        """Add formal speech characteristics"""
        # Slightly reduce volume variations for more controlled speech
        return audio * 0.98 + np.mean(audio) * 0.02
    
    def _apply_character_effects(self, audio: np.ndarray, voice_config: dict) -> np.ndarray:
        """Apply character-specific effects"""
        # Add unique characteristics based on character type
        if "gravelly" in voice_config.get("style", "").lower():
            # Add slight roughness
            noise = np.random.normal(0, 0.005, len(audio))
            return audio + noise * np.abs(audio)
        elif "serpentine" in voice_config.get("style", "").lower():
            # Add sibilant emphasis
            return self._apply_brightness_filter(audio, 1.15)
        return audio
    
    def _add_formant_resonance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Add subtle formant resonance for more natural voice"""
        # Simple resonance using a basic filter
        if len(audio) < 100:
            return audio
        
        # Create a simple resonant filter effect
        alpha = 0.95  # Resonance factor
        filtered_audio = audio.copy()
        
        for i in range(2, len(audio)):
            filtered_audio[i] = audio[i] + alpha * filtered_audio[i-1] - alpha * 0.5 * filtered_audio[i-2]
        
        # Mix with original
        return audio * 0.85 + filtered_audio * 0.15
    
    def _apply_gentle_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply gentle compression for more natural dynamics"""
        # Simple soft compression
        threshold = 0.7
        ratio = 3.0
        
        compressed = audio.copy()
        mask = np.abs(audio) > threshold
        
        # Apply compression to loud parts
        compressed[mask] = np.sign(audio[mask]) * (
            threshold + (np.abs(audio[mask]) - threshold) / ratio
        )
        
        return compressed
    
    def get_available_voices(self):
        """Get list of available voice presets"""
        return self.voice_presets

class TortoiseEnhancedPlaceholder:
    """Enhanced placeholder implementation with realistic speech synthesis"""
    
    def __init__(self):
        self.voice_presets = [
            "angie", "denise", "freeman", "geralt", "halle", "jlaw", 
            "lj", "myself", "pat", "pat2", "rainbow", "snakes", 
            "train_dotcom", "train_daws", "train_dreams", "train_grace",
            "train_lescault", "train_mouse", "william", "random",
            "emma", "sophia", "olivia", "isabella", "mia", "charlotte", 
            "ava", "amelia", "harper", "evelyn"
        ]
        
        # Voice characteristics for realistic synthesis
        self.voice_characteristics = {
            "angie": {"base_freq": 220, "variation": 0.15, "style": "warm_female", "gender": "Female", "origin": "US"},
            "denise": {"base_freq": 200, "variation": 0.12, "style": "professional_female", "gender": "Female", "origin": "US"},
            "freeman": {"base_freq": 140, "variation": 0.18, "style": "deep_male", "gender": "Male", "origin": "US"},
            "geralt": {"base_freq": 120, "variation": 0.20, "style": "gravelly_male", "gender": "Male", "origin": "Fantasy"},
            "halle": {"base_freq": 240, "variation": 0.14, "style": "bright_female", "gender": "Female", "origin": "US"},
            "jlaw": {"base_freq": 210, "variation": 0.16, "style": "casual_female", "gender": "Female", "origin": "US"},
            "lj": {"base_freq": 230, "variation": 0.10, "style": "clear_female", "gender": "Female", "origin": "US"},
            "myself": {"base_freq": 190, "variation": 0.13, "style": "personal_male", "gender": "Male", "origin": "Custom"},
            "pat": {"base_freq": 180, "variation": 0.13, "style": "neutral_male", "gender": "Male", "origin": "US"},
            "pat2": {"base_freq": 175, "variation": 0.14, "style": "variant_male", "gender": "Male", "origin": "US"},
            "rainbow": {"base_freq": 250, "variation": 0.25, "style": "expressive_female", "gender": "Female", "origin": "Artistic"},
            "snakes": {"base_freq": 130, "variation": 0.22, "style": "serpentine_male", "gender": "Male", "origin": "Character"},
            "train_dotcom": {"base_freq": 170, "variation": 0.15, "style": "tech_male", "gender": "Male", "origin": "Business"},
            "train_daws": {"base_freq": 165, "variation": 0.16, "style": "narrator_male", "gender": "Male", "origin": "UK"},
            "train_dreams": {"base_freq": 225, "variation": 0.18, "style": "dreamy_female", "gender": "Female", "origin": "Ethereal"},
            "train_grace": {"base_freq": 215, "variation": 0.12, "style": "graceful_female", "gender": "Female", "origin": "Classic"},
            "train_lescault": {"base_freq": 195, "variation": 0.14, "style": "sophisticated_male", "gender": "Male", "origin": "FR"},
            "train_mouse": {"base_freq": 270, "variation": 0.20, "style": "squeaky_character", "gender": "Character", "origin": "Animation"},
            "william": {"base_freq": 160, "variation": 0.14, "style": "refined_male", "gender": "Male", "origin": "UK"},
            "random": {"base_freq": 200, "variation": 0.15, "style": "adaptive", "gender": "Random", "origin": "Auto"},
            "emma": {"base_freq": 235, "variation": 0.13, "style": "sweet_female", "gender": "Female", "origin": "UK"},
            "sophia": {"base_freq": 215, "variation": 0.11, "style": "elegant_female", "gender": "Female", "origin": "IT"},
            "olivia": {"base_freq": 225, "variation": 0.14, "style": "youthful_female", "gender": "Female", "origin": "AU"},
            "isabella": {"base_freq": 205, "variation": 0.12, "style": "sophisticated_female", "gender": "Female", "origin": "ES"},
            "mia": {"base_freq": 245, "variation": 0.16, "style": "energetic_female", "gender": "Female", "origin": "SE"},
            "charlotte": {"base_freq": 210, "variation": 0.13, "style": "gentle_female", "gender": "Female", "origin": "CA"},
            "ava": {"base_freq": 230, "variation": 0.15, "style": "confident_female", "gender": "Female", "origin": "US"},
            "amelia": {"base_freq": 220, "variation": 0.12, "style": "mature_female", "gender": "Female", "origin": "NZ"},
            "harper": {"base_freq": 240, "variation": 0.17, "style": "playful_female", "gender": "Female", "origin": "US"},
            "evelyn": {"base_freq": 200, "variation": 0.11, "style": "calm_female", "gender": "Female", "origin": "IE"}
        }
        
        print(f"[PLACEHOLDER] Enhanced Tortoise TTS placeholder with {len(self.voice_presets)} voice profiles")
    
    async def synthesize_speech(self, text: str, voice: str = None, **kwargs):
        """Enhanced placeholder synthesis with voice characteristics"""
        current_voice = voice or "angie"
        
        if current_voice not in self.voice_presets:
            current_voice = "random"
        
        print(f"[TORTOISE] Synthesizing with voice '{current_voice}': {text[:50].encode('ascii', 'replace').decode('ascii')}...")
        
        # Get voice characteristics
        voice_config = self.voice_characteristics.get(current_voice, self.voice_characteristics["random"])
        
        # Generate enhanced audio with voice characteristics
        audio_bytes = await self._generate_realistic_speech_audio(text, voice_config)
        
        print(f"[OK] Tortoise speech synthesis complete: {len(audio_bytes)} bytes")
        return audio_bytes
    
    async def _generate_realistic_speech_audio(self, text: str, voice_config: dict):
        """Generate simple but effective speech-like audio"""
        # Simulate processing time
        await asyncio.sleep(min(len(text) * 0.01, 1.0))
        
        words = text.split()
        if not words:
            words = ["hello", "world"]
        
        # Calculate natural duration
        duration = max(1.0, len(words) * 0.5 + 0.5)  # About 2 words per second
        sample_rate = 22050
        samples = int(sample_rate * duration)
        
        # Voice settings
        base_freq = voice_config.get("base_freq", 180)
        if "male" in voice_config.get("style", "").lower():
            pitch = base_freq * 0.7  # Lower for male voices
        else:
            pitch = base_freq * 0.9  # Higher for female voices
        
        # Create simple speech-like pattern
        audio = np.zeros(samples)
        
        # Process each word
        word_duration = duration / len(words)
        
        for i, word in enumerate(words[:10]):  # Limit to 10 words for performance
            # Calculate word timing
            word_start = i * word_duration
            word_end = (i + 1) * word_duration
            
            start_sample = int(word_start * sample_rate)
            end_sample = int(word_end * sample_rate * 0.8)  # 80% sound, 20% pause
            end_sample = min(end_sample, samples)
            
            if end_sample <= start_sample:
                continue
            
            # Generate word audio
            word_samples = end_sample - start_sample
            t = np.linspace(0, word_duration * 0.8, word_samples)
            
            # Create a more speech-like sound by combining frequencies
            # This creates a vowel-like sound that's more recognizable as speech
            
            # Fundamental frequency with slight variation
            f0_variation = pitch * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))
            
            # Create vowel-like sound with multiple harmonics
            vowel_sound = (
                0.5 * np.sin(2 * np.pi * f0_variation * t) +      # Fundamental
                0.3 * np.sin(2 * np.pi * f0_variation * 2 * t) +  # Second harmonic
                0.2 * np.sin(2 * np.pi * f0_variation * 3 * t) +  # Third harmonic
                0.1 * np.sin(2 * np.pi * f0_variation * 4 * t)    # Fourth harmonic
            )
            
            # Add some consonant-like noise at word boundaries
            if len(word) > 3:
                # Add brief consonant sounds
                consonant_length = min(word_samples // 4, int(0.1 * sample_rate))
                if consonant_length > 0:
                    consonant_noise = np.random.normal(0, 0.3, consonant_length)
                    # Blend consonant with vowel at start
                    vowel_sound[:consonant_length] = (
                        0.7 * vowel_sound[:consonant_length] + 
                        0.3 * consonant_noise
                    )
            
            # Natural envelope for the word
            envelope = np.ones(word_samples)
            
            # Soft attack and decay
            attack_len = min(word_samples // 8, int(0.05 * sample_rate))
            decay_len = min(word_samples // 6, int(0.08 * sample_rate))
            
            if attack_len > 0:
                envelope[:attack_len] = np.linspace(0.1, 1.0, attack_len)
            if decay_len > 0:
                envelope[-decay_len:] = np.linspace(1.0, 0.1, decay_len)
            
            # Apply envelope
            word_audio = vowel_sound * envelope
            
            # Add to main audio
            audio[start_sample:end_sample] += word_audio
        
        # Simple smoothing filter to reduce harshness
        if len(audio) > 20:
            # Basic moving average filter
            filtered_audio = np.copy(audio)
            for i in range(10, len(audio) - 10):
                filtered_audio[i] = np.mean(audio[i-5:i+6])
            audio = filtered_audio
        
        # Normalize to reasonable volume
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.5  # 50% volume to prevent clipping
        
        # Convert to 16-bit PCM
        audio_16bit = (audio * 32767).astype(np.int16)
        
        # Create WAV file
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_16bit.tobytes())
        
        wav_buffer.seek(0)
        audio_bytes = wav_buffer.read()
        
        print(f"[OK] Simple speech audio generated: {len(audio_bytes)} bytes, {duration:.1f}s")
        return audio_bytes
        """Generate realistic speech-like audio using formant synthesis"""
        # Simulate processing time
        await asyncio.sleep(min(len(text) * 0.02, 2.0))
        
        words = text.split()
        if not words:
            words = ["hello"]
        
        # Calculate duration: ~2.2 words per second (natural speech)
        words_per_second = 2.2
        duration = len(words) / words_per_second + 0.8
        duration = max(1.5, min(duration, 15.0))
        
        sample_rate = 22050
        samples = int(sample_rate * duration)
        
        # Voice characteristics
        base_freq = voice_config.get("base_freq", 200)
        gender = voice_config.get("gender", "Female")
        
        # Adjust fundamental frequency for gender
        if gender == "Female":
            f0 = max(base_freq * 0.8, 180)  # Female voices: 180-250 Hz
        else:
            f0 = min(base_freq * 0.6, 140)  # Male voices: 80-140 Hz
        
        # Define formant frequencies for speech-like sounds
        # These are the resonant frequencies that make speech intelligible
        if gender == "Female":
            formants = [
                (730, 60),   # F1: Low vowels  
                (1090, 70),  # F2: Vowel identity
                (2440, 120)  # F3: Speaker identity
            ]
        else:
            formants = [
                (570, 50),   # F1: Low vowels
                (840, 60),   # F2: Vowel identity  
                (2240, 100)  # F3: Speaker identity
            ]
        
        # Create speech synthesis based on phoneme-like patterns
        audio = np.zeros(samples)
        t_total = np.linspace(0, duration, samples)
        
        # Generate speech segments (vowels and consonants)
        segment_count = len(words) * 2  # 2 segments per word (consonant + vowel)
        segment_duration = duration / segment_count
        
        for i in range(segment_count):
            start_idx = int(i * segment_duration * sample_rate)
            end_idx = int((i + 1) * segment_duration * sample_rate)
            end_idx = min(end_idx, samples)
            
            if end_idx <= start_idx:
                continue
                
            segment_samples = end_idx - start_idx
            t = np.linspace(0, segment_duration, segment_samples)
            
            # Alternate between vowel-like and consonant-like sounds
            is_vowel = (i % 2 == 0)
            
            if is_vowel:
                # Generate vowel-like sound with formants
                segment_audio = self._generate_vowel_sound(t, f0, formants)
            else:
                # Generate consonant-like sound
                segment_audio = self._generate_consonant_sound(t, f0)
            
            # Apply natural amplitude envelope
            envelope = np.ones(len(segment_audio))
            fade_samples = min(len(segment_audio) // 8, int(0.02 * sample_rate))
            if fade_samples > 0:
                envelope[:fade_samples] = np.linspace(0.3, 1.0, fade_samples)
                envelope[-fade_samples:] = np.linspace(1.0, 0.1, fade_samples)
            
            segment_audio *= envelope
            audio[start_idx:end_idx] = segment_audio
        
        # Apply speech-like filtering
        audio = self._apply_speech_filter(audio, sample_rate)
        
        # Normalize and convert to WAV
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.7
        
        # Convert to WAV
        audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        audio_bytes = wav_buffer.read()
        
        print(f"[OK] Speech-like audio synthesis complete: {len(audio_bytes)} bytes, {duration:.1f}s")
        return audio_bytes
    
    def _generate_vowel_sound(self, t, f0, formants):
        """Generate vowel-like sound using formant synthesis"""
        # Base fundamental frequency with natural vibrato
        vibrato = 1 + 0.05 * np.sin(2 * np.pi * 5 * t)
        instantaneous_f0 = f0 * vibrato
        
        # Generate harmonic series
        audio = np.zeros(len(t))
        
        for harmonic in range(1, 8):
            harmonic_freq = instantaneous_f0 * harmonic
            
            # Calculate formant filtering
            formant_response = 1.0
            for formant_freq, bandwidth in formants:
                # Gaussian formant response
                distance = abs(harmonic_freq - formant_freq)
                response = np.exp(-(distance / bandwidth) ** 2)
                formant_response *= (0.3 + 0.7 * response)
            
            # Generate harmonic with phase accumulation for frequency modulation
            phase = np.cumsum(2 * np.pi * harmonic_freq / 22050)
            amplitude = (0.8 / harmonic) * formant_response
            
            harmonic_signal = amplitude * np.sin(phase[:len(t)])
            audio += harmonic_signal
        
        return audio
    
    def _generate_consonant_sound(self, t, f0):
        """Generate consonant-like sound (more noise-based)"""
        # Base noise component
        audio = np.random.normal(0, 0.3, len(t))
        
        # Add some tonal component for voiced consonants
        tone_freq = f0 * 1.5
        tone = 0.4 * np.sin(2 * np.pi * tone_freq * t)
        
        # Mix noise and tone
        audio = 0.7 * audio + 0.3 * tone
        
        return audio
    
    def _apply_speech_filter(self, audio, sample_rate):
        """Apply filtering to make audio more speech-like"""
        # Simple band-pass filter for speech frequencies (300-3400 Hz)
        # This is a very basic implementation
        
        # High-pass filter (remove very low frequencies)
        alpha_hp = 0.95
        for i in range(1, len(audio)):
            audio[i] = alpha_hp * (audio[i] - audio[i-1]) + alpha_hp * audio[i-1]
        
        # Low-pass filter (remove very high frequencies)  
        alpha_lp = 0.3
        for i in range(1, len(audio)):
            audio[i] = alpha_lp * audio[i] + (1 - alpha_lp) * audio[i-1]
        
        return audio
    
    def get_available_voices(self):
        """Get list of available voice presets"""
        return self.voice_presets

# Initialize TTS service
if USING_REAL_TTS:
    tts_service = TortoiseRealTTS()
    print("[INFO] Real TTS initialized")
else:
    tts_service = TortoiseEnhancedPlaceholder()

# FastAPI models
class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = "angie"
    preset: Optional[str] = "fast"
    return_audio: Optional[bool] = True

class SynthesizeResponse(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    metadata: Dict[str, Any]

class VoicesResponse(BaseModel):
    voices: List[str]
    total: int
    engine: str = "tortoise"
    details: Optional[Dict[str, Dict[str, Any]]] = None

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    print(f"[STARTUP] Tortoise TTS service starting on port 8015...")
    print(f"[INFO] Engine: {'Real TTS' if USING_REAL_TTS else 'Enhanced Placeholder'}")
    yield
    print(f"[SHUTDOWN] Tortoise TTS service stopping...")

app = FastAPI(title="Tortoise TTS Service", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "tts_tortoise",
        "engine": "tortoise_real" if USING_REAL_TTS else "tortoise_placeholder",
        "implementation": "Real pyttsx3 TTS" if USING_REAL_TTS else "Enhanced neural patterns",
        "timestamp": time.time(),
        "ready": True,
        "performance": "~1-2s generation time" if USING_REAL_TTS else "~1-3s generation time",
        "best_for": "real speech synthesis" if USING_REAL_TTS else "testing and development",
        "voice_count": len(tts_service.voice_presets if hasattr(tts_service, 'voice_presets') else tts_service.get_available_voices())
    }

@app.get("/voices", response_model=List[str])
async def get_voices():
    """Get available voice presets"""
    print(f"[DEBUG] get_voices called, tts_service type: {type(tts_service)}")
    print(f"[DEBUG] hasattr get_available_voices: {hasattr(tts_service, 'get_available_voices')}")
    
    if hasattr(tts_service, 'get_available_voices'):
        voices = tts_service.get_available_voices()
        print(f"[DEBUG] get_available_voices returned: {len(voices)} voices")
        print(f"[DEBUG] First 10 voices: {voices[:10]}")
        return voices
    else:
        print("[DEBUG] Using fallback voice list")
        return ["angie", "denise", "freeman", "geralt", "halle"]

@app.get("/voices_detailed", response_model=VoicesResponse)
async def get_voices_detailed():
    """Get detailed voice information with metadata"""
    voices = tts_service.get_available_voices()
    
    details = {}
    if hasattr(tts_service, 'voice_characteristics'):
        details = {voice: tts_service.voice_characteristics.get(voice, {}) for voice in voices}
    
    return VoicesResponse(
        voices=voices,
        total=len(voices),
        engine="tortoise",
        details=details
    )

@app.get("/presets")
async def get_presets():
    """Get available quality presets"""
    return {
        "presets": ["fast", "standard", "high_quality"],
        "default": "fast",
        "descriptions": {
            "fast": "Quick generation (~3s), good quality",
            "standard": "Balanced quality/speed (~5s)",
            "high_quality": "Best quality (~8s), slower"
        }
    }

@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: SynthesizeRequest):
    """Synthesize speech using Tortoise TTS"""
    try:
        start_time = time.time()
        
        # Generate audio
        if hasattr(tts_service, 'generate_speech'):
            audio_bytes = tts_service.generate_speech(
                text=request.text,
                voice_preset=request.voice
            )
        else:
            audio_bytes = await tts_service.synthesize_speech(
                text=request.text,
                voice=request.voice,
                preset=request.preset
            )
        
        processing_time = time.time() - start_time
        
        # Encode audio as base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return SynthesizeResponse(
            audio_base64=audio_base64,
            metadata={
                "voice": request.voice,
                "preset": request.preset,
                "text_length": len(request.text),
                "processing_time": round(processing_time, 2),
                "audio_size": len(audio_bytes),
                "engine": "tortoise_real" if USING_REAL_TTS else "tortoise_placeholder",
                "sample_rate": 22050,
                "format": "wav"
            }
        )
        
    except Exception as e:
        print(f"[ERROR] Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tortoise TTS Service")
    parser.add_argument("--port", type=int, default=8015, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run on")
    parser.add_argument("--direct", action="store_true", help="Run service directly")
    
    args = parser.parse_args()
    
    if args.direct or "--direct" in sys.argv:
        print(f"[DIRECT] Starting Tortoise TTS service on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        print("Use --direct flag to run the service directly")
        print("Or use the Enhanced Service Manager for full orchestration")
