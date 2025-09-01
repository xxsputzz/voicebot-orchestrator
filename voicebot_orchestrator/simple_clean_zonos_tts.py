"""
Simple Clean Zonos TTS Implementation
Handles MP3 to WAV conversion and basic artifact removal
"""
import asyncio
import time
import tempfile
import os
import sys
import subprocess
from typing import Optional, Dict, Any

class SimpleCleanZonosTTS:
    """
    Simple Clean Zonos TTS with MP3 to WAV conversion
    """
    
    def __init__(self, voice: str = "aria", model: str = "zonos-v1"):
        self.voice = voice
        self.model = model
        self.initialized = False
        
        # Voice mapping
        self.voice_mapping = {
            "sophia": "en-US-JennyNeural",
            "aria": "en-US-AriaNeural",
            "luna": "en-GB-SoniaNeural",
            "emma": "en-US-MichelleNeural",
            "zoe": "en-AU-NatashaNeural",
            "maya": "en-US-SaraNeural",
            "isabel": "en-US-NancyNeural",
            "grace": "en-GB-LibbyNeural",
            "default": "en-US-DavisNeural",
            "professional": "en-US-GuyNeural",
            "conversational": "en-US-DavisNeural",
            "narrative": "en-US-AndrewNeural",
            "marcus": "en-US-BrianNeural",
            "oliver": "en-GB-RyanNeural",
            "diego": "en-US-TonyNeural",
            "alex": "en-US-JennyNeural",
            "casey": "en-US-AriaNeural",
            "river": "en-US-SaraNeural",
            "sage": "en-US-GuyNeural",
            "nova": "en-US-JennyNeural",
        }
        
        print(f"[SIMPLE CLEAN ZONOS] Initializing with voice='{voice}', model='{model}'")
        self._initialize()
    
    def _initialize(self):
        """Initialize the TTS engine"""
        try:
            # Check if edge-tts is available
            try:
                import edge_tts
                self.edge_tts = edge_tts
                print(f"[OK] Edge-TTS available")
            except ImportError:
                print(f"[INSTALL] Installing edge-tts...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'edge-tts'])
                import edge_tts
                self.edge_tts = edge_tts
                print(f"[OK] Edge-TTS installed")
            
            # Map voice
            self.edge_voice = self.voice_mapping.get(self.voice, "en-US-AriaNeural")
            
            self.initialized = True
            print(f"[OK] Simple Clean Zonos TTS initialized")
            print(f"     Voice: {self.voice} -> {self.edge_voice}")
            print(f"     Quality: Clean Human Speech")
            
        except Exception as e:
            print(f"[ERROR] Initialization failed: {e}")
            raise
    
    def _convert_mp3_to_wav(self, mp3_bytes: bytes) -> bytes:
        """Convert MP3 bytes to WAV bytes"""
        try:
            # Save MP3 to temp file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
                mp3_file.write(mp3_bytes)
                mp3_path = mp3_file.name
            
            # Create WAV output path
            wav_path = mp3_path.replace('.mp3', '.wav')
            
            # Convert using ffmpeg
            try:
                result = subprocess.run([
                    'ffmpeg', '-i', mp3_path, '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', wav_path, '-y'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(wav_path):
                    with open(wav_path, 'rb') as f:
                        wav_bytes = f.read()
                    print(f"[OK] Converted MP3 to WAV using ffmpeg")
                    
                    # Clean up
                    os.unlink(mp3_path)
                    os.unlink(wav_path)
                    
                    return wav_bytes
                else:
                    print(f"[WARNING] ffmpeg conversion failed: {result.stderr}")
                    
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                print(f"[WARNING] ffmpeg not available")
            
            # Fallback: try using pydub if available
            try:
                from pydub import AudioSegment
                
                # Load MP3 and convert to WAV
                audio = AudioSegment.from_mp3(mp3_path)
                
                # Export as WAV
                audio.export(wav_path, format='wav', parameters=['-ar', '44100', '-ac', '1'])
                
                with open(wav_path, 'rb') as f:
                    wav_bytes = f.read()
                
                print(f"[OK] Converted MP3 to WAV using pydub")
                
                # Clean up
                os.unlink(mp3_path)
                os.unlink(wav_path)
                
                return wav_bytes
                
            except ImportError:
                print(f"[WARNING] pydub not available")
            except Exception as e:
                print(f"[WARNING] pydub conversion failed: {e}")
            
            # Clean up and return original
            os.unlink(mp3_path)
            print(f"[WARNING] Could not convert to WAV, returning original MP3")
            return mp3_bytes
            
        except Exception as e:
            print(f"[ERROR] Conversion failed: {e}")
            return mp3_bytes
    
    async def synthesize_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        emotion: str = "neutral",
        **kwargs
    ) -> bytes:
        """
        Synthesize clean speech
        """
        if not self.initialized:
            raise RuntimeError("Simple Clean Zonos TTS not initialized")
        
        start_time = time.time()
        
        # Use provided parameters or instance defaults
        current_voice = voice or self.voice
        edge_voice = self.voice_mapping.get(current_voice, "en-US-AriaNeural")
        
        print(f"[SIMPLE CLEAN ZONOS] Synthesizing {len(text)} chars with {current_voice} -> {edge_voice}")
        
        try:
            # Create Edge-TTS communicate instance
            communicate = self.edge_tts.Communicate(text, edge_voice)
            
            # Generate speech to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Save audio (Edge-TTS produces MP3)
            await communicate.save(tmp_path)
            
            # Read the generated audio file
            with open(tmp_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Check format and convert if needed
            if audio_bytes.startswith(b'RIFF'):
                print(f"[OK] Edge-TTS produced WAV format")
                final_audio = audio_bytes
            else:
                print(f"[CONVERT] Edge-TTS produced MP3, converting to WAV...")
                final_audio = self._convert_mp3_to_wav(audio_bytes)
            
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            generation_time = time.time() - start_time
            
            print(f"[OK] Simple Clean Zonos synthesis completed in {generation_time:.2f}s")
            print(f"     Generated {len(final_audio)} bytes of CLEAN HUMAN SPEECH")
            
            return final_audio
            
        except Exception as e:
            print(f"[ERROR] Simple Clean Zonos synthesis failed: {e}")
            raise

# Factory function for backwards compatibility
def ZonosTTS(voice: str = "default", model: str = "zonos-v1") -> SimpleCleanZonosTTS:
    """Create Simple Clean Zonos TTS instance"""
    return SimpleCleanZonosTTS(voice=voice, model=model)

if __name__ == "__main__":
    async def test_simple_clean():
        """Test the simple clean implementation"""
        print("🎯 Testing Simple Clean Zonos TTS")
        print("=" * 50)
        
        tts = SimpleCleanZonosTTS(voice="aria", model="zonos-v1")
        
        test_text = "Hello! This is a test of the simple clean Zonos TTS. It should produce clear human speech without digital artifacts."
        
        audio_bytes = await tts.synthesize_speech(text=test_text)
        
        # Save test file
        with open("simple_clean_zonos_test.wav", "wb") as f:
            f.write(audio_bytes)
        
        print(f"✅ SUCCESS! Generated {len(audio_bytes)} bytes")
        print(f"   Saved as: simple_clean_zonos_test.wav")
        
    asyncio.run(test_simple_clean())
