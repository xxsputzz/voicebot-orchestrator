"""
🔧 ZONOS TTS REAL SPEECH PATCH
=====================================
This patches the existing Zonos TTS to use real neural TTS instead of synthetic audio
Maintains full API compatibility while providing natural speech
"""

import sys
import os
import asyncio
import logging
from typing import Optional

# Add the path to our enhanced TTS
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from enhanced_real_tts import enhanced_tts
    REAL_TTS_AVAILABLE = True
    print("✅ Real TTS engine loaded successfully!")
except ImportError as e:
    REAL_TTS_AVAILABLE = False
    print(f"⚠️ Real TTS not available: {e}")
    print("📥 Run 'install_real_tts.bat' to install required packages")

# Import the original ZonosTTS class
try:
    from voicebot_orchestrator.zonos_tts_original import ZonosTTS as OriginalZonosTTS
except ImportError:
    # If original backup doesn't exist, create a simple base class
    class OriginalZonosTTS:
        def __init__(self, voice: str = "default", model: str = "zonos-v1"):
            self.voice = voice
            self.model = model
            self.available_voices = {"default": {}, "professional": {}, "conversational": {}}
            self.available_emotions = {"neutral": {}, "happy": {}, "calm": {}}
        
        async def synthesize_speech(self, text: str, voice: str = None, emotion: str = "neutral", speed: float = 1.0, seed: Optional[int] = None, **kwargs):
            # Simple fallback synthesis
            import wave
            import io
            import numpy as np
            
            # Generate simple beep sound
            sample_rate = 22050
            duration = len(text) * 0.05
            t = np.linspace(0, duration, int(sample_rate * duration))
            signal = 0.3 * np.sin(2 * np.pi * 440 * t)
            pcm_data = (signal * 32767).astype(np.int16)
            
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data.tobytes())
            
            return wav_buffer.getvalue()

class ZonosTTS(OriginalZonosTTS):
    """
    🎙️ Enhanced Zonos TTS with Real Speech Synthesis
    
    This patched version provides:
    - Real neural TTS instead of synthetic audio
    - All original voice and emotion options
    - Maintains full API compatibility
    - Falls back to enhanced synthetic if real TTS unavailable
    """
    
    def __init__(self, voice: str = "default", model: str = "zonos-v1"):
        """Initialize enhanced Zonos TTS"""
        super().__init__(voice, model)
        self.real_tts_available = REAL_TTS_AVAILABLE
        
        if self.real_tts_available:
            logging.info("🎙️ Zonos TTS initialized with real neural speech")
        else:
            logging.info("⚠️ Zonos TTS using enhanced synthetic speech")
    
    async def synthesize_speech(
        self,
        text: str,
        voice: str = None,
        emotion: str = "neutral",
        speed: float = 1.0,
        seed: Optional[int] = None,
        model: str = None,
        **kwargs
    ) -> bytes:
        """
        🎵 Synthesize speech with real neural TTS
        
        Args:
            text: Text to synthesize
            voice: Voice selection
            emotion: Emotion/style
            speed: Speech rate
            seed: Random seed for reproducibility
            model: Model selection
            
        Returns:
            bytes: WAV audio data
        """
        
        # Use provided parameters or defaults
        final_voice = voice or self.voice
        final_model = model or self.model
        
        logging.info(f"🎭 Zonos TTS synthesizing: voice={final_voice}, emotion={emotion}, speed={speed}")
        
        if self.real_tts_available:
            try:
                # Map Zonos voices to real TTS voices
                real_voice = self._map_zonos_to_real_voice(final_voice)
                
                # Synthesize with real TTS
                audio_data = await enhanced_tts.synthesize_speech(
                    text=text,
                    voice=real_voice,
                    emotion=emotion,
                    speed=speed,
                    seed=seed,
                    model=final_model
                )
                
                logging.info(f"✅ Real TTS synthesis complete: {len(audio_data)} bytes")
                return audio_data
                
            except Exception as e:
                logging.error(f"❌ Real TTS failed: {e}")
                logging.info("🔄 Falling back to enhanced synthetic speech")
                return await self._enhanced_synthetic_synthesis(text, final_voice, emotion, speed, seed)
        else:
            # Use enhanced synthetic speech
            return await self._enhanced_synthetic_synthesis(text, final_voice, emotion, speed, seed)
    
    def _map_zonos_to_real_voice(self, zonos_voice: str) -> str:
        """Map Zonos voice names to real TTS voice names"""
        
        voice_mapping = {
            # Female voices
            "sophia": "female_professional",
            "aria": "female_conversational", 
            "luna": "female_warm",
            "emma": "female_authoritative",
            "zoe": "female_energetic",
            "maya": "female_narrative",
            "isabel": "female_calm",
            "grace": "female_professional",
            "natasha": "female_conversational",
            "chloe": "female_energetic",
            "rachel": "female_warm",
            
            # Male voices
            "default": "male_professional",
            "professional": "male_professional",
            "conversational": "male_conversational",
            "narrative": "male_narrative",
            "alex": "male_professional",
            "james": "male_authoritative",
            "david": "male_conversational",
            "michael": "male_warm",
            "andrew": "male_narrative",
            "ryan": "male_energetic",
            "thomas": "male_calm",
            "daniel": "male_professional",
            "noah": "male_conversational",
        }
        
        return voice_mapping.get(zonos_voice, "female_conversational")
    
    async def _enhanced_synthetic_synthesis(self, text: str, voice: str, emotion: str, speed: float, seed: Optional[int]) -> bytes:
        """Enhanced synthetic speech synthesis (fallback)"""
        
        if self.real_tts_available:
            # Use the enhanced synthetic from real TTS engine
            try:
                voice_config = enhanced_tts.supported_voices.get(
                    self._map_zonos_to_real_voice(voice),
                    enhanced_tts.supported_voices['default']
                )
                emotion_config = enhanced_tts.emotion_styles.get(emotion, enhanced_tts.emotion_styles['neutral'])
                
                return await enhanced_tts._synthesize_enhanced_synthetic(
                    text, voice_config, emotion_config, speed, 1.0, seed
                )
            except Exception as e:
                logging.error(f"❌ Enhanced synthetic failed: {e}")
        
        # Fall back to original synthetic (last resort)
        logging.warning("🔄 Using original synthetic speech (install real TTS for better quality)")
        return await super().synthesize_speech(text, voice, emotion, speed, seed)
    
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        if self.real_tts_available:
            # Return both original Zonos voices and real TTS voices
            zonos_voices = list(self.available_voices.keys())
            real_voices = enhanced_tts.get_voices()
            return list(set(zonos_voices + real_voices))
        else:
            return list(self.available_voices.keys())
    
    def get_available_emotions(self) -> list:
        """Get list of available emotions"""
        if self.real_tts_available:
            return enhanced_tts.get_emotions()
        else:
            return list(self.available_emotions.keys())
    
    async def stream_synthesis(self, text: str, chunk_size: int = 100, **kwargs):
        """Stream synthesis for long texts"""
        if self.real_tts_available:
            try:
                async for audio_chunk in enhanced_tts.stream_synthesis(text, chunk_size, **kwargs):
                    yield audio_chunk
                return
            except Exception as e:
                logging.error(f"❌ Real TTS streaming failed: {e}")
        
        # Fallback to simple chunking
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        for chunk in chunks:
            audio_data = await self.synthesize_speech(chunk, **kwargs)
            yield audio_data

# Status information
def get_tts_status():
    """Get current TTS engine status"""
    status = {
        "real_tts_available": REAL_TTS_AVAILABLE,
        "engines": [],
        "recommendation": ""
    }
    
    if REAL_TTS_AVAILABLE:
        status["engines"] = enhanced_tts.available_engines
        status["recommendation"] = "✅ Real neural TTS active - excellent quality"
    else:
        status["engines"] = ["synthetic"]
        status["recommendation"] = "⚠️ Using synthetic speech - install real TTS for better quality"
    
    return status

def install_real_tts():
    """Instructions for installing real TTS"""
    instructions = """
🎙️ INSTALL REAL TTS FOR NATURAL SPEECH
=====================================

Run this command to install real TTS engines:

Windows:
    install_real_tts.bat

Manual installation:
    pip install edge-tts gtts pyttsx3 pydub

After installation, restart the TTS service to use real neural voices!
    """
    print(instructions)
    return instructions

# For testing
async def test_enhanced_zonos():
    """Test the enhanced Zonos TTS"""
    print("🧪 Testing Enhanced Zonos TTS...")
    
    tts = ZonosTTS()
    
    test_text = "Hello! This is the enhanced Zonos TTS with real neural speech synthesis."
    
    print(f"📊 TTS Status: {get_tts_status()}")
    print(f"🎭 Available voices: {len(tts.get_available_voices())}")
    print(f"😊 Available emotions: {len(tts.get_available_emotions())}")
    
    # Test synthesis
    audio_data = await tts.synthesize_speech(
        text=test_text,
        voice="aria",
        emotion="happy",
        speed=1.0,
        seed=12345
    )
    
    # Save test file
    filename = f"enhanced_zonos_test_{int(time.time())}.wav"
    with open(filename, 'wb') as f:
        f.write(audio_data)
    
    print(f"✅ Test complete! Generated: {filename}")
    print(f"📊 Size: {len(audio_data):,} bytes")

if __name__ == "__main__":
    import time
    asyncio.run(test_enhanced_zonos())
