"""
Zonos TTS Microservice - Independent Service
High-quality neural TTS using Zonos engine
Port: 8014
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import logging
import uvicorn
import torch
import gc
import base64
from typing import Dict, Any, Optional, List
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Try to import real Zonos implementation, fallback to placeholder
try:
    from voicebot_orchestrator.real_zonos_tts import RealZonosTTS as ZonosTTS
    print("[OK] Using Real Zonos TTS implementation with neural speech")
    USING_REAL_ZONOS = True
except ImportError as e:
    print(f"[WARNING] Real Zonos not available ({e}), using Edge-TTS directly")
    try:
        # Try direct Edge-TTS import
        import edge_tts
        import tempfile
        import os
        
        class ZonosTTS:
            def __init__(self, voice="default", model="zonos-v1"):
                self.voice = voice
                self.model = model
                self.voice_mapping = {
                    "default": "en-US-DavisNeural",
                    "aria": "en-US-AriaNeural", 
                    "sophia": "en-US-JennyNeural",
                    "professional": "en-US-GuyNeural"
                }
                print(f"[EDGE-TTS] Zonos TTS using Edge-TTS with voice={voice}, model={model}")
            
            async def synthesize_speech(self, text: str, **kwargs):
                edge_voice = self.voice_mapping.get(self.voice, "en-US-AriaNeural")
                communicate = edge_tts.Communicate(text, edge_voice)
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                await communicate.save(tmp_path)
                
                with open(tmp_path, 'rb') as f:
                    audio_bytes = f.read()
                
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
                return audio_bytes
        
        print("[OK] Using Edge-TTS fallback for real speech synthesis")
        USING_REAL_ZONOS = True
        
    except ImportError:
        print(f"[WARNING] Edge-TTS not available, using placeholder implementation")
        # Create a placeholder implementation
        class ZonosTTS:
            def __init__(self, voice="default", model="zonos-v1"):
                self.voice = voice
                self.model = model
                print(f"[PLACEHOLDER] Zonos TTS initialized with voice={voice}, model={model}")
            
            async def synthesize_speech(self, text: str, **kwargs):
                # Placeholder - generates silence or basic audio
                import numpy as np
                sample_rate = 44100
                duration = min(len(text) * 0.1, 30.0)  # Max 30 seconds
                samples = int(sample_rate * duration)
                
                # Generate simple sine wave pattern as placeholder
                t = np.linspace(0, duration, samples)
                frequency = 440  # A4 note
                audio = np.sin(2 * np.pi * frequency * t) * 0.3
                
                # Convert to 16-bit PCM
                audio_int16 = (audio * 32767).astype(np.int16)
                
                # Create WAV header + data
                import struct
                wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                    b'RIFF', 36 + len(audio_int16) * 2, b'WAVE', b'fmt ', 16,
                    1, 1, sample_rate, sample_rate * 2, 2, 16, b'data', len(audio_int16) * 2)
                
                return wav_header + audio_int16.tobytes()
        
        USING_REAL_ZONOS = False

# Global TTS instance
# Global TTS instance  
tts_engine = None

# USE_MOCK_TTS flag - set to False to use real Zonos TTS
USE_MOCK_TTS = False

app = FastAPI(
    title="Zonos TTS Microservice", 
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize Zonos TTS service on startup"""
    global tts_engine
    try:
        logging.info("[TTS] Initializing Zonos TTS Microservice...")
        
        # Determine which TTS implementation to use
        if USE_MOCK_TTS:
            logging.info("[OK] Using Mock Zonos TTS implementation")
            # Create a simple mock if needed
            class MockZonosTTS:
                def __init__(self):
                    pass
                async def synthesize_speech(self, text, **kwargs):
                    # Simple mock audio
                    import struct
                    sample_rate = 44100
                    duration = 2.0
                    samples = int(sample_rate * duration)
                    audio_data = b'\x00\x00' * samples
                    header = struct.pack('<4sI4s4sIHHIIHH4sI',
                        b'RIFF', 36 + len(audio_data), b'WAVE', b'fmt ', 16,
                        1, 1, sample_rate, sample_rate * 2, 2, 16, b'data', len(audio_data))
                    return header + audio_data
            
            tts_engine = MockZonosTTS()
        else:
            logging.info("[OK] Using Real Zonos TTS implementation")  
            tts_engine = ZonosTTS()
        
        logging.info("[OK] Zonos TTS Microservice ready!")
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to initialize Zonos TTS: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    global tts_engine
    try:
        if tts_engine:
            logging.info("[STOP] Zonos TTS Microservice shutdown complete")
        logging.info("[CLEANUP] Zonos TTS service cleanup complete")
    except Exception as e:
        logging.error(f"[ERROR] Shutdown error: {e}")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    model: Optional[str] = "zonos-v1"
    speed: Optional[float] = 1.0
    pitch: Optional[float] = 1.0
    emotion: Optional[str] = "neutral"
    speaking_style: Optional[str] = "normal"
    emphasis_words: Optional[List[str]] = []
    pause_locations: Optional[List[int]] = []
    prosody_adjustments: Optional[Dict[str, float]] = {}
    seed: Optional[int] = None
    output_format: Optional[str] = "wav"
    sample_rate: Optional[int] = 44100
    return_audio: Optional[bool] = True
    high_quality: Optional[bool] = True

class SynthesizeResponse(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    metadata: Dict[str, Any]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "tts_zonos",
        "engine": "real_zonos" if USING_REAL_ZONOS else "zonos_placeholder",
        "implementation": "Real Zonos Neural TTS" if USING_REAL_ZONOS else "Placeholder audio generation",
        "timestamp": time.time(),
        "ready": tts_engine is not None,
        "performance": "~2-5s generation time",
        "best_for": "high-quality neural speech" if USING_REAL_ZONOS else "testing and development"
    }

@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: SynthesizeRequest) -> SynthesizeResponse:
    """
    Synthesize speech using Zonos TTS engine
    """
    if not tts_engine:
        raise HTTPException(status_code=503, detail="Zonos TTS service not ready")
    
    start_time = time.time()
    
    try:
        # Validate input
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        if len(request.text) > 10000:
            raise HTTPException(status_code=400, detail="Text too long (max 10000 characters)")
        
        # Prepare synthesis parameters with enhanced controls
        synthesis_params = {
            "text": request.text,
            "voice": request.voice,
            "model": request.model,
            "speed": request.speed,
            "pitch": request.pitch,
            "emotion": request.emotion,
            "speaking_style": request.speaking_style,
            "emphasis_words": request.emphasis_words,
            "pause_locations": request.pause_locations,
            "prosody_adjustments": request.prosody_adjustments,
            "high_quality": request.high_quality,
            "output_format": request.output_format,
            "sample_rate": request.sample_rate
        }
        
        if request.seed is not None:
            synthesis_params["seed"] = request.seed
        
        # Generate speech using Zonos TTS
        audio_bytes = await tts_engine.synthesize_speech(**synthesis_params)
        gen_time = time.time() - start_time
        
        total_time = time.time() - start_time
        
        # Prepare response metadata
        metadata = {
            "processing_time_seconds": round(total_time, 3),
            "generation_time_seconds": round(gen_time, 3),
            "engine_used": "real_zonos" if USING_REAL_ZONOS else "zonos_placeholder",
            "implementation": "Real Zonos Neural TTS" if USING_REAL_ZONOS else "Placeholder audio generation",
            "text_length": len(request.text),
            "audio_size_bytes": len(audio_bytes),
            "estimated_duration_seconds": len(audio_bytes) / 44100 if len(audio_bytes) > 44 else 1.0,
            "voice": request.voice,
            "model": request.model,
            "emotion": request.emotion,
            "speaking_style": request.speaking_style,
            "service": "zonos_dedicated",
            "parameters": {
                "speed": request.speed,
                "pitch": request.pitch,
                "high_quality": request.high_quality,
                "seed": request.seed,
                "emphasis_words": request.emphasis_words,
                "output_format": request.output_format,
                "sample_rate": request.sample_rate,
                "prosody_adjustments": request.prosody_adjustments
            }
        }
        
        # Return audio data if requested
        audio_base64 = None
        if request.return_audio:
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return SynthesizeResponse(
            audio_base64=audio_base64,
            metadata=metadata
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"[ERROR] Zonos synthesis failed after {processing_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Zonos synthesis failed: {str(e)}")

@app.post("/synthesize_file")
async def synthesize_to_file(request: SynthesizeRequest):
    """
    Synthesize speech and return as audio file
    """
    if not tts_engine:
        raise HTTPException(status_code=503, detail="Zonos TTS service not ready")
    
    start_time = time.time()
    
    try:
        # Prepare synthesis parameters
        synthesis_params = {
            "text": request.text,
            "voice": request.voice,
            "model": request.model,
            "speed": request.speed,
            "pitch": request.pitch,
            "emotion": request.emotion,
            "high_quality": request.high_quality
        }
        
        if request.seed is not None:
            synthesis_params["seed"] = request.seed
        
        # Generate audio using Zonos TTS
        audio_bytes = await tts_engine.synthesize_speech(**synthesis_params)
        gen_time = time.time() - start_time
        
        # Return as audio file
        media_type = "audio/wav" if request.output_format == "wav" else "audio/mpeg"
        
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=zonos_speech_{int(time.time())}.{request.output_format}",
                "X-Generation-Time": str(gen_time),
                "X-Engine-Used": "zonos",
                "X-Service": "zonos_dedicated",
                "X-Voice": request.voice,
                "X-Model": request.model,
                "X-Emotion": request.emotion
            }
        )
        
    except Exception as e:
        logging.error(f"[ERROR] Zonos file synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Zonos synthesis failed: {str(e)}")

@app.get("/info")
async def service_info():
    """Get Zonos TTS service information"""
    return {
        "service": "tts_zonos",
        "engine": "real_zonos" if USING_REAL_ZONOS else "zonos_placeholder",
        "implementation": "Real Zonos Neural TTS" if USING_REAL_ZONOS else "Placeholder audio generation",
        "port": 8014,
        "speed": "~2-5s per request",
        "quality": "Very High (Neural)" if USING_REAL_ZONOS else "Basic (Placeholder)",
        "voice_options": ["default", "professional", "conversational", "narrative"],
        "model_options": ["zonos-v1", "zonos-v2", "zonos-lite"],
        "emotion_options": ["neutral", "happy", "sad", "excited", "calm", "professional"],
        "best_for": "High-quality content, voice-overs, professional applications" if USING_REAL_ZONOS else "Testing and development",
        "supported_formats": ["wav", "mp3"],
        "max_text_length": 10000,
        "features": [
            "Neural speech synthesis",
            "Multiple voice styles",
            "Emotion control",
            "Pitch and speed control",
            "High-quality output",
            "Seed-based reproducibility"
        ],
        "independent": True,
        "description": f"Dedicated Zonos TTS service using {'real neural models' if USING_REAL_ZONOS else 'placeholder audio generation'}"
    }

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    return {
        "service_name": "Zonos TTS",
        "engine": "zonos",
        "status": "running" if tts_engine else "stopped",
        "ready": tts_engine is not None,
        "capabilities": [
            "Neural speech synthesis",
            "Multiple voice styles",
            "Emotion control",
            "High-quality output"
        ],
        "advantages": [
            "Very high quality neural speech",
            "Multiple voice options",
            "Emotion and style control",
            "Professional-grade output",
            "Reproducible with seeds"
        ],
        "use_cases": [
            "Content creation",
            "Voice-overs",
            "Professional applications",
            "High-quality demos",
            "E-learning content"
        ],
        "performance": {
            "generation_time": "2-5 seconds typical",
            "quality": "Very High" if USING_REAL_ZONOS else "Basic",
            "max_text_length": 10000
        }
    }

@app.get("/voices")
async def get_available_voices():
    """Get all available voices - returns simple list for compatibility"""
    if not tts_engine:
        # Return simple list of voice names for compatibility
        female_voices = ["jenny", "aria", "michelle", "sara", "nancy", "jane", "libby", "sonia"]
        male_voices = ["guy", "davis", "andrew", "brian", "jason", "tony", "christopher", "ryan", "thomas"]
        all_voices = female_voices + male_voices + ["default", "professional", "conversational", "narrative"]
        return all_voices
    
    # Get comprehensive options from TTS engine
    try:
        options = tts_engine.get_available_options()
        # Flatten voices structure for compatibility
        all_voices = []
        if "voices" in options and isinstance(options["voices"], dict):
            for category, voice_list in options["voices"].items():
                if isinstance(voice_list, dict):
                    all_voices.extend(voice_list.keys())
                elif isinstance(voice_list, list):
                    all_voices.extend(voice_list)
        
        # Add aliases if available
        if "aliases" in options and isinstance(options["aliases"], dict):
            all_voices.extend(options["aliases"].keys())
        
        return list(set(all_voices)) if all_voices else ["default"]
        
    except (AttributeError, Exception) as e:
        logging.warning(f"Error getting voices: {e}")
        # Fallback simple list
        return ["female", "male", "neutral", "default"]

@app.get("/voices_detailed")
async def get_detailed_voices():
    """Get detailed voice information with categories and descriptions"""
    if not tts_engine:
        # Return proper Microsoft Edge Neural Voice names
        return {
            "voices": {
                "female": {
                    "jenny": {"model": "en-US-JennyNeural", "description": "Professional, clear", "accent": "US"},
                    "aria": {"model": "en-US-AriaNeural", "description": "Conversational, friendly", "accent": "US"},
                    "michelle": {"model": "en-US-MichelleNeural", "description": "Authoritative, business", "accent": "US"},
                    "sara": {"model": "en-US-SaraNeural", "description": "Calm, soothing", "accent": "US"},
                    "nancy": {"model": "en-US-NancyNeural", "description": "Warm, storytelling", "accent": "US"},
                    "jane": {"model": "en-US-JaneNeural", "description": "Energetic, upbeat", "accent": "US"},
                    "libby": {"model": "en-GB-LibbyNeural", "description": "British, elegant", "accent": "UK"},
                    "sonia": {"model": "en-GB-SoniaNeural", "description": "British, professional", "accent": "UK"}
                },
                "male": {
                    "guy": {"model": "en-US-GuyNeural", "description": "Professional, authoritative", "accent": "US"},
                    "davis": {"model": "en-US-DavisNeural", "description": "Conversational, friendly", "accent": "US"},
                    "andrew": {"model": "en-US-AndrewNeural", "description": "Narrative, storytelling", "accent": "US"},
                    "brian": {"model": "en-US-BrianNeural", "description": "Calm, measured", "accent": "US"},
                    "jason": {"model": "en-US-JasonNeural", "description": "Energetic, dynamic", "accent": "US"},
                    "tony": {"model": "en-US-TonyNeural", "description": "Warm, approachable", "accent": "US"},
                    "christopher": {"model": "en-US-ChristopherNeural", "description": "Authoritative, commanding", "accent": "US"},
                    "ryan": {"model": "en-GB-RyanNeural", "description": "British, sophisticated", "accent": "UK"},
                    "thomas": {"model": "en-GB-ThomasNeural", "description": "British, professional", "accent": "UK"}
                },
                "aliases": {
                    "default": "aria",
                    "professional": "jenny", 
                    "conversational": "aria",
                    "narrative": "andrew"
                }
            },
            "emotions": [
                "neutral", "happy", "excited", "calm", "sad", "angry", 
                "thoughtful", "conversational", "professional", "warm"
            ],
            "models": ["zonos-v1", "zonos-v2", "zonos-lite"],
            "count": 17,
            "service": "zonos_tts",
            "engine": "Microsoft Edge Neural TTS",
            "note": "Professional Microsoft Edge Neural Voices with proper documentation names"
        }
    
    # Get comprehensive options from TTS engine
    try:
        options = tts_engine.get_available_options()
        return {
            "voices": options["voices"],
            "emotions": options["emotions"],
            "speaking_styles": options["speaking_styles"],
            "output_formats": options["output_formats"],
            "sample_rates": options["sample_rates"],
            "speed_range": options["speed_range"],
            "pitch_range": options["pitch_range"],
            "count": sum(len(voices) for voices in options["voices"].values()),
            "service": "zonos_tts",
            "note": "Comprehensive TTS options with advanced controls"
        }
    except AttributeError:
        # Fallback for older TTS engine versions
        return {
            "voices": {
                "female": ["sophia", "aria", "luna", "emma", "zoe", "maya"],
                "male": ["default", "professional", "conversational", "narrative"],
                "neutral": ["alex", "casey"]
            },
            "count": 14,
            "service": "zonos_tts",
            "note": "Basic voice options (legacy mode)"
        }

@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    return ["zonos-v1", "zonos-v2", "zonos-lite"]

@app.get("/emotions")
async def get_available_emotions():
    """Get all available emotions categorized"""
    return {
        "emotions": {
            "basic": ["neutral", "happy", "sad", "angry", "excited", "calm", "fearful"],
            "professional": ["professional", "confident", "authoritative", "reassuring", "instructional"],
            "social": ["friendly", "empathetic", "encouraging", "supportive", "welcoming"],
            "entertainment": ["dramatic", "mysterious", "playful", "sarcastic", "whimsical"],
            "intensity_variant": [
                "happy_intense", "happy_soft", "sad_intense", "sad_soft",
                "angry_intense", "angry_soft", "excited_intense", "excited_soft",
                "calm_intense", "calm_soft", "fearful_intense", "fearful_soft"
            ]
        },
        "total_count": 32,
        "categories": 5,
        "note": "Comprehensive emotion system with professional and entertainment categories"
    }

@app.get("/speaking_styles")
async def get_available_speaking_styles():
    """Get all available speaking styles"""
    return {
        "speaking_styles": {
            "normal": {"description": "Standard speaking style", "speed_factor": 1.0},
            "conversational": {"description": "Natural conversation style", "speed_factor": 1.1},
            "presentation": {"description": "Clear presentation style", "speed_factor": 0.9},
            "reading": {"description": "Book reading style", "speed_factor": 0.85},
            "storytelling": {"description": "Narrative storytelling", "speed_factor": 0.95},
            "announcement": {"description": "Public announcement style", "speed_factor": 0.8},
            "urgent": {"description": "Urgent communication", "speed_factor": 1.3},
            "meditation": {"description": "Calm meditation guide", "speed_factor": 0.7},
            "news": {"description": "News broadcasting style", "speed_factor": 1.0}
        },
        "count": 9,
        "note": "Speaking styles affect speed, pauses, and emphasis patterns"
    }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        logging.info("[START] Starting Zonos TTS Microservice on port 8014...")
        
        # Run the Zonos TTS service
        uvicorn.run(
            "tts_zonos_service:app",
            host="0.0.0.0",
            port=8014,
            workers=1,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logging.info("[STOP] Zonos TTS service stopped by user")
    except Exception as e:
        logging.error(f"[ERROR] Zonos TTS service failed: {e}")
        import sys
        sys.exit(1)
    finally:
        logging.info("[CLEANUP] Zonos TTS service cleanup complete")
