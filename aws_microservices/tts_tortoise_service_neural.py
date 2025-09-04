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
    engine: str = "tortoise_neural"
    details: Optional[Dict[str, Dict[str, Any]]] = None

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    print(f"[STARTUP] Tortoise TTS service starting on port 8015...")
    
    # Initialize TTS
    success = initialize_tts()
    if not success:
        print("[ERROR] Failed to initialize TTS service")
    else:
        print("[INFO] Neural Tortoise TTS service ready")
    
    yield
    print(f"[SHUTDOWN] Tortoise TTS service stopping...")

app = FastAPI(title="Tortoise TTS Service", version="2.0.0", lifespan=lifespan)

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
    global tts_service
    return {
        "status": "healthy",
        "service": "tts_tortoise",
        "engine": "neural_tortoise",
        "implementation": "Neural Network TTS with PyTorch",
        "timestamp": time.time(),
        "ready": tts_service is not None,
        "performance": "~2-5s generation time",
        "best_for": "ultra high-quality voice synthesis",
        "voice_count": len(tts_service.get_available_voices()) if tts_service else 0,
        "features": [
            "30 unique voice personalities",
            "Neural network synthesis",
            "Voice cloning capabilities", 
            "Emotion-aware generation",
            "High-quality audio output"
        ]
    }

@app.get("/voices", response_model=List[str])
async def get_voices():
    """Get available voice presets"""
    global tts_service
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    voices = tts_service.get_available_voices()
    print(f"[DEBUG] Returning {len(voices)} voices")
    return voices

@app.get("/voices_detailed", response_model=VoicesResponse)
async def get_voices_detailed():
    """Get detailed voice information with metadata"""
    global tts_service
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    voices = tts_service.get_available_voices()
    
    # Get voice configurations if available
    details = {}
    if hasattr(tts_service, 'voice_configs'):
        details = {voice: tts_service.voice_configs.get(voice, {}) for voice in voices}
    
    return VoicesResponse(
        voices=voices,
        total=len(voices),
        engine="neural_tortoise",
        details=details
    )

@app.get("/presets")
async def get_presets():
    """Get available quality presets"""
    return {
        "presets": ["ultrafast", "fast", "standard", "high_quality"],
        "default": "fast",
        "descriptions": {
            "ultrafast": "Quick generation (~1s), good quality",
            "fast": "Balanced generation (~2-3s), high quality",
            "standard": "Standard quality (~3-5s), very high quality",
            "high_quality": "Best quality (~5-8s), ultra high quality"
        }
    }

@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: SynthesizeRequest):
    """Synthesize speech using Neural Tortoise TTS"""
    global tts_service
    
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    try:
        start_time = time.time()
        
        print(f"[INFO] Synthesizing '{request.text[:50]}...' with voice '{request.voice}'")
        
        # Generate audio using neural Tortoise TTS
        audio_base64, metadata = tts_service.synthesize_to_base64(
            text=request.text,
            voice=request.voice or "angie",
            preset=request.preset or "fast"
        )
        
        processing_time = time.time() - start_time
        
        # Update metadata with processing info
        metadata.update({
            "processing_time": round(processing_time, 2),
            "text_length": len(request.text),
            "engine": "neural_tortoise",
            "format": "wav",
            "audio_size": len(base64.b64decode(audio_base64))
        })
        
        print(f"[SUCCESS] Synthesis completed in {processing_time:.2f}s")
        
        return SynthesizeResponse(
            audio_base64=audio_base64,
            metadata=metadata
        )
        
    except Exception as e:
        print(f"[ERROR] Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    global tts_service
    return {
        "service": "tortoise_tts_neural",
        "version": "2.0.0",
        "status": "ready" if tts_service else "not_ready",
        "model_loaded": tts_service is not None,
        "device": tts_service.device if tts_service else "unknown",
        "available_voices": len(tts_service.get_available_voices()) if tts_service else 0,
        "capabilities": [
            "Neural synthesis",
            "Voice cloning",
            "Emotion control",
            "High fidelity audio",
            "Real-time generation"
        ]
    }

@app.post("/test_voice")
async def test_voice(voice: str = "angie"):
    """Test a specific voice with sample text"""
    global tts_service
    
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    if voice not in tts_service.get_available_voices():
        raise HTTPException(status_code=400, detail=f"Voice '{voice}' not available")
    
    test_text = f"Hello, this is a test of the {voice} voice using Neural Tortoise TTS."
    
    try:
        audio_base64, metadata = tts_service.synthesize_to_base64(
            text=test_text,
            voice=voice,
            preset="fast"
        )
        
        return {
            "voice": voice,
            "test_text": test_text,
            "audio_base64": audio_base64,
            "metadata": metadata,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice test failed: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Neural Tortoise TTS Service")
    parser.add_argument("--port", type=int, default=8015, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run on")
    parser.add_argument("--direct", action="store_true", help="Run service directly")
    
    args = parser.parse_args()
    
    if args.direct or "--direct" in sys.argv:
        print(f"[DIRECT] Starting Neural Tortoise TTS service on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        print("Use --direct flag to run the service directly")
        print("Or use the Enhanced Service Manager for full orchestration")
