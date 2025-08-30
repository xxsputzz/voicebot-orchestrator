"""
Kokoro TTS Microservice - Independent Service
Fast, real-time TTS using Kokoro engine only
Port: 8011
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import asyncio
import logging
import uvicorn
import torch
import gc
import base64
from typing import Dict, Any, Optional
import time

# Import your existing TTS implementation
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from voicebot_orchestrator.tts import KokoroTTS

app = FastAPI(title="Kokoro TTS Microservice", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global TTS instance - Kokoro only
tts_service = None

class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = "af_bella"
    speed: Optional[float] = 1.0
    output_format: Optional[str] = "wav"
    return_audio: Optional[bool] = True

class SynthesizeResponse(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    metadata: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize Kokoro TTS service on startup"""
    global tts_service
    logging.info("[TTS] Initializing Kokoro TTS Microservice...")
    
    try:
        # Initialize basic Kokoro TTS
        tts_service = KokoroTTS(voice="af_bella")
        
        logging.info("[OK] Kokoro TTS Microservice ready!")
        
    except Exception as e:
        logging.error(f"[ERROR] Kokoro TTS initialization failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global tts_service
    if tts_service:
        try:
            # Basic KokoroTTS doesn't need cleanup
            tts_service = None
            logging.info("[STOP] Kokoro TTS Microservice shutdown complete")
        except Exception as e:
            logging.error(f"[WARNING] Cleanup warning: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "tts_kokoro",
        "engine": "kokoro",
        "timestamp": time.time(),
        "ready": tts_service is not None,
        "performance": "~0.8s generation time",
        "best_for": "real-time conversation"
    }

@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: SynthesizeRequest) -> SynthesizeResponse:
    """
    Synthesize speech using Kokoro TTS engine
    """
    if not tts_service:
        raise HTTPException(status_code=503, detail="Kokoro TTS service not ready")
    
    start_time = time.time()
    
    try:
        # Validate input
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        if len(request.text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")
        
        # Generate speech using basic Kokoro TTS
        audio_bytes = await tts_service.synthesize_speech(text=request.text)
        gen_time = time.time() - start_time
        
        total_time = time.time() - start_time
        
        # Prepare response
        metadata = {
            "processing_time_seconds": round(total_time, 3),
            "generation_time_seconds": round(gen_time, 3),
            "engine_used": "kokoro",
            "text_length": len(request.text),
            "audio_size_bytes": len(audio_bytes),
            "estimated_duration_seconds": len(audio_bytes) / 32000,
            "voice": request.voice,
            "service": "kokoro_dedicated"
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
        logging.error(f"[ERROR] Kokoro synthesis failed after {processing_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Kokoro synthesis failed: {str(e)}")

@app.post("/synthesize_file")
async def synthesize_to_file(request: SynthesizeRequest):
    """
    Synthesize speech and return as audio file
    """
    if not tts_service:
        raise HTTPException(status_code=503, detail="Kokoro TTS service not ready")
    
    start_time = time.time()
    
    try:
        # Generate audio using basic Kokoro TTS
        audio_bytes = await tts_service.synthesize_speech(text=request.text)
        gen_time = time.time() - start_time
        
        # Return as audio file
        media_type = "audio/wav" if request.output_format == "wav" else "audio/mpeg"
        
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=kokoro_speech_{int(time.time())}.{request.output_format}",
                "X-Generation-Time": str(gen_time),
                "X-Engine-Used": "kokoro",
                "X-Service": "kokoro_dedicated"
            }
        )
        
    except Exception as e:
        logging.error(f"[ERROR] Kokoro file synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Kokoro synthesis failed: {str(e)}")

@app.get("/info")
async def service_info():
    """Get Kokoro TTS service information"""
    return {
        "service": "tts_kokoro",
        "engine": "kokoro",
        "port": 8011,
        "speed": "~0.8s per request",
        "quality": "Good",
        "voice": "af_bella (professional female)",
        "best_for": "Real-time conversation",
        "supported_formats": ["wav", "mp3"],
        "max_text_length": 5000,
        "voices": ["af_bella", "af_nicole", "af_sarah", "am_adam", "am_michael"],
        "independent": True,
        "description": "Dedicated Kokoro TTS service for fast speech synthesis"
    }

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    return {
        "service_name": "Kokoro TTS",
        "engine": "kokoro",
        "status": "running" if tts_service else "stopped",
        "ready": tts_service is not None,
        "advantages": [
            "Fast generation (~0.8s)",
            "Low resource usage",
            "Good for real-time conversation",
            "Reliable and stable"
        ],
        "use_cases": [
            "Real-time voice chat",
            "Quick responses",
            "Live demonstrations",
            "Interactive applications"
        ]
    }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the Kokoro TTS service
    uvicorn.run(
        "tts_kokoro_service:app",
        host="0.0.0.0",
        port=8011,
        workers=1,
        log_level="info"
    )
