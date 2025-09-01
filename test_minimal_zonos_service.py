"""
Minimal Zonos TTS Service - Debugging Version
"""
import asyncio
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    emotion: Optional[str] = "neutral"
    seed: Optional[int] = None

# Global TTS instance
tts_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_engine
    logger.info("[STARTUP] Initializing minimal Zonos TTS service...")
    
    try:
        # Import and initialize TTS
        import sys
        import os
        sys.path.append('.')
        
        from voicebot_orchestrator.zonos_tts import ZonosTTS
        
        logger.info("[TTS] Creating ZonosTTS instance...")
        tts_engine = ZonosTTS(voice="default", model="zonos-v1")
        logger.info("[TTS] ZonosTTS initialized successfully")
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize TTS: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    logger.info("[STARTUP] Minimal Zonos TTS service ready!")
    yield
    
    logger.info("[SHUTDOWN] Cleaning up minimal Zonos TTS service...")

# Create FastAPI app
app = FastAPI(
    title="Minimal Zonos TTS Service",
    description="Debugging version of Zonos TTS",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    """Health check endpoint"""
    logger.info("[HEALTH] Health check requested")
    return {
        "status": "healthy",
        "service": "minimal-zonos-tts",
        "version": "1.0.0",
        "tts_ready": tts_engine is not None
    }

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech from text"""
    logger.info(f"[SYNTHESIZE] Request: {request.text[:50]}...")
    
    if not tts_engine:
        raise HTTPException(status_code=503, detail="TTS engine not initialized")
    
    try:
        # Synthesize audio
        audio_bytes = await tts_engine.synthesize_speech(
            text=request.text,
            voice=request.voice,
            emotion=request.emotion,
            seed=request.seed
        )
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        logger.info(f"[SYNTHESIZE] Generated {len(audio_bytes)} bytes of audio")
        
        return {
            "success": True,
            "audio_base64": audio_base64,
            "size_bytes": len(audio_bytes),
            "voice": request.voice,
            "emotion": request.emotion
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.get("/voices")
async def list_voices():
    """List available voices"""
    if not tts_engine:
        raise HTTPException(status_code=503, detail="TTS engine not initialized")
    
    return {
        "voices": tts_engine.list_voices(),
        "models": tts_engine.list_models()
    }

def main():
    """Run the minimal service"""
    logger.info("Starting minimal Zonos TTS service on port 8014...")
    
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0", 
        port=8014,
        log_level="info",
        access_log=True,
        loop="asyncio"
    )
    
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    main()
