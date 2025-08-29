"""
TTS Microservice for AWS Deployment
Runs on high-GPU instances (p3.2xlarge or g4dn.xlarge)
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
from voicebot_orchestrator.enhanced_tts_manager import EnhancedTTSManager, TTSEngine

app = FastAPI(title="TTS Microservice", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global TTS instance
tts_service = None

class SynthesizeRequest(BaseModel):
    text: str
    engine: Optional[str] = "auto"  # "kokoro", "nari_dia", "auto"
    voice: Optional[str] = "af_bella"
    speed: Optional[float] = 1.0
    output_format: Optional[str] = "wav"
    return_audio: Optional[bool] = True  # Return audio data or just metadata

class SynthesizeResponse(BaseModel):
    audio_base64: Optional[str] = None  # Base64 encoded audio
    audio_url: Optional[str] = None     # If saved to S3/storage
    metadata: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize TTS service on startup"""
    global tts_service
    logging.info("🎭 Initializing TTS Microservice...")
    
    try:
        # Initialize TTS manager
        tts_service = EnhancedTTSManager()
        
        # Check GPU availability for Nari Dia
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"🎮 GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Initialize both engines on GPU instances
            await tts_service.initialize_engines(
                load_kokoro=True,
                load_nari=True
            )
        else:
            logging.warning("⚠️ No GPU available, loading Kokoro only")
            # CPU-only fallback
            await tts_service.initialize_engines(
                load_kokoro=True,
                load_nari=False
            )
        
        logging.info("✅ TTS Microservice ready!")
        
    except Exception as e:
        logging.error(f"❌ TTS initialization failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global tts_service
    if tts_service:
        try:
            tts_service.cleanup()
            tts_service = None
            logging.info("🛑 TTS Microservice shutdown complete")
        except Exception as e:
            logging.error(f"⚠️ Cleanup warning: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_memory = None
    available_engines = []
    
    if gpu_available:
        gpu_memory = {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    if tts_service:
        available_engines = [e.value for e in tts_service.get_available_engines()]
    
    return {
        "status": "healthy",
        "service": "tts",
        "timestamp": time.time(),
        "ready": tts_service is not None,
        "available_engines": available_engines,
        "gpu_available": gpu_available,
        "gpu_memory": gpu_memory
    }

@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: SynthesizeRequest) -> SynthesizeResponse:
    """
    Synthesize speech from text
    
    Args:
        request: Synthesis request with text and parameters
        
    Returns:
        Audio data and metadata
    """
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not ready")
    
    start_time = time.time()
    
    try:
        # Validate input
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        if len(request.text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")
        
        # Parse engine parameter
        if request.engine == "auto":
            engine = TTSEngine.AUTO
        elif request.engine == "kokoro":
            engine = TTSEngine.KOKORO
        elif request.engine == "nari_dia":
            engine = TTSEngine.NARI_DIA
        else:
            raise HTTPException(status_code=400, detail=f"Unknown engine: {request.engine}")
        
        # Generate speech
        audio_bytes, gen_time, used_engine = await tts_service.generate_speech(
            text=request.text,
            engine=engine
        )
        
        total_time = time.time() - start_time
        
        # Prepare response
        metadata = {
            "processing_time_seconds": round(total_time, 3),
            "generation_time_seconds": round(gen_time, 3),
            "engine_used": used_engine,
            "text_length": len(request.text),
            "audio_size_bytes": len(audio_bytes),
            "estimated_duration_seconds": len(audio_bytes) / 32000  # Rough estimate
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
        logging.error(f"❌ Synthesis failed after {processing_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.post("/synthesize_file")
async def synthesize_to_file(request: SynthesizeRequest):
    """
    Synthesize speech and return as audio file
    """
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not ready")
    
    try:
        # Parse engine
        if request.engine == "auto":
            engine = TTSEngine.AUTO
        elif request.engine == "kokoro":
            engine = TTSEngine.KOKORO
        elif request.engine == "nari_dia":
            engine = TTSEngine.NARI_DIA
        else:
            engine = TTSEngine.KOKORO  # Default fallback
        
        # Generate audio
        audio_bytes, gen_time, used_engine = await tts_service.generate_speech(
            text=request.text,
            engine=engine
        )
        
        # Return as audio file
        media_type = "audio/wav" if request.output_format == "wav" else "audio/mpeg"
        
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech_{int(time.time())}.{request.output_format}",
                "X-Generation-Time": str(gen_time),
                "X-Engine-Used": used_engine
            }
        )
        
    except Exception as e:
        logging.error(f"❌ File synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.post("/switch_engine")
async def switch_engine(engine: str):
    """Switch TTS engine"""
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not ready")
    
    try:
        if engine == "kokoro":
            tts_service.set_engine(TTSEngine.KOKORO)
        elif engine == "nari_dia":
            tts_service.set_engine(TTSEngine.NARI_DIA)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")
        
        return {
            "engine_switched": True,
            "current_engine": tts_service.get_current_engine().value,
            "available_engines": [e.value for e in tts_service.get_available_engines()]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine switch failed: {str(e)}")

@app.get("/engines")
async def get_engines():
    """Get available TTS engines"""
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not ready")
    
    available_engines = tts_service.get_available_engines()
    current_engine = tts_service.get_current_engine()
    
    engines_info = []
    for engine in available_engines:
        info = {
            "name": engine.value,
            "active": engine == current_engine,
        }
        
        if engine == TTSEngine.KOKORO:
            info.update({
                "speed": "~0.8s generation",
                "quality": "Good",
                "voice": "af_bella (professional female)",
                "best_for": "Real-time conversation"
            })
        elif engine == TTSEngine.NARI_DIA:
            info.update({
                "speed": "~3+ minutes generation", 
                "quality": "Maximum",
                "voice": "Adaptive dialogue-focused",
                "best_for": "Highest quality output"
            })
        
        engines_info.append(info)
    
    return {
        "current_engine": current_engine.value,
        "available_engines": engines_info
    }

@app.get("/info")
async def service_info():
    """Get TTS service information"""
    gpu_available = torch.cuda.is_available()
    
    return {
        "service": "tts",
        "engines": ["kokoro", "nari_dia", "auto"],
        "supported_formats": ["wav", "mp3"],
        "max_text_length": 5000,
        "voices": ["af_bella", "af_nicole", "af_sarah", "am_adam", "am_michael"],
        "gpu_available": gpu_available,
        "gpu_recommended": True,
        "performance": {
            "kokoro": "~0.8s per request",
            "nari_dia": "~3+ minutes per request"
        }
    }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the service
    uvicorn.run(
        "tts_service:app",
        host="0.0.0.0",
        port=8003,
        workers=1,  # Single worker for GPU models
        log_level="info"
    )
