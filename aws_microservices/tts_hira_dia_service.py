"""
Hira Dia TTS Microservice - Independent Service
High-quality TTS using Nari Dia engine only
Port: 8012
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

app = FastAPI(title="Hira Dia TTS Microservice", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global TTS instance - Hira Dia only
tts_service = None

class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    speed: Optional[float] = 1.0
    output_format: Optional[str] = "wav"
    return_audio: Optional[bool] = True
    high_quality: Optional[bool] = True  # Always high quality for Hira Dia

class SynthesizeResponse(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    metadata: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize Hira Dia TTS service on startup"""
    global tts_service
    logging.info("[TTS] Initializing Hira Dia TTS Microservice...")
    
    try:
        # Check GPU availability (required for Hira Dia)
        if not torch.cuda.is_available():
            raise Exception("GPU required for Hira Dia TTS - no CUDA available")
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"[GPU] GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Initialize TTS manager with Hira Dia only
        tts_service = EnhancedTTSManager()
        
        # Initialize only Nari Dia engine
        await tts_service.initialize_engines(
            load_kokoro=False,  # Explicitly disable Kokoro
            load_nari=True
        )
        
        # Force set to Nari Dia engine
        tts_service.set_engine(TTSEngine.NARI_DIA)
        
        logging.info("[OK] Hira Dia TTS Microservice ready!")
        
    except Exception as e:
        logging.error(f"[ERROR] Hira Dia TTS initialization failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global tts_service
    if tts_service:
        try:
            tts_service.cleanup()
            # GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            tts_service = None
            logging.info("🛑 Hira Dia TTS Microservice shutdown complete")
        except Exception as e:
            logging.error(f"⚠️ Cleanup warning: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_memory = None
    
    if gpu_available:
        gpu_memory = {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    return {
        "status": "healthy",
        "service": "tts_hira_dia",
        "engine": "nari_dia",
        "timestamp": time.time(),
        "ready": tts_service is not None,
        "gpu_available": gpu_available,
        "gpu_memory": gpu_memory,
        "performance": "~3+ minutes generation time",
        "quality": "Maximum",
        "best_for": "highest quality output"
    }

@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: SynthesizeRequest) -> SynthesizeResponse:
    """
    Synthesize speech using Hira Dia (Nari Dia) TTS engine
    """
    if not tts_service:
        raise HTTPException(status_code=503, detail="Hira Dia TTS service not ready")
    
    start_time = time.time()
    
    try:
        # Validate input
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        if len(request.text) > 3000:  # Smaller limit for high-quality synthesis
            raise HTTPException(status_code=400, detail="Text too long for Hira Dia (max 3000 characters)")
        
        # Generate speech using Hira Dia
        audio_bytes, gen_time, used_engine = await tts_service.generate_speech(
            text=request.text,
            engine=TTSEngine.NARI_DIA  # Force Nari Dia
        )
        
        total_time = time.time() - start_time
        
        # Prepare response
        metadata = {
            "processing_time_seconds": round(total_time, 3),
            "generation_time_seconds": round(gen_time, 3),
            "engine_used": "nari_dia",
            "text_length": len(request.text),
            "audio_size_bytes": len(audio_bytes),
            "estimated_duration_seconds": len(audio_bytes) / 32000,
            "voice": request.voice,
            "quality": "maximum",
            "service": "hira_dia_dedicated"
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
        logging.error(f"[ERROR] Hira Dia synthesis failed after {processing_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Hira Dia synthesis failed: {str(e)}")

@app.post("/synthesize_file")
async def synthesize_to_file(request: SynthesizeRequest):
    """
    Synthesize speech and return as audio file
    """
    if not tts_service:
        raise HTTPException(status_code=503, detail="Hira Dia TTS service not ready")
    
    try:
        # Generate audio using Hira Dia
        audio_bytes, gen_time, used_engine = await tts_service.generate_speech(
            text=request.text,
            engine=TTSEngine.NARI_DIA
        )
        
        # Return as audio file
        media_type = "audio/wav" if request.output_format == "wav" else "audio/mpeg"
        
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=hira_dia_speech_{int(time.time())}.{request.output_format}",
                "X-Generation-Time": str(gen_time),
                "X-Engine-Used": "nari_dia",
                "X-Service": "hira_dia_dedicated",
                "X-Quality": "maximum"
            }
        )
        
    except Exception as e:
        logging.error(f"[ERROR] Hira Dia file synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hira Dia synthesis failed: {str(e)}")

@app.get("/info")
async def service_info():
    """Get Hira Dia TTS service information"""
    return {
        "service": "tts_hira_dia",
        "engine": "nari_dia",
        "port": 8012,
        "speed": "~3+ minutes per request",
        "quality": "Maximum",
        "voice": "Adaptive dialogue-focused",
        "best_for": "Highest quality output",
        "supported_formats": ["wav", "mp3"],
        "max_text_length": 3000,
        "gpu_required": True,
        "independent": True,
        "description": "Dedicated Hira Dia TTS service for maximum quality speech synthesis"
    }

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3
        }
    
    return {
        "service_name": "Hira Dia TTS",
        "engine": "nari_dia",
        "status": "running" if tts_service else "stopped",
        "ready": tts_service is not None,
        "gpu_info": gpu_info,
        "advantages": [
            "Maximum quality output",
            "Adaptive dialogue-focused",
            "Professional-grade synthesis",
            "Advanced neural processing"
        ],
        "use_cases": [
            "High-quality content creation",
            "Professional presentations",
            "Audio book narration",
            "Premium voice applications"
        ],
        "requirements": [
            "GPU required",
            "Higher memory usage",
            "Longer processing time",
            "Professional use cases"
        ]
    }

@app.get("/gpu_status")
async def get_gpu_status():
    """Get detailed GPU status for Hira Dia"""
    if not torch.cuda.is_available():
        return {"gpu_available": False, "error": "No CUDA GPU available"}
    
    return {
        "gpu_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "allocated_memory_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_memory_gb": torch.cuda.memory_reserved() / 1024**3,
        "free_memory_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3,
        "compute_capability": torch.cuda.get_device_capability(0),
        "device_count": torch.cuda.device_count()
    }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the Hira Dia TTS service
    uvicorn.run(
        "tts_hira_dia_service:app",
        host="0.0.0.0",
        port=8012,
        workers=1,
        log_level="info"
    )
