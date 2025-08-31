"""
Dia-1.6B-4bit TTS Microservice - Independent Service
High-quality TTS using Dia-1.6B-4bit engine only
Port: 8013
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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from voicebot_orchestrator.enhanced_tts_manager import EnhancedTTSManager, TTSEngine

app = FastAPI(title="Dia-1.6B-4bit TTS Microservice", version="1.0.0-4bit")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts_service = None

class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    speed: Optional[float] = 1.0
    output_format: Optional[str] = "wav"
    return_audio: Optional[bool] = True
    high_quality: Optional[bool] = False  # 4bit is optimized for speed/size

class SynthesizeResponse(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    metadata: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    global tts_service
    logging.info("[TTS] Initializing Dia-1.6B-4bit TTS Microservice...")
    try:
        if not torch.cuda.is_available():
            raise Exception("GPU required for Dia-1.6B-4bit TTS - no CUDA available")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"[GPU] GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
        tts_service = EnhancedTTSManager()
        # Initialize only Dia-1.6B-4bit engine (custom logic needed in EnhancedTTSManager)
        await tts_service.initialize_engines(
            load_kokoro=False,
            load_nari=False,
            load_dia_4bit=True  # You will need to implement this in EnhancedTTSManager
        )
        tts_service.set_engine(TTSEngine.DIA_4BIT)  # You will need to add this enum value
        logging.info("[OK] Dia-1.6B-4bit TTS Microservice ready!")
    except Exception as e:
        logging.error(f"[ERROR] Dia-1.6B-4bit TTS initialization failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    global tts_service
    if tts_service:
        try:
            tts_service.cleanup()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            tts_service = None
            logging.info("🛑 Dia-1.6B-4bit TTS Microservice shutdown complete")
        except Exception as e:
            logging.error(f"⚠️ Cleanup warning: {e}")

@app.get("/health")
async def health_check():
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
        "service": "tts_dia_4bit",
        "engine": "dia_1.6b_4bit",
        "timestamp": time.time(),
        "ready": tts_service is not None,
        "gpu_available": gpu_available,
        "gpu_memory": gpu_memory,
        "performance": "Faster, smaller model",
        "quality": "Optimized",
        "best_for": "speed and lower memory usage"
    }

@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: SynthesizeRequest) -> SynthesizeResponse:
    if not tts_service:
        raise HTTPException(status_code=503, detail="Dia-1.6B-4bit TTS service not ready")
    start_time = time.time()
    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        if len(request.text) > 3000:
            raise HTTPException(status_code=400, detail="Text too long for Dia-1.6B-4bit (max 3000 characters)")
        audio_bytes, gen_time, used_engine = await tts_service.generate_speech(
            text=request.text,
            engine=TTSEngine.DIA_4BIT  # You will need to add this enum value
        )
        total_time = time.time() - start_time
        metadata = {
            "processing_time_seconds": round(total_time, 3),
            "generation_time_seconds": round(gen_time, 3),
            "engine_used": "dia_1.6b_4bit",
            "text_length": len(request.text),
            "audio_size_bytes": len(audio_bytes),
            "estimated_duration_seconds": len(audio_bytes) / 32000,
            "voice": request.voice,
            "quality": "optimized",
            "service": "dia_4bit_dedicated"
        }
        audio_base64 = None
        if request.return_audio:
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return SynthesizeResponse(
            audio_base64=audio_base64,
            metadata=metadata
        )
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"[ERROR] Dia-1.6B-4bit synthesis failed after {processing_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Dia-1.6B-4bit synthesis failed: {str(e)}")

@app.post("/synthesize_file")
async def synthesize_to_file(request: SynthesizeRequest):
    if not tts_service:
        raise HTTPException(status_code=503, detail="Dia-1.6B-4bit TTS service not ready")
    try:
        audio_bytes, gen_time, used_engine = await tts_service.generate_speech(
            text=request.text,
            engine=TTSEngine.DIA_4BIT
        )
        media_type = "audio/wav" if request.output_format == "wav" else "audio/mpeg"
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=dia_4bit_speech_{int(time.time())}.{request.output_format}",
                "X-Generation-Time": str(gen_time),
                "X-Engine-Used": "dia_1.6b_4bit",
                "X-Service": "dia_4bit_dedicated",
                "X-Quality": "optimized"
            }
        )
    except Exception as e:
        logging.error(f"[ERROR] Dia-1.6B-4bit file synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dia-1.6B-4bit synthesis failed: {str(e)}")

@app.get("/info")
async def service_info():
    return {
        "service": "tts_dia_4bit",
        "engine": "dia_1.6b_4bit",
        "port": 8013,
        "speed": "Faster, smaller model",
        "quality": "Optimized",
        "voice": "Adaptive dialogue-focused",
        "best_for": "Speed and lower memory usage",
        "supported_formats": ["wav", "mp3"],
        "max_text_length": 3000,
        "gpu_required": True,
        "independent": True,
        "description": "Dedicated Dia-1.6B-4bit TTS service for optimized speech synthesis"
    }

@app.get("/status")
async def get_status():
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3
        }
    return {
        "service_name": "Dia-1.6B-4bit TTS",
        "engine": "dia_1.6b_4bit",
        "status": "running" if tts_service else "stopped",
        "ready": tts_service is not None,
        "gpu_info": gpu_info,
        "advantages": [
            "Optimized for speed and memory",
            "Adaptive dialogue-focused",
            "Professional-grade synthesis",
            "Advanced neural processing"
        ],
        "use_cases": [
            "Fast content creation",
            "Professional presentations",
            "Audio book narration",
            "Premium voice applications"
        ],
        "requirements": [
            "GPU required",
            "Lower memory usage",
            "Faster processing time",
            "Professional use cases"
        ]
    }

@app.get("/gpu_status")
async def get_gpu_status():
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    uvicorn.run(
        "tts_dia_4bit_service:app",
        host="0.0.0.0",
        port=8013,
        workers=1,
        log_level="info"
    )
