"""
Unified Hira Dia TTS Microservice - Dual Mode Support
Supports both Full Dia (quality) and Dia-4bit (speed) engines
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
from typing import Dict, Any, Optional, Literal
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from voicebot_orchestrator.enhanced_tts_manager import EnhancedTTSManager, TTSEngine

app = FastAPI(title="Unified Hira Dia TTS Microservice", version="2.0.0-unified")

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
    high_quality: Optional[bool] = True  # True = Full Dia, False = 4-bit Dia
    engine_preference: Optional[Literal["full", "4bit", "auto"]] = "auto"

class SynthesizeResponse(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    metadata: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    global tts_service
    logging.info("[TTS] Initializing Unified Hira Dia TTS Microservice...")
    try:
        if not torch.cuda.is_available():
            raise Exception("GPU required for Dia TTS - no CUDA available")
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"[GPU] GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
        
        tts_service = EnhancedTTSManager()
        
        # Initialize both Dia engines (will need to modify EnhancedTTSManager)
        await tts_service.initialize_engines(
            load_kokoro=False,
            load_nari=True,        # Full Dia model
            load_dia_4bit=True     # 4-bit Dia model
        )
        
        # Default to full quality
        tts_service.set_engine(TTSEngine.NARI_DIA)
        
        logging.info("[OK] Unified Hira Dia TTS Microservice ready!")
        logging.info("    ✅ Full Dia (NARI_DIA) - Maximum quality")
        logging.info("    ✅ 4-bit Dia (DIA_4BIT) - Speed optimized")
        
    except Exception as e:
        logging.error(f"[ERROR] Unified Dia TTS initialization failed: {e}")
        raise

def _select_engine(request: SynthesizeRequest) -> TTSEngine:
    """Smart engine selection based on request parameters"""
    if request.engine_preference == "full":
        return TTSEngine.NARI_DIA
    elif request.engine_preference == "4bit":
        return TTSEngine.DIA_4BIT
    elif request.engine_preference == "auto":
        # Auto-selection logic
        text_length = len(request.text)
        
        # High quality requested - use full model
        if request.high_quality:
            return TTSEngine.NARI_DIA
        
        # Speed optimization for shorter text
        if text_length < 100:
            return TTSEngine.DIA_4BIT
        elif text_length < 300:
            return TTSEngine.DIA_4BIT if not request.high_quality else TTSEngine.NARI_DIA
        else:
            # For longer text, consider speed vs quality tradeoff
            return TTSEngine.NARI_DIA if request.high_quality else TTSEngine.DIA_4BIT
    
    # Default fallback
    return TTSEngine.NARI_DIA if request.high_quality else TTSEngine.DIA_4BIT

@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: SynthesizeRequest) -> SynthesizeResponse:
    if not tts_service:
        raise HTTPException(status_code=503, detail="Unified Dia TTS service not ready")
    
    start_time = time.time()
    
    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        if len(request.text) > 3000:
            raise HTTPException(status_code=400, detail="Text too long for Dia TTS (max 3000 characters)")
        
        # Smart engine selection
        selected_engine = _select_engine(request)
        
        # Validate engine availability
        available_engines = tts_service.get_available_engines()
        if selected_engine not in available_engines:
            # Fallback to available engine
            if TTSEngine.NARI_DIA in available_engines:
                selected_engine = TTSEngine.NARI_DIA
            elif TTSEngine.DIA_4BIT in available_engines:
                selected_engine = TTSEngine.DIA_4BIT
            else:
                raise HTTPException(status_code=503, detail="No Dia engines available")
        
        logging.info(f"[ENGINE] Selected: {selected_engine.value}")
        
        # Generate speech
        audio_bytes, gen_time, used_engine = await tts_service.generate_speech(
            text=request.text,
            engine=selected_engine
        )
        
        total_time = time.time() - start_time
        
        # Determine quality and performance metrics
        if selected_engine == TTSEngine.NARI_DIA:
            quality_level = "maximum"
            performance_note = "High-quality synthesis"
        else:  # DIA_4BIT
            quality_level = "optimized"
            performance_note = "Speed-optimized synthesis"
        
        metadata = {
            "processing_time_seconds": round(total_time, 3),
            "generation_time_seconds": round(gen_time, 3),
            "engine_used": used_engine.lower().replace("_", "-"),
            "engine_selected": selected_engine.value,
            "text_length": len(request.text),
            "audio_size_bytes": len(audio_bytes),
            "estimated_duration_seconds": len(audio_bytes) / 32000,
            "voice": request.voice,
            "quality": quality_level,
            "performance": performance_note,
            "service": "unified_hira_dia",
            "selection_method": request.engine_preference
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
        logging.error(f"[ERROR] Unified Dia synthesis failed after {processing_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Unified Dia synthesis failed: {str(e)}")

@app.post("/switch_engine")
async def switch_engine(engine: Literal["full", "4bit", "nari_dia", "dia_4bit"]):
    """Switch between Dia engines"""
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not ready")
    
    try:
        if engine in ["full", "nari_dia"]:
            tts_service.set_engine(TTSEngine.NARI_DIA)
            current_engine = "full_dia"
        elif engine in ["4bit", "dia_4bit"]:
            tts_service.set_engine(TTSEngine.DIA_4BIT)
            current_engine = "4bit_dia"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")
        
        return {
            "engine_switched": True,
            "current_engine": current_engine,
            "available_engines": [e.value for e in tts_service.get_available_engines()],
            "description": "Full Dia (quality) vs 4-bit Dia (speed)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine switch failed: {str(e)}")

@app.get("/engines")
async def get_engines():
    """Get available Dia engines"""
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not ready")
    
    available_engines = tts_service.get_available_engines()
    current_engine = tts_service.get_current_engine()
    
    engines_info = []
    for engine in available_engines:
        if engine in [TTSEngine.NARI_DIA, TTSEngine.DIA_4BIT]:
            info = {
                "name": engine.value,
                "active": engine == current_engine,
            }
            
            if engine == TTSEngine.NARI_DIA:
                info.update({
                    "display_name": "Full Dia (Quality)",
                    "speed": "~3+ minutes generation",
                    "quality": "Maximum",
                    "voice": "Adaptive dialogue-focused",
                    "best_for": "Highest quality output",
                    "memory_usage": "Higher"
                })
            elif engine == TTSEngine.DIA_4BIT:
                info.update({
                    "display_name": "4-bit Dia (Speed)",
                    "speed": "~30-60 seconds generation",
                    "quality": "Optimized",
                    "voice": "Adaptive dialogue-focused",
                    "best_for": "Speed and lower memory usage",
                    "memory_usage": "Lower"
                })
            
            engines_info.append(info)
    
    return {
        "service": "unified_hira_dia",
        "current_engine": current_engine.value if current_engine else None,
        "available_engines": engines_info,
        "recommendation": "Use 'full' for quality, '4bit' for speed, 'auto' for smart selection"
    }

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
    
    available_engines = []
    if tts_service:
        available_engines = [e.value for e in tts_service.get_available_engines()]
    
    return {
        "status": "healthy",
        "service": "unified_hira_dia",
        "engines_available": available_engines,
        "timestamp": time.time(),
        "ready": tts_service is not None,
        "gpu_available": gpu_available,
        "gpu_memory": gpu_memory,
        "capabilities": {
            "full_dia": "Maximum quality, slower processing",
            "4bit_dia": "Optimized quality, faster processing",
            "auto_selection": "Smart engine selection based on request"
        }
    }

@app.get("/info")
async def service_info():
    return {
        "service": "unified_hira_dia",
        "version": "2.0.0-unified",
        "port": 8012,
        "engines": {
            "full_dia": {
                "quality": "Maximum",
                "speed": "~3+ minutes",
                "memory": "Higher usage"
            },
            "4bit_dia": {
                "quality": "Optimized", 
                "speed": "~30-60 seconds",
                "memory": "Lower usage"
            }
        },
        "features": [
            "Dual engine support",
            "Auto engine selection",
            "Quality/speed optimization",
            "Runtime engine switching"
        ],
        "supported_formats": ["wav", "mp3"],
        "max_text_length": 3000,
        "gpu_required": True,
        "description": "Unified Dia TTS service supporting both quality and speed optimized engines"
    }

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
            logging.info("🛑 Unified Hira Dia TTS Microservice shutdown complete")
        except Exception as e:
            logging.error(f"⚠️ Cleanup warning: {e}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    uvicorn.run(
        "tts_hira_dia_unified_service:app",
        host="0.0.0.0",
        port=8012,
        workers=1,
        log_level="info"
    )
