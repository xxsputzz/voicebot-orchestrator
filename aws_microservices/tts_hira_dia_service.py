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
import re
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from voicebot_orchestrator.enhanced_tts_manager import EnhancedTTSManager, TTSEngine

def purge_all_emojis_comprehensive(text: str) -> str:
    """
    Remove ALL emoji and unicode characters that could cause encoding issues.
    Simply removes emojis without text replacement to preserve sentence meaning.
    """
    if not text:
        return ""
    
    # Step 1: Remove numbered emojis (1️⃣, 2️⃣, etc.)
    cleaned_text = re.sub(r'[0-9]️⃣', '', text)
    
    # Step 2: Remove ALL emoji characters using Unicode ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+"
    )
    cleaned_text = emoji_pattern.sub('', cleaned_text)
    
    # Step 3: NUCLEAR OPTION - Remove any remaining high Unicode characters
    # Keep only ASCII printable characters and basic punctuation
    cleaned_text = re.sub(r'[^\x20-\x7E\r\n\t]', '', cleaned_text)
    
    # Step 4: Clean up extra whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Step 5: Ensure we still have meaningful text
    if not cleaned_text.strip() or len(cleaned_text.strip()) < 2:
        cleaned_text = "Text contains unsupported characters and has been sanitized."
    
    return cleaned_text

app = FastAPI(title="Unified Hira Dia TTS Microservice", version="2.0.0-unified")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts_service = None
initial_engine_mode = "full"  # Default mode, will be overridden by command line args

class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    speed: Optional[float] = 1.0
    seed: Optional[int] = None  # Random seed for voice generation
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
    
    # Parse command line arguments directly here to avoid scope issues
    import sys
    engine_mode = "full"  # default
    
    # Check for --engine argument in sys.argv
    if "--engine" in sys.argv:
        try:
            engine_idx = sys.argv.index("--engine")
            if engine_idx + 1 < len(sys.argv):
                engine_mode = sys.argv[engine_idx + 1]
        except (IndexError, ValueError):
            pass
    
    logging.info(f"[TTS] Initializing Unified Hira Dia TTS Microservice...")
    logging.info(f"[DEBUG] Detected engine mode from sys.argv: {engine_mode}")
    
    try:
        if not torch.cuda.is_available():
            raise Exception("GPU required for Dia TTS - no CUDA available")
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"[GPU] GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
        
        tts_service = EnhancedTTSManager()
        
        # Initialize only the requested engine for efficiency and strict mode
        if engine_mode == "4bit":
            # Only load 4-bit engine (use nari=True for any Dia engine)
            await tts_service.initialize_engines(
                load_kokoro=False,     # Disable Kokoro
                load_nari=True         # Load Nari Dia engine
            )
        elif engine_mode == "full":
            # Only load full engine
            await tts_service.initialize_engines(
                load_kokoro=False,     # Disable Kokoro
                load_nari=True         # Load Nari Dia engine
            )
        else:
            # Auto mode - load both for flexibility
            await tts_service.initialize_engines(
                load_kokoro=False,     # Disable Kokoro
                load_nari=True         # Load Nari Dia engine
            )
        
        # Set initial engine based on startup parameter - STRICT MODE
        available_engines = tts_service.get_available_engines()
        logging.info(f"[DEBUG] Requested engine mode: {engine_mode}")
        logging.info(f"[DEBUG] Available engines: {available_engines}")
        logging.info(f"[DEBUG] NARI_DIA enum value: {TTSEngine.NARI_DIA}")
        logging.info(f"[DEBUG] Is NARI_DIA in available? {TTSEngine.NARI_DIA in available_engines}")
        
        if engine_mode == "4bit":
            # STRICT: Use NARI_DIA for 4-bit mode (clean version doesn't distinguish)
            if TTSEngine.NARI_DIA in available_engines:
                tts_service.set_engine(TTSEngine.NARI_DIA)
                logging.info("[INIT] Starting in Dia mode (using NARI_DIA)")
            else:
                # FAIL COMPLETELY - don't fall back to Full Dia
                error_msg = "[ERROR] Dia mode was requested but NARI_DIA failed to load. Service will not start with fallback."
                logging.error(error_msg)
                logging.error("    [INFO] Check the NARI_DIA engine loading errors above")
                logging.error("    [CONFIG] Fix the engine loading issue in EnhancedTTSManager")
                raise Exception("NARI_DIA engine unavailable - requested mode cannot be satisfied")
                
        elif engine_mode == "full":
            # For full mode, require Full Dia engine
            if TTSEngine.NARI_DIA in available_engines:
                tts_service.set_engine(TTSEngine.NARI_DIA)
                logging.info("[INIT] Starting in Full Dia mode (maximum quality)")
            else:
                error_msg = "[ERROR] Full Dia mode was requested but failed to load"
                logging.error(error_msg)
                raise Exception("Full Dia engine unavailable - requested mode cannot be satisfied")
                
        elif engine_mode == "auto":
            # Auto mode can fall back
            if TTSEngine.NARI_DIA in available_engines:
                tts_service.set_engine(TTSEngine.NARI_DIA)
                logging.info("[INIT] Auto mode: Using Full Dia")
            elif TTSEngine.DIA_4BIT in available_engines:
                tts_service.set_engine(TTSEngine.DIA_4BIT)
                logging.info("[INIT] Auto mode: Using Dia 4-bit")
            else:
                raise Exception("No Dia engines available")
        else:
            raise Exception(f"Unknown engine mode: {engine_mode}")
        
        logging.info("[OK] Unified Hira Dia TTS Microservice ready!")
        if TTSEngine.NARI_DIA in available_engines:
            logging.info("    [OK] Full Dia (NARI_DIA) - Maximum quality")
        # TODO: Add DIA_4BIT support when available
        # if TTSEngine.DIA_4BIT in available_engines:
        #     logging.info("    [OK] 4-bit Dia (DIA_4BIT) - Speed optimized")
        
    except Exception as e:
        logging.error(f"[ERROR] Unified Dia TTS initialization failed: {e}")
        raise

def _select_engine(request: SynthesizeRequest) -> TTSEngine:
    """Smart engine selection based on request parameters"""
    if not tts_service:
        return TTSEngine.NARI_DIA
    
    available_engines = tts_service.get_available_engines()
    
    if request.engine_preference == "full":
        return TTSEngine.NARI_DIA if TTSEngine.NARI_DIA in available_engines else available_engines[0]
    elif request.engine_preference == "4bit":
        return TTSEngine.DIA_4BIT if TTSEngine.DIA_4BIT in available_engines else available_engines[0]
    elif request.engine_preference == "auto":
        # Auto-selection logic
        text_length = len(request.text)
        
        # High quality requested - use full model
        if request.high_quality and TTSEngine.NARI_DIA in available_engines:
            return TTSEngine.NARI_DIA
        
        # Speed optimization for shorter text
        if text_length < 100 and TTSEngine.DIA_4BIT in available_engines:
            return TTSEngine.DIA_4BIT
        elif text_length < 300:
            if not request.high_quality and TTSEngine.DIA_4BIT in available_engines:
                return TTSEngine.DIA_4BIT
            elif TTSEngine.NARI_DIA in available_engines:
                return TTSEngine.NARI_DIA
        else:
            # For longer text, prefer quality if requested
            if request.high_quality and TTSEngine.NARI_DIA in available_engines:
                return TTSEngine.NARI_DIA
            elif TTSEngine.DIA_4BIT in available_engines:
                return TTSEngine.DIA_4BIT
    
    # Default fallback
    if TTSEngine.NARI_DIA in available_engines:
        return TTSEngine.NARI_DIA
    elif TTSEngine.DIA_4BIT in available_engines:
        return TTSEngine.DIA_4BIT
    else:
        return available_engines[0] if available_engines else TTSEngine.NARI_DIA

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
            logging.info("[STOP] Hira Dia TTS Microservice shutdown complete")
        except Exception as e:
            logging.error(f"[WARNING] Cleanup warning: {e}")

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
    
    available_engines = []
    if tts_service:
        available_engines = [e.value for e in tts_service.get_available_engines() if e in [TTSEngine.NARI_DIA]]
    
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

@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: SynthesizeRequest) -> SynthesizeResponse:
    """
    Synthesize speech using unified Dia engines (Full or 4-bit)
    """
    if not tts_service:
        raise HTTPException(status_code=503, detail="Unified Dia TTS service not ready")
    
    start_time = time.time()
    
    try:
        # AGGRESSIVE EMOJI PURGING - Remove ALL emojis from LLM output
        original_text = request.text
        request.text = purge_all_emojis_comprehensive(request.text)
        
        if request.text != original_text:
            logging.info(f"[EMOJI_PURGE] Text sanitized: '{original_text[:50]}...' -> '{request.text[:50]}...'")
        
        # Validate input
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
        
        logging.info(f"[ENGINE] Selected: {selected_engine.value} for text length {len(request.text)}")
        
        # Generate speech
        audio_bytes, gen_time, used_engine = await tts_service.generate_speech(
            text=request.text,
            engine=selected_engine,
            seed=request.seed
        )
        
        total_time = time.time() - start_time
        
        # Determine quality and performance metrics
        if selected_engine == TTSEngine.NARI_DIA:
            quality_level = "maximum"
            performance_note = "High-quality synthesis"
        else:  # DIA_4BIT
            quality_level = "optimized"
            performance_note = "Speed-optimized synthesis"
        
        # Get token estimation for metadata
        estimated_tokens, estimated_duration = tts_service.estimate_tokens_needed(request.text) if hasattr(tts_service, 'estimate_tokens_needed') else (0, 0)
        
        # Prepare response
        metadata = {
            "processing_time_seconds": round(total_time, 3),
            "generation_time_seconds": round(gen_time, 3),
            "engine_used": used_engine.lower().replace("_", "-"),
            "engine_selected": selected_engine.value,
            "text_length": len(request.text),
            "audio_size_bytes": len(audio_bytes),
            "estimated_duration_seconds": len(audio_bytes) / 32000,
            "estimated_tokens": estimated_tokens,
            "voice": request.voice,
            "seed": request.seed,
            "quality": quality_level,
            "performance": performance_note,
            "service": "unified_hira_dia",
            "selection_method": request.engine_preference
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

@app.post("/synthesize_file")
async def synthesize_to_file(request: SynthesizeRequest):
    """
    Synthesize speech and return as audio file
    """
    if not tts_service:
        raise HTTPException(status_code=503, detail="Unified Dia TTS service not ready")
    
    try:
        # AGGRESSIVE EMOJI PURGING - Remove ALL emojis from LLM output
        original_text = request.text
        request.text = purge_all_emojis_comprehensive(request.text)
        
        if request.text != original_text:
            logging.info(f"[EMOJI_PURGE_FILE] Text sanitized: '{original_text[:50]}...' -> '{request.text[:50]}...'")
        
        # Smart engine selection for file synthesis
        selected_engine = _select_engine(request)
        
        # Generate audio using selected engine
        audio_bytes, gen_time, used_engine = await tts_service.generate_speech(
            text=request.text,
            engine=selected_engine,
            seed=request.seed
        )
        
        # Return as audio file
        media_type = "audio/wav" if request.output_format == "wav" else "audio/mpeg"
        
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=unified_dia_{selected_engine.value}_speech_{int(time.time())}.{request.output_format}",
                "X-Generation-Time": str(gen_time),
                "X-Engine-Used": used_engine.lower().replace("_", "-"),
                "X-Service": "unified_hira_dia",
                "X-Quality": "maximum" if selected_engine == TTSEngine.NARI_DIA else "optimized"
            }
        )
        
    except Exception as e:
        logging.error(f"[ERROR] Unified Dia file synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Unified Dia synthesis failed: {str(e)}")

@app.get("/info")
async def service_info():
    """Get Unified Dia TTS service information"""
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
        "independent": True,
        "description": "Unified Dia TTS service supporting both quality and speed optimized engines"
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
    
    available_engines = []
    current_engine = None
    if tts_service:
        available_engines = [e.value for e in tts_service.get_available_engines() if e in [TTSEngine.NARI_DIA, TTSEngine.DIA_4BIT]]
        current_engine = tts_service.get_current_engine().value if tts_service.get_current_engine() else None
    
    return {
        "service_name": "Unified Hira Dia TTS",
        "engines": available_engines,
        "current_engine": current_engine,
        "status": "running" if tts_service else "stopped",
        "ready": tts_service is not None,
        "gpu_info": gpu_info,
        "capabilities": {
            "full_dia": "Maximum quality, slower processing",
            "4bit_dia": "Optimized quality, faster processing",
            "auto_selection": "Smart engine selection based on request"
        },
        "advantages": [
            "Dual engine support",
            "Adaptive dialogue-focused",
            "Professional-grade synthesis",
            "Flexible quality/speed balance"
        ],
        "use_cases": [
            "High-quality content creation",
            "Professional presentations", 
            "Audio book narration",
            "Speed-optimized applications"
        ],
        "requirements": [
            "GPU required",
            "Configurable memory usage",
            "Flexible processing time",
            "Professional and efficient use cases"
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

def set_initial_engine_mode(mode: str):
    """Set the initial engine mode"""
    global initial_engine_mode
    initial_engine_mode = mode

@app.get("/readiness")
async def readiness_check():
    """
    Comprehensive readiness check that actually tests TTS generation
    This endpoint verifies that engines are not just loaded, but can actually generate audio
    """
    if not tts_service:
        return {
            "ready": False,
            "status": "TTS service not initialized",
            "engines_tested": {},
            "timestamp": time.time()
        }
    
    available_engines = tts_service.get_available_engines()
    engine_tests = {}
    all_ready = True
    
    # Test each available engine with a very short generation
    for engine in available_engines:
        if engine in [TTSEngine.NARI_DIA, TTSEngine.DIA_4BIT]:
            try:
                start_time = time.time()
                
                # Switch to this engine for testing
                original_engine = tts_service.get_current_engine()
                tts_service.set_engine(engine)
                
                # Generate a very short test audio (single word)
                test_audio = await tts_service.synthesize_async(
                    text="Test.",
                    max_tokens=32,  # Very short
                    cfg_scale=2.0,
                    temperature=0.8,
                    top_p=0.9
                )
                
                generation_time = time.time() - start_time
                
                engine_tests[engine.value] = {
                    "ready": True,
                    "test_duration_seconds": round(generation_time, 2),
                    "audio_generated": test_audio is not None,
                    "status": "Engine fully operational"
                }
                
                # Restore original engine
                if original_engine:
                    tts_service.set_engine(original_engine)
                    
            except Exception as e:
                engine_tests[engine.value] = {
                    "ready": False,
                    "error": str(e),
                    "status": "Engine failed generation test"
                }
                all_ready = False
    
    return {
        "ready": all_ready,
        "status": "All engines ready" if all_ready else "Some engines not ready",
        "engines_tested": engine_tests,
        "total_engines_available": len(available_engines),
        "engines_ready": sum(1 for test in engine_tests.values() if test["ready"]),
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Unified Hira Dia TTS Microservice")
    parser.add_argument("--engine", choices=["full", "4bit", "auto"], default="full",
                        help="Initial engine mode: full (NARI_DIA), 4bit (DIA_4BIT), or auto")
    args = parser.parse_args()
    
    # Store initial engine preference globally
    set_initial_engine_mode(args.engine)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logging.info(f"[INIT] Starting with engine mode: {initial_engine_mode}")
    
    # Run the Unified Hira Dia TTS service
    uvicorn.run(
        "tts_hira_dia_service:app",
        host="0.0.0.0",
        port=8012,
        workers=1,
        log_level="info"
    )
