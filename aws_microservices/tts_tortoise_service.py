#!/usr/bin/env python3
"""
Neural Tortoise TTS Service
Real neural network TTS with GPU acceleration, unlimited timeout, and 29 voices
"""

import asyncio
import signal
import sys
import threading
import time
import atexit
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn

# Import GPU management system
from tortoise_gpu_manager import get_gpu_manager, cleanup_tortoise_gpu, emergency_gpu_cleanup

# Import timeout management system
from tortoise_timeout_config import get_timeout_manager

# Global variables
tts_service = None
force_gpu_mode = False  # Global flag for GPU forcing
shutdown_event = threading.Event()
gpu_manager = get_gpu_manager()  # Initialize GPU manager
timeout_manager = get_timeout_manager()  # Initialize timeout manager

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully with comprehensive GPU cleanup"""
    print(f"\n[SHUTDOWN] Received signal {signum}, shutting down gracefully...")
    shutdown_event.set()
    
    # Clean up TTS service with GPU management
    global tts_service
    if tts_service:
        try:
            print("[SHUTDOWN] Cleaning up TTS service...")
            
            # Use GPU manager for comprehensive cleanup
            gpu_manager.cleanup_all()
            
            tts_service = None
            print("[SHUTDOWN] TTS service cleaned up")
        except Exception as e:
            print(f"[WARNING] Error during TTS cleanup: {e}")
            # Emergency cleanup if normal cleanup fails
            try:
                emergency_gpu_cleanup()
            except Exception as emergency_error:
                print(f"[ERROR] Emergency cleanup failed: {emergency_error}")
    
    print("[SHUTDOWN] Graceful shutdown complete")
    sys.exit(0)

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    # Handle common termination signals
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, signal_handler)

def initialize_tts(force_gpu=None):
    """Initialize the Tortoise TTS service with comprehensive GPU management"""
    global tts_service, force_gpu_mode
    
    # Use global setting if not overridden
    if force_gpu is None:
        force_gpu = force_gpu_mode
    
    try:
        print("[INFO] Loading Tortoise TTS neural model with GPU management...")
        
        # Import torch first
        import torch
        
        # Initialize GPU manager first
        device = gpu_manager.initialize_device(
            device='cuda' if force_gpu or torch.cuda.is_available() else 'cpu',
            force_gpu=force_gpu
        )
        
        # Import the real implementation
        from tortoise_tts_implementation_real import create_tortoise_tts
        if device == 'cuda':
            print(f"[INFO] CUDA detected: {torch.cuda.get_device_name(0)}")
            try:
                # Create TTS service with GPU management context
                with gpu_manager.gpu_context() as managed_device:
                    tts_service = create_tortoise_tts(device=managed_device)
                    
                    # Track models for memory management
                    if hasattr(tts_service, 'engine') and hasattr(tts_service.engine, 'tts'):
                        gpu_manager.force_models_to_device(tts_service.engine, managed_device)
                
                print(f"[SUCCESS] Tortoise TTS ready on GPU with {len(tts_service.get_available_voices())} voices")
                print(f"[GPU_STATUS] {gpu_manager.get_memory_status()}")
                return True
                
            except Exception as cuda_error:
                if force_gpu:
                    print(f"[ERROR] GPU forced but initialization failed: {cuda_error}")
                    print("[ERROR] Cannot start - GPU was explicitly required with --gpu flag")
                    return False
                else:
                    print(f"[WARNING] GPU initialization failed: {cuda_error}")
                    print("[INFO] Falling back to CPU...")
                    # Cleanup GPU before fallback
                    gpu_manager.clear_gpu_cache(aggressive=True)
                    # Fallback to CPU
                    tts_service = create_tortoise_tts(device='cpu')
                    print(f"[SUCCESS] Tortoise TTS ready on CPU with {len(tts_service.get_available_voices())} voices")
                    return True
        else:
            if force_gpu:
                print("[ERROR] GPU forced but CUDA not available")
                print("[ERROR] Cannot start - GPU was explicitly required with --gpu flag")
                return False
            else:
                print("[INFO] Using CPU")
                tts_service = create_tortoise_tts(device='cpu')
                print(f"[SUCCESS] Tortoise TTS ready on CPU with {len(tts_service.get_available_voices())} voices")
                return True
                
    except ImportError as e:
        print(f"[ERROR] Failed to import neural Tortoise TTS: {e}")
        print("[ERROR] Make sure tortoise_tts_implementation_real.py is available")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to initialize Tortoise TTS: {e}")
        # Emergency cleanup on failure
        try:
            gpu_manager.cleanup_all()
        except:
            pass
        return False

# FastAPI models
class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = "angie"
    preset: Optional[str] = "ultra_fast"
    return_audio: Optional[bool] = True

class SynthesizeResponse(BaseModel):
    audio_base64: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None

class VoicesResponse(BaseModel):
    voices: List[str]
    total: int
    engine: str = "tortoise_neural"
    details: Optional[Dict[str, Dict[str, Any]]] = None

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management with comprehensive GPU cleanup"""
    print(f"[STARTUP] Tortoise TTS service starting on port 8015...")
    
    # Initialize TTS with GPU management
    success = initialize_tts()
    if not success:
        print("[ERROR] Failed to initialize TTS service")
        # Emergency cleanup before exit
        try:
            gpu_manager.cleanup_all()
        except:
            pass
        sys.exit(1)
    else:
        print("[INFO] Neural Tortoise TTS service ready with GPU management")
    
    yield
    
    print(f"[SHUTDOWN] Tortoise TTS service stopping...")
    
    # Comprehensive cleanup on shutdown
    try:
        print("[SHUTDOWN] Performing comprehensive GPU cleanup...")
        gpu_manager.cleanup_all()
        print("[SHUTDOWN] GPU cleanup completed")
    except Exception as e:
        print(f"[WARNING] Error during shutdown cleanup: {e}")
        try:
            emergency_gpu_cleanup()
        except Exception as emergency_error:
            print(f"[ERROR] Emergency cleanup failed: {emergency_error}")
    
    print("[SHUTDOWN] Service shutdown complete")

app = FastAPI(title="Tortoise TTS Service", version="2.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint with GPU memory status"""
    if tts_service is None:
        return {
            "status": "unhealthy",
            "service": "tts_tortoise",
            "engine": "neural_tortoise",
            "ready": False,
            "error": "TTS service not initialized"
        }
    
    try:
        voice_count = len(tts_service.get_available_voices())
        device_info = getattr(tts_service, 'device', 'unknown')
        
        # Get GPU memory status
        gpu_status = gpu_manager.get_memory_status()
        
        return {
            "status": "healthy",
            "service": "tts_tortoise",
            "engine": "neural_tortoise",
            "implementation": "Neural Network TTS with PyTorch",
            "timestamp": time.time(),
            "ready": True,
            "performance": "~2-5s generation time",
            "best_for": "ultra high-quality voice synthesis",
            "voice_count": voice_count,
            "device": device_info,
            "acceleration": "GPU acceleration" if device_info == "cuda" else "CPU processing",
            "gpu_memory": gpu_status,
            "features": [
                "30 unique voice personalities",
                "Neural network synthesis", 
                "Voice cloning capabilities",
                "Emotion-aware generation",
                "High-quality audio output",
                "Comprehensive GPU memory management"
            ]
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "service": "tts_tortoise",
            "error": str(e)
        }

@app.get("/voices", response_model=VoicesResponse)
async def get_voices():
    """Get available voices"""
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    try:
        voices = tts_service.get_available_voices()
        return VoicesResponse(
            voices=voices,
            total=len(voices),
            engine="tortoise_neural"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get voices: {str(e)}")

@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: SynthesizeRequest):
    """Synthesize speech using Tortoise TTS with GPU management, progress tracking, and timeout handling"""
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    start_time = time.time()
    
    # Use the advanced timeout manager for accurate calculations
    text_length = len(request.text)
    voice = request.voice or "angie"
    preset = request.preset or "ultra_fast"
    
    # Calculate realistic timeout using performance-based system
    estimated_timeout = timeout_manager.calculate_timeout(
        text_length=text_length,
        voice=voice,
        preset=preset
    )
    
    print(f"[SYNTHESIS] Starting synthesis for {text_length} characters")
    print(f"[SYNTHESIS] Voice: '{voice}', Preset: '{preset}'")
    print(f"[SYNTHESIS] Calculated timeout: {estimated_timeout:.1f}s ({estimated_timeout/60:.1f} minutes)")
    
    # Get performance stats for this configuration
    perf_stats = timeout_manager.get_performance_stats()
    config_key = f"{voice}_{preset}"
    if config_key in perf_stats:
        avg_time = perf_stats[config_key]['avg_char_time']
        expected_time = text_length * avg_time
        print(f"[SYNTHESIS] Based on history: ~{expected_time:.1f}s expected ({expected_time/60:.1f} minutes)")
    
    synthesis_start_time = time.time()
    
    try:
        # Use GPU management context for synthesis with memory check
        # More reasonable memory calculation: base 3GB + small buffer for longer text
        required_memory = 3000 + min(text_length * 2, 1000)  # Base 3GB + max 1GB extra for very long text
        
        # Progressive timeout retry system for ultra-high quality synthesis
        max_retries = 2
        retry_timeouts = timeout_manager.get_retry_timeouts(estimated_timeout)
        
        for attempt in range(max_retries + 1):
            current_timeout = retry_timeouts[attempt] if attempt < len(retry_timeouts) else retry_timeouts[-1]
            
            try:
                print(f"[SYNTHESIS] Attempt {attempt + 1}/{max_retries + 1} with timeout: {current_timeout:.1f}s")
                
                with gpu_manager.gpu_context(tts_service, required_memory_mb=required_memory):
                    # Generate audio using neural Tortoise TTS with progressive timeout
                    audio_base64, metadata = await asyncio.to_thread(
                        tts_service.synthesize_to_base64,
                        text=request.text,
                        voice=request.voice or "angie",
                        preset=request.preset or "ultra_fast",
                        timeout_seconds=current_timeout,
                        save_audio=True
                    )
                
                # Success - break out of retry loop
                break
                
            except Exception as attempt_error:
                if "timeout" in str(attempt_error).lower() and attempt < max_retries:
                    processing_time_so_far = time.time() - synthesis_start_time
                    print(f"[SYNTHESIS] Attempt {attempt + 1} timed out after {processing_time_so_far:.1f}s")
                    print(f"[SYNTHESIS] Retrying with extended timeout: {retry_timeouts[attempt+1] if attempt+1 < len(retry_timeouts) else 'maximum'}s")
                    
                    # Clear GPU cache before retry
                    gpu_manager.clear_gpu_cache(aggressive=True)
                    continue
                else:
                    # Final attempt failed or non-timeout error
                    raise attempt_error
        
        processing_time = time.time() - start_time
        synthesis_time = time.time() - synthesis_start_time
        
        # Record performance for future timeout improvements
        timeout_manager.record_performance(
            text_length=text_length,
            voice=voice,
            preset=preset,
            actual_time=synthesis_time
        )
        
        # Non-blocking GPU cache cleanup to prevent freezing
        print("[SYNTHESIS] Starting background GPU cleanup...")
        def background_cleanup():
            try:
                gpu_manager.clear_gpu_cache()
                print("[SYNTHESIS] Background GPU cleanup completed")
            except Exception as cleanup_error:
                print(f"[SYNTHESIS] Background cleanup error: {cleanup_error}")
        
        import threading
        cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
        cleanup_thread.start()
        
        print(f"[SYNTHESIS] ✅ Completed in {processing_time:.1f}s (synthesis: {synthesis_time:.1f}s)")
        print(f"[PERFORMANCE] {synthesis_time/text_length:.2f}s per character")
        
        return SynthesizeResponse(
            audio_base64=audio_base64,
            metadata=metadata,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"[ERROR] Synthesis failed after {processing_time:.1f}s: {e}")
        
        # Emergency cleanup on synthesis failure
        try:
            gpu_manager.clear_gpu_cache(aggressive=True)
            print(f"[CLEANUP] Emergency GPU cleanup completed")
        except Exception as cleanup_error:
            print(f"[WARNING] Emergency cleanup failed: {cleanup_error}")
        
        # Determine error type for better user feedback
        if "timeout" in str(e).lower() or "cancelled" in str(e).lower():
            # Calculate how much longer it might need
            expected_time = estimated_timeout * 2.0  # Double the original estimate
            error_msg = (f"❌ Synthesis timed out after {processing_time:.1f} seconds\n"
                        f"   Tortoise TTS requires more time for ultra-high quality synthesis\n"
                        f"   Estimated time needed: ~{expected_time/60:.1f} minutes\n"
                        f"   Try: shorter text, faster preset, or wait for completion")
        elif "memory" in str(e).lower():
            error_msg = f"Insufficient GPU memory for synthesis. Try shorter text or restart service."
        else:
            error_msg = f"Synthesis failed: {str(e)}"
        
        print(f"📊 Result: ❌ FAILED")
        print(f"   Error: {error_msg}")
        
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/gpu-status")
async def get_gpu_status():
    """Get detailed GPU memory status and management information"""
    return {
        "gpu_manager": gpu_manager.get_memory_status(),
        "tracked_models": len(gpu_manager.allocated_models),
        "monitoring_active": gpu_manager.monitoring_active,
        "device": gpu_manager.device
    }

@app.post("/gpu-cleanup")
async def manual_gpu_cleanup():
    """Manually trigger GPU cleanup with timeout protection"""
    try:
        import threading
        import time
        
        before_status = gpu_manager.get_memory_status()
        cleanup_result = {"status": "unknown", "message": ""}
        
        def safe_cleanup():
            try:
                gpu_manager.clear_gpu_cache(aggressive=False)  # Start with non-aggressive
                cleanup_result["status"] = "success"
                cleanup_result["message"] = "GPU cleanup completed safely"
            except Exception as e:
                cleanup_result["status"] = "error"
                cleanup_result["message"] = f"GPU cleanup failed: {str(e)}"
        
        # Run cleanup in thread with timeout
        cleanup_thread = threading.Thread(target=safe_cleanup, daemon=True)
        cleanup_thread.start()
        cleanup_thread.join(timeout=10.0)  # 10 second timeout
        
        if cleanup_thread.is_alive():
            cleanup_result["status"] = "timeout"
            cleanup_result["message"] = "GPU cleanup timed out - running in background"
        
        after_status = gpu_manager.get_memory_status()
        
        return {
            "status": cleanup_result["status"],
            "message": cleanup_result["message"],
            "before": before_status,
            "after": after_status,
            "background_active": cleanup_thread.is_alive()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Cleanup endpoint failed: {str(e)}"
        }

@app.post("/emergency-gpu-reset")
async def emergency_gpu_reset():
    """Emergency GPU reset for when the service is frozen"""
    try:
        print("[EMERGENCY] Performing emergency GPU reset...")
        
        # Use the emergency cleanup from the GPU manager
        emergency_gpu_cleanup()
        
        # Try to reinitialize if possible
        global tts_service  # Move global declaration to the top
        if tts_service:
            try:
                # Clear service reference
                tts_service = None
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Reinitialize
                if initialize_tts():
                    return {
                        "status": "success",
                        "message": "Emergency reset completed - service reinitialized"
                    }
                else:
                    return {
                        "status": "partial",
                        "message": "Emergency reset completed - manual restart recommended"
                    }
            except Exception as reinit_error:
                return {
                    "status": "partial", 
                    "message": f"GPU reset completed but reinit failed: {str(reinit_error)}"
                }
        else:
            return {
                "status": "success",
                "message": "Emergency GPU reset completed"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Emergency reset failed: {str(e)}"
        }

@app.post("/manual-system-cleanup")
async def manual_system_cleanup():
    """Manually trigger comprehensive system cleanup in background"""
    try:
        background_thread = gpu_manager.comprehensive_system_cleanup_background()
        return {
            "status": "started",
            "message": "Comprehensive system cleanup started in background",
            "background_active": background_thread is not None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to start system cleanup: {str(e)}"
        }

@app.get("/timeout-stats")
async def get_timeout_stats():
    """Get timeout performance statistics and configuration"""
    return {
        "timeout_config": {
            "base_overhead": timeout_manager.config.base_overhead,
            "char_processing_time": timeout_manager.config.char_processing_time,
            "safety_buffer": timeout_manager.config.safety_buffer,
            "min_timeout": timeout_manager.config.min_timeout,
            "max_timeout": timeout_manager.config.max_timeout,
        },
        "voice_complexity": timeout_manager.VOICE_COMPLEXITY,
        "preset_complexity": timeout_manager.PRESET_COMPLEXITY,
        "performance_history": timeout_manager.get_performance_stats(),
        "total_recordings": sum(len(data) for data in timeout_manager.performance_history.values())
    }

@app.post("/reset-timeout-history")
async def reset_timeout_history():
    """Reset timeout performance history"""
    try:
        timeout_manager.performance_history = {}
        timeout_manager.save_config()
        return {
            "status": "success",
            "message": "Timeout performance history reset"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to reset history: {str(e)}"
        }

@app.get("/info")
async def get_info():
    """Get detailed service information including GPU memory status"""
    if tts_service is None:
        return {"error": "TTS service not initialized"}
    
    try:
        voices = tts_service.get_available_voices()
        device_info = getattr(tts_service, 'device', 'unknown')
        gpu_status = gpu_manager.get_memory_status()
        
        return {
            "service": "tortoise_tts_neural",
            "version": "2.0.0",
            "engine": "neural_tortoise",
            "voice_count": len(voices),
            "available_voices": voices[:10],  # First 10 voices for brevity
            "device": device_info,
            "acceleration": "GPU acceleration" if device_info == "cuda" else "CPU processing",
            "timeout": "unlimited",
            "status": "ready",
            "gpu_memory": gpu_status,
            "memory_management": "comprehensive GPU cleanup enabled"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("[INFO] Initializing Neural Tortoise TTS Service")
    
    import argparse
    parser = argparse.ArgumentParser(description="Neural Tortoise TTS Service")
    parser.add_argument("--port", type=int, default=8015, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run on")
    parser.add_argument("--direct", action="store_true", help="Run service directly")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage (CUDA required)")
    
    args = parser.parse_args()
    
    # Set global GPU forcing flag
    force_gpu_mode = args.gpu
    if force_gpu_mode:
        print("[INFO] GPU mode forced via --gpu flag")
    
    if args.direct or "--direct" in sys.argv:
        print(f"[DIRECT] Starting Neural Tortoise TTS service on {args.host}:{args.port}")
        
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers()
        
        try:
            uvicorn.run(app, host=args.host, port=args.port, access_log=False)
        except KeyboardInterrupt:
            print("\n[INFO] Service interrupted by user")
        except Exception as e:
            print(f"[ERROR] Service failed: {e}")
    else:
        print("Use --direct flag to run the service directly")
        print("Or use the Enhanced Service Manager for full orchestration")
