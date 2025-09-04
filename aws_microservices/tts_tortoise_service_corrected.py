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

# Global variables
tts_service = None
force_gpu_mode = False  # Global flag for GPU forcing
shutdown_event = threading.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n[SHUTDOWN] Received signal {signum}, shutting down gracefully...")
    shutdown_event.set()
    
    # Clean up TTS service
    global tts_service
    if tts_service:
        try:
            print("[SHUTDOWN] Cleaning up TTS service...")
            tts_service = None
            print("[SHUTDOWN] TTS service cleaned up")
        except Exception as e:
            print(f"[WARNING] Error during TTS cleanup: {e}")
    
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
    """Initialize the Tortoise TTS service with GPU acceleration and optional forcing"""
    global tts_service, force_gpu_mode
    
    # Use global setting if not overridden
    if force_gpu is None:
        force_gpu = force_gpu_mode
    
    try:
        print("[INFO] Loading Tortoise TTS neural model...")
        
        # Import the real implementation
        from tortoise_tts_implementation_real import create_tortoise_tts
        
        # First check CUDA availability explicitly
        import torch
        if torch.cuda.is_available():
            print(f"[INFO] CUDA detected: {torch.cuda.get_device_name(0)}")
            try:
                # Try GPU first
                device = 'cuda' if force_gpu or True else None  # Default to GPU when available
                tts_service = create_tortoise_tts(device=device)
                print(f"[SUCCESS] Tortoise TTS ready on GPU with {len(tts_service.get_available_voices())} voices")
                return True
            except Exception as cuda_error:
                if force_gpu:
                    print(f"[ERROR] GPU forced but initialization failed: {cuda_error}")
                    print("[ERROR] Cannot start - GPU was explicitly required with --gpu flag")
                    return False
                else:
                    print(f"[WARNING] GPU initialization failed: {cuda_error}")
                    print("[INFO] Falling back to CPU...")
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
                print("[INFO] CUDA not available, using CPU")
                tts_service = create_tortoise_tts(device='cpu')
                print(f"[SUCCESS] Tortoise TTS ready on CPU with {len(tts_service.get_available_voices())} voices")
                return True
    except ImportError as e:
        print(f"[ERROR] Failed to import neural Tortoise TTS: {e}")
        print("[ERROR] Make sure tortoise_tts_implementation_real.py is available")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to initialize Tortoise TTS: {e}")
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
    """Application lifecycle management"""
    print(f"[STARTUP] Tortoise TTS service starting on port 8015...")
    
    # Initialize TTS
    success = initialize_tts()
    if not success:
        print("[ERROR] Failed to initialize TTS service")
        sys.exit(1)
    else:
        print("[INFO] Neural Tortoise TTS service ready")
    
    yield
    print(f"[SHUTDOWN] Tortoise TTS service stopping...")

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
    """Health check endpoint"""
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
            "features": [
                "30 unique voice personalities",
                "Neural network synthesis", 
                "Voice cloning capabilities",
                "Emotion-aware generation",
                "High-quality audio output"
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
    """Synthesize speech using Tortoise TTS with unlimited timeout"""
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    start_time = time.time()
    
    try:
        # Generate audio using neural Tortoise TTS with unlimited time
        # No timeout - let synthesis complete regardless of how long it takes
        audio_base64, metadata = await asyncio.to_thread(
            tts_service.synthesize_to_base64,
            text=request.text,
            voice=request.voice or "angie",
            preset=request.preset or "ultra_fast"
        )
        
        processing_time = time.time() - start_time
        
        return SynthesizeResponse(
            audio_base64=audio_base64,
            metadata=metadata,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"[ERROR] Synthesis failed after {processing_time:.1f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.get("/info")
async def get_info():
    """Get detailed service information"""
    if tts_service is None:
        return {"error": "TTS service not initialized"}
    
    try:
        voices = tts_service.get_available_voices()
        device_info = getattr(tts_service, 'device', 'unknown')
        
        return {
            "service": "tortoise_tts_neural",
            "version": "2.0.0",
            "engine": "neural_tortoise",
            "voice_count": len(voices),
            "available_voices": voices[:10],  # First 10 voices for brevity
            "device": device_info,
            "acceleration": "GPU acceleration" if device_info == "cuda" else "CPU processing",
            "timeout": "unlimited",
            "status": "ready"
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
