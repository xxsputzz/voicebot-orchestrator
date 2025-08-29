"""
STT Microservice for AWS Deployment
Runs on CPU-optimized instances (c5.xlarge)
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import uvicorn
import tempfile
import os
from typing import Dict, Any
import time

# Import your existing STT implementation
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from voicebot_orchestrator.stt import WhisperSTT

app = FastAPI(title="STT Microservice", version="1.0.0")

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global STT instance
stt_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize STT service on startup"""
    global stt_service
    logging.info("🎙️ Initializing STT Microservice...")
    
    try:
        # Initialize with optimal settings for AWS
        stt_service = WhisperSTT(
            model_name="base",  # Good balance of speed/accuracy
            device="cpu"        # CPU-only for cost optimization
        )
        logging.info("✅ STT Microservice ready!")
    except Exception as e:
        logging.error(f"❌ STT initialization failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global stt_service
    if stt_service:
        # Cleanup STT resources
        stt_service = None
        logging.info("🛑 STT Microservice shutdown complete")

@app.get("/health")
async def health_check():
    """Health check endpoint for AWS load balancer"""
    return {
        "status": "healthy",
        "service": "stt",
        "timestamp": time.time(),
        "ready": stt_service is not None
    }

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Transcribe audio file to text
    
    Args:
        audio: Audio file (wav, mp3, m4a, etc.)
        
    Returns:
        JSON with transcribed text and metadata
    """
    if not stt_service:
        raise HTTPException(status_code=503, detail="STT service not ready")
    
    start_time = time.time()
    
    try:
        # Validate file format
        if not stt_service.validate_audio_format(audio.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported audio format. Supported: {stt_service.get_supported_formats()}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio.filename.split('.')[-1]}") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Transcribe the audio
            transcript = await stt_service.transcribe_file(temp_path)
            processing_time = time.time() - start_time
            
            return {
                "transcript": transcript,
                "confidence": 0.95,  # Mock confidence score
                "processing_time_seconds": round(processing_time, 3),
                "audio_duration_seconds": len(content) / 16000,  # Approximate
                "model": "whisper-base",
                "language": "en"
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"❌ Transcription failed after {processing_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe_text")
async def transcribe_raw_audio(audio_data: bytes) -> Dict[str, Any]:
    """
    Transcribe raw audio bytes to text
    
    Args:
        audio_data: Raw audio bytes
        
    Returns:
        JSON with transcribed text
    """
    if not stt_service:
        raise HTTPException(status_code=503, detail="STT service not ready")
    
    start_time = time.time()
    
    try:
        transcript = await stt_service.transcribe_audio(audio_data)
        processing_time = time.time() - start_time
        
        return {
            "transcript": transcript,
            "processing_time_seconds": round(processing_time, 3),
            "model": "whisper-base"
        }
        
    except Exception as e:
        logging.error(f"❌ Raw audio transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/info")
async def service_info():
    """Get STT service information"""
    return {
        "service": "stt",
        "model": "whisper-base",
        "supported_formats": ["wav", "mp3", "m4a", "flac", "ogg"],
        "max_file_size_mb": 25,
        "optimal_sample_rate": 16000,
        "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
    }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the service
    uvicorn.run(
        "stt_service:app",
        host="0.0.0.0",
        port=8001,
        workers=1,  # Single worker for STT
        log_level="info"
    )
