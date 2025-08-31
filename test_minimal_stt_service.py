#!/usr/bin/env python3
"""
Minimal STT Service - Debug Version
===================================
"""
import sys
import os
import tempfile
import logging
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Minimal STT Service")

# Global STT instance
whisper_stt = None

@app.on_event("startup")
async def startup_event():
    """Initialize STT on startup"""
    global whisper_stt
    logger.info("Starting STT service initialization...")
    
    try:
        from voicebot_orchestrator.real_whisper_stt import WhisperSTT
        whisper_stt = WhisperSTT(model_name="base", device="cpu")
        
        if hasattr(whisper_stt, '_use_real') and whisper_stt._use_real:
            logger.info("✅ STT service ready with REAL Whisper!")
        else:
            logger.warning("⚠️ STT service using MOCK implementation")
            
    except Exception as e:
        logger.error(f"❌ STT initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/health")
async def health():
    """Health check"""
    try:
        return {
            "status": "healthy",
            "stt_ready": whisper_stt is not None,
            "implementation": "real" if (whisper_stt and hasattr(whisper_stt, '_use_real') and whisper_stt._use_real) else "mock",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe audio file"""
    if not whisper_stt:
        raise HTTPException(status_code=503, detail="STT service not ready")
    
    start_time = time.time()
    temp_path = None
    
    try:
        logger.info(f"Received audio file: {audio.filename}, size: {audio.size if hasattr(audio, 'size') else 'unknown'}")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
            
        logger.info(f"Saved temp file: {temp_path}, size: {len(content)} bytes")
        
        # Transcribe
        logger.info("Starting transcription...")
        text = await whisper_stt.transcribe_file(temp_path)
        
        processing_time = time.time() - start_time
        logger.info(f"Transcription complete: '{text}' ({processing_time:.2f}s)")
        
        return {
            "text": text.strip(),
            "confidence": 0.95,
            "processing_time_seconds": processing_time
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")

if __name__ == "__main__":
    print("🎙️ Starting Minimal STT Service...")
    print("📋 Service will be available at: http://localhost:8003")
    print("🔍 Health: http://localhost:8003/health")
    print("-" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )
