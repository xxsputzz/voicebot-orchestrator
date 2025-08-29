"""
Whisper STT Microservice - Independent Service
Speech-to-Text using OpenAI Whisper
Port: 8002
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import logging
import uvicorn
import tempfile
import os
import sys
from typing import Dict, Any, Optional
import time

# Import your existing STT implementation
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from voicebot_orchestrator.stt import WhisperSTT

app = FastAPI(title="Whisper STT Microservice", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global STT instance
whisper_stt = None

class TranscribeRequest(BaseModel):
    audio_base64: Optional[str] = None
    language: Optional[str] = "auto"
    task: Optional[str] = "transcribe"  # or "translate"

class TranscribeResponse(BaseModel):
    text: str
    language_detected: str
    processing_time_seconds: float
    confidence: Optional[float] = None
    segments: Optional[list] = None

@app.on_event("startup")
async def startup_event():
    """Initialize Whisper STT service on startup"""
    global whisper_stt
    logging.info("[STT] Initializing Whisper STT Microservice...")
    
    try:
        # Initialize Whisper STT (model loads lazily)
        whisper_stt = WhisperSTT(
            model_name="base",  # Can be: tiny, base, small, medium, large
            device="cpu"  # Use cpu for now
        )
        
        logging.info("✅ Whisper STT Microservice ready!")
        
    except Exception as e:
        logging.error(f"❌ Whisper STT initialization failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global whisper_stt
    logging.info("🛑 Whisper STT Microservice shutdown complete")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "whisper_stt",
        "timestamp": time.time(),
        "ready": whisper_stt is not None,
        "model": "whisper-base" if whisper_stt else None
    }

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = "auto",
    task: str = "transcribe"
):
    """
    Transcribe audio using Whisper STT
    
    Args:
        audio: Audio file (WAV, MP3, etc.)
        language: Language code (auto for detection)
        task: "transcribe" or "translate"
    """
    if not whisper_stt:
        raise HTTPException(status_code=503, detail="Whisper STT service not ready")
    
    start_time = time.time()
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await audio.read())
            temp_path = temp_file.name
        
        try:
            # Transcribe audio using the correct method
            text = await whisper_stt.transcribe_file(temp_path)
            
            processing_time = time.time() - start_time
            
            return TranscribeResponse(
                text=text,
                language_detected="auto",  # Mock for now
                processing_time_seconds=processing_time,
                confidence=0.95,  # Mock confidence
                segments=[]  # Mock segments
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"❌ Transcription failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "processing_time": processing_time
            }
        )

@app.post("/transcribe_base64", response_model=TranscribeResponse)
async def transcribe_base64(request: TranscribeRequest):
    """
    Transcribe audio from base64 data
    
    Args:
        request: TranscribeRequest with base64 audio data
    """
    if not whisper_stt:
        raise HTTPException(status_code=503, detail="Whisper STT service not ready")
    
    if not request.audio_base64:
        raise HTTPException(status_code=400, detail="No audio_base64 provided")
    
    start_time = time.time()
    
    try:
        import base64
        
        # Decode base64 audio
        audio_data = base64.b64decode(request.audio_base64)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            # Transcribe audio using the correct method
            text = await whisper_stt.transcribe_file(temp_path)
            
            processing_time = time.time() - start_time
            
            return TranscribeResponse(
                text=text,
                language_detected="auto",  # Mock for now
                processing_time_seconds=processing_time,
                confidence=0.95,  # Mock confidence  
                segments=[]  # Mock segments
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"❌ Transcription failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "processing_time": processing_time
            }
        )

@app.get("/models")
async def get_available_models():
    """Get available Whisper models"""
    return {
        "available_models": [
            {"name": "tiny", "size": "~39 MB", "speed": "very fast", "accuracy": "basic"},
            {"name": "base", "size": "~74 MB", "speed": "fast", "accuracy": "good"},
            {"name": "small", "size": "~244 MB", "speed": "medium", "accuracy": "better"},
            {"name": "medium", "size": "~769 MB", "speed": "slow", "accuracy": "very good"},
            {"name": "large", "size": "~1550 MB", "speed": "very slow", "accuracy": "best"}
        ],
        "current_model": "base",
        "supported_languages": "100+ languages",
        "supported_formats": ["wav", "mp3", "flac", "m4a", "ogg"]
    }

@app.get("/status")
async def get_service_status():
    """Get detailed service status"""
    return {
        "service": "whisper_stt",
        "status": "healthy" if whisper_stt else "unhealthy",
        "model": "whisper-base",
        "version": "1.0.0",
        "ready": whisper_stt is not None,
        "port": 8002,
        "endpoints": [
            "/health",
            "/transcribe",
            "/transcribe_base64",
            "/models",
            "/status"
        ]
    }

def safe_print(text):
    """Safe print function that handles Unicode characters for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback to ASCII-safe version
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)

def launch_in_new_terminal():
    """Launch the service in a new terminal window."""
    import subprocess
    import sys
    import os
    
    # Get the current script path
    script_path = os.path.abspath(__file__)
    python_exe = sys.executable
    
    # Command to run this script with --direct flag
    cmd = [python_exe, script_path, "--direct"]
    
    try:
        # Launch in new PowerShell window
        subprocess.Popen([
            "powershell.exe", 
            "-Command", 
            f"Start-Process -FilePath '{python_exe}' -ArgumentList '{script_path}', '--direct' -WindowStyle Normal"
        ], shell=True)
        
        safe_print("🚀 Launching Whisper STT service in new terminal...")
        safe_print("📋 Service will be available at: http://localhost:8002")
        safe_print("✅ Original terminal is now free for other commands")
        
    except Exception as e:
        safe_print(f"❌ Failed to launch in new terminal: {e}")
        safe_print("🔄 Falling back to current terminal...")
        return False
    
    return True

if __name__ == "__main__":
    import sys
    
    # Check if this is a direct launch (from new terminal)
    if "--direct" in sys.argv:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        safe_print(">>> Starting Whisper STT Service in dedicated terminal...")
        safe_print(">>> Port: 8002")
        safe_print(">>> Health: http://localhost:8002/health")
        safe_print(">>> Status: http://localhost:8002/status")
        safe_print(">>> Close this window to stop the service")
        safe_print("-" * 50)
        
        # Run the service
        uvicorn.run(
            "stt_whisper_service:app",
            host="0.0.0.0",
            port=8002,
            workers=1,
            log_level="info"
        )
    else:
        # Try to launch in new terminal, fallback to current if it fails
        if not launch_in_new_terminal():
            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Run the service in current terminal
            uvicorn.run(
                "stt_whisper_service:app",
                host="0.0.0.0",
                port=8002,
                workers=1,
                log_level="info"
            )
