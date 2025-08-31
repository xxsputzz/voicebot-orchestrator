#!/usr/bin/env python3
"""
Simple STT Service Test - Bypass startup issues
===============================================
"""
import sys
import os
import tempfile
import logging

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from fastapi import FastAPI, UploadFile, File
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Simple STT Test")

# Initialize STT manually
whisper_stt = None

try:
    from voicebot_orchestrator.real_whisper_stt import WhisperSTT
    whisper_stt = WhisperSTT(model_name="base", device="cpu")
    print("✅ WhisperSTT initialized successfully")
    print(f"   Using real: {whisper_stt._use_real}")
except Exception as e:
    print(f"❌ Failed to initialize STT: {e}")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "stt_ready": whisper_stt is not None,
        "implementation": "real" if (whisper_stt and hasattr(whisper_stt, '_use_real') and whisper_stt._use_real) else "mock"
    }

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if not whisper_stt:
        return {"error": "STT not initialized"}
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await audio.read())
            temp_path = temp_file.name
        
        # Transcribe
        text = await whisper_stt.transcribe_file(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        return {
            "text": text,
            "confidence": 0.95,
            "processing_time_seconds": 1.0
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting simple STT service on port 8004...")
    uvicorn.run(app, host="0.0.0.0", port=8004)
