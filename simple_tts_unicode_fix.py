#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'voicebot_orchestrator'))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import asyncio
import uvicorn
import base64
import logging
from typing import Optional
import time

app = FastAPI(title="Simple TTS with Unicode Fix", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts_manager = None

class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    return_audio: Optional[bool] = True
    high_quality: Optional[bool] = False
    engine_preference: Optional[str] = "auto"

@app.on_event("startup")
async def startup_event():
    global tts_manager
    try:
        from enhanced_tts_manager import EnhancedTTSManager
        print("🔄 Starting Simple TTS with Unicode Fix...")
        tts_manager = EnhancedTTSManager()
        print("✅ TTS Manager initialized with Unicode sanitization!")
    except Exception as e:
        print(f"❌ Failed to initialize TTS: {e}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "simple_tts_unicode_fix",
        "unicode_fix": "enabled"
    }

@app.post("/synthesize")
async def synthesize_speech(request: SynthesizeRequest):
    if not tts_manager:
        raise HTTPException(status_code=503, detail="TTS service not ready")
    
    try:
        print(f"🎤 TTS Request: '{request.text}'")
        
        # This will automatically use our Unicode sanitization
        audio_data = await tts_manager.generate_speech(request.text, "KOKORO", "default")
        
        if audio_data and len(audio_data) > 1000:
            print(f"✅ TTS Success: Generated {len(audio_data)} bytes")
            
            if request.return_audio:
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                return {
                    "audio_base64": audio_base64,
                    "metadata": {
                        "text_length": len(request.text),
                        "audio_size_bytes": len(audio_data),
                        "unicode_fix": "applied",
                        "service": "simple_tts_unicode_fix"
                    }
                }
            else:
                return Response(content=audio_data, media_type='audio/wav')
        else:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
            
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🚀 Starting Simple TTS with Unicode Fix on port 8012...")
    uvicorn.run(app, host="0.0.0.0", port=8012, log_level="info")
