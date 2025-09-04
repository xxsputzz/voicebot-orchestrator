"""
Simple test version of Zonos TTS service to isolate HTTP issues
"""
from fastapi import FastAPI
import uvicorn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from voicebot_orchestrator.zonos_tts import ZonosTTS

# Create simple app without lifespan for debugging
app = FastAPI(title="Simple Zonos Test", version="1.0.0")

# Global TTS instance
tts_service = None

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    global tts_service
    print("[STARTUP] Initializing simple Zonos service...")
    tts_service = ZonosTTS(voice="default", model="zonos-v1")
    print("[STARTUP] Simple Zonos service ready!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Simple Zonos TTS Test Service", "status": "running"}

@app.get("/health")
async def health():
    """Health endpoint"""
    return {
        "status": "healthy",
        "service": "simple_zonos_test",
        "tts_ready": tts_service is not None
    }

@app.post("/synthesize")
async def synthesize(text: str = "Hello world"):
    """Simple synthesis endpoint"""
    if not tts_service:
        return {"error": "TTS not initialized"}
    
    try:
        audio_bytes = await tts_service.synthesize_speech(
            text=text,
            emotion="neutral"
        )
        
        return {
            "success": True,
            "text": text,
            "audio_size": len(audio_bytes),
            "message": "Audio generated successfully"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("[START] Starting simple Zonos test service...")
    uvicorn.run(
        "test_simple_zonos_service:app",
        host="127.0.0.1",
        port=8015,
        reload=False,
        workers=1,
        log_level="info"
    )
