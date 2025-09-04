"""
Alternative Zonos TTS Service using asyncio approach
"""
import asyncio
import json
import logging
from aiohttp import web, web_request
import base64
import sys
import os

# Add project root to path
sys.path.append('.')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global TTS instance
tts_engine = None

async def init_tts():
    """Initialize TTS engine"""
    global tts_engine
    try:
        from voicebot_orchestrator.zonos_tts import ZonosTTS
        logger.info("[TTS] Initializing ZonosTTS...")
        tts_engine = ZonosTTS(voice="default", model="zonos-v1")
        logger.info("[TTS] ZonosTTS initialized successfully")
        return True
    except Exception as e:
        logger.error(f"[ERROR] TTS initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def health_handler(request):
    """Health check endpoint"""
    logger.info("[HEALTH] Health check requested")
    response_data = {
        "status": "healthy",
        "service": "aiohttp-zonos-tts",
        "version": "1.0.0",
        "tts_ready": tts_engine is not None
    }
    return web.json_response(response_data)

async def synthesize_handler(request):
    """Synthesize speech endpoint"""
    try:
        data = await request.json()
        text = data.get('text', '')
        voice = data.get('voice', 'default')
        emotion = data.get('emotion', 'neutral')
        seed = data.get('seed')
        
        logger.info(f"[SYNTHESIZE] Request: {text[:50]}...")
        
        if not tts_engine:
            return web.json_response(
                {"error": "TTS engine not initialized"}, 
                status=503
            )
        
        # Synthesize audio
        audio_bytes = await tts_engine.synthesize_speech(
            text=text,
            voice=voice,
            emotion=emotion,
            seed=seed
        )
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        logger.info(f"[SYNTHESIZE] Generated {len(audio_bytes)} bytes of audio")
        
        response_data = {
            "success": True,
            "audio_base64": audio_base64,
            "size_bytes": len(audio_bytes),
            "voice": voice,
            "emotion": emotion
        }
        
        return web.json_response(response_data)
        
    except Exception as e:
        logger.error(f"[ERROR] Synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response(
            {"error": f"Synthesis failed: {str(e)}"}, 
            status=500
        )

async def voices_handler(request):
    """List voices endpoint"""
    if not tts_engine:
        return web.json_response(
            {"error": "TTS engine not initialized"}, 
            status=503
        )
    
    response_data = {
        "voices": tts_engine.list_voices(),
        "models": tts_engine.list_models()
    }
    return web.json_response(response_data)

async def create_app():
    """Create aiohttp application"""
    # Initialize TTS
    tts_ready = await init_tts()
    if not tts_ready:
        logger.error("Failed to initialize TTS engine")
        return None
    
    # Create app
    app = web.Application()
    
    # Add routes
    app.router.add_get('/health', health_handler)
    app.router.add_post('/synthesize', synthesize_handler)
    app.router.add_get('/voices', voices_handler)
    
    logger.info("[APP] aiohttp Zonos TTS service created")
    return app

async def main():
    """Main function"""
    logger.info("Starting aiohttp Zonos TTS service on port 8014...")
    
    app = await create_app()
    if not app:
        logger.error("Failed to create application")
        return
    
    # Run the app
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', 8014)
    await site.start()
    
    logger.info("🚀 aiohttp Zonos TTS service running on http://0.0.0.0:8014")
    logger.info("📊 Available endpoints:")
    logger.info("   GET  /health    - Health check")
    logger.info("   POST /synthesize - Synthesize speech")
    logger.info("   GET  /voices    - List voices")
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down service...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Service stopped")
    except Exception as e:
        logger.error(f"❌ Service error: {e}")
        import traceback
        traceback.print_exc()
