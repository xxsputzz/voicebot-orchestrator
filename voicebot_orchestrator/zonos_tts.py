"""
Zonos TTS Implementation - CLEAN NEURAL SPEECH  
High-quality neural text-to-speech engine with artifact removal
Uses Microsoft Edge TTS with digital noise filtering for pure human voices
"""

# Import the clean implementation with fallback
try:
    from .clean_zonos_tts import CleanZonosTTS
    print("[ZONOS] Using Clean implementation (artifact-free)")
    ZonosTTSImpl = CleanZonosTTS
except ImportError:
    try:
        from .real_zonos_tts import RealZonosTTS
        print("[ZONOS] Fallback to Real implementation")
        ZonosTTSImpl = RealZonosTTS
    except ImportError:
        print("[ZONOS] ERROR: No TTS implementation available")
        ZonosTTSImpl = None

if ZonosTTSImpl:
    # Create aliases for backwards compatibility
    ZonosTTS = ZonosTTSImpl
    create_zonos_tts = lambda voice="default", model="zonos-v1": ZonosTTSImpl(voice=voice, model=model)
    
    # For direct async creation
    async def create_zonos_tts_async(voice: str = "default", model: str = "zonos-v1"):
        """Create and initialize Clean Zonos TTS instance"""
        return ZonosTTSImpl(voice=voice, model=model)
else:
    # Fallback if no implementation available
    class ZonosTTS:
        def __init__(self, *args, **kwargs):
            raise ImportError("No Zonos TTS implementation available")
    
    create_zonos_tts = lambda *args, **kwargs: ZonosTTS()
    
    async def create_zonos_tts_async(*args, **kwargs):
        return ZonosTTS()

