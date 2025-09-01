"""
Debug Enhanced TTS Service Integration
"""
import sys
import os
sys.path.append('.')

from voicebot_orchestrator.zonos_tts import ZonosTTS
import asyncio

async def debug_synthesis():
    """Debug the synthesis process"""
    print("🔍 Debugging TTS Synthesis")
    print("=" * 40)
    
    # Test direct synthesis
    tts = ZonosTTS()
    
    try:
        print("\n1. Testing direct synthesis with sophia...")
        result = await tts.synthesize_speech(
            text="Hello, I'm Sophia",
            voice="sophia",
            emotion="friendly"
        )
        print(f"✅ Direct synthesis successful: {len(result)} bytes")
        
    except Exception as e:
        print(f"❌ Direct synthesis failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("\n2. Testing with enhanced parameters...")
        result = await tts.synthesize_speech(
            text="Testing enhanced parameters",
            voice="sophia",
            emotion="friendly",
            speaking_style="conversational",
            emphasis_words=["enhanced"],
            speed=1.1,
            pitch=1.0
        )
        print(f"✅ Enhanced synthesis successful: {len(result)} bytes")
        
    except Exception as e:
        print(f"❌ Enhanced synthesis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_synthesis())
