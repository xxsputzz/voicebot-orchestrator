#!/usr/bin/env python3
"""
Debug script to test Zonos TTS imports
"""

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print("=== Testing WebSocket Service Registry Import ===")
try:
    from ws_service_registry import StreamingMessage, ServiceRegistration, ServiceCapabilities
    print("✅ WebSocket service registry imported successfully")
    WEBSOCKET_AVAILABLE = True
except ImportError as e:
    print(f"❌ WebSocket import failed: {e}")
    WEBSOCKET_AVAILABLE = False

print("\n=== Testing Zonos TTS Import ===")
try:
    from voicebot_orchestrator.zonos_tts import ZonosTTS, create_zonos_tts
    print("✅ Zonos TTS imported successfully")
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Zonos TTS import failed: {e}")
    try:
        from voicebot_orchestrator.real_zonos_tts import RealZonosTTS
        print("✅ RealZonosTTS imported as fallback")
        TTS_AVAILABLE = True
    except ImportError as e2:
        print(f"❌ Zonos TTS fallback also failed: {e2}")
        TTS_AVAILABLE = False

print(f"\nWebSocket Available: {WEBSOCKET_AVAILABLE}")
print(f"TTS Available: {TTS_AVAILABLE}")

if TTS_AVAILABLE and 'create_zonos_tts' in locals():
    try:
        tts_instance = create_zonos_tts("neutral")
        print("✅ TTS instance created successfully")
    except Exception as e:
        print(f"❌ TTS instance creation failed: {e}")
