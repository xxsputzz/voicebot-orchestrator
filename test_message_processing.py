#!/usr/bin/env python3
"""
Test script to verify message processing fixes
"""

import json
import asyncio
from datetime import datetime
from ws_orchestrator_service import WebSocketOrchestrator

async def test_message_processing():
    """Test the fixed message processing logic"""
    
    # Create a minimal orchestrator instance for testing
    orchestrator = WebSocketOrchestrator()
    
    # Test service health message (the type that was causing errors)
    health_message = {
        "type": "service_health",
        "data": {
            "service_id": "stt_whisper_ws",
            "status": "healthy",
            "active_sessions": 0,
            "model": "whisper-1",
            "device": "cpu",
            "implementation": "test",
            "timestamp": 1725567600.0
        }
    }
    
    print("Testing service health message processing...")
    try:
        await orchestrator.process_service_message("stt_whisper_ws", json.dumps(health_message))
        print("✅ Service health message processed successfully!")
    except Exception as e:
        print(f"❌ Error processing service health message: {e}")
    
    # Test StreamingMessage format
    streaming_message = {
        "session_id": "test_session",
        "timestamp": datetime.utcnow().isoformat(),
        "type": "heartbeat",
        "data": {"status": "alive"},
        "metadata": None
    }
    
    print("\nTesting StreamingMessage format...")
    try:
        await orchestrator.process_service_message("stt_whisper_ws", json.dumps(streaming_message))
        print("✅ StreamingMessage processed successfully!")
    except Exception as e:
        print(f"❌ Error processing StreamingMessage: {e}")

if __name__ == "__main__":
    asyncio.run(test_message_processing())
