#!/usr/bin/env python3
"""
WebSocket Zonos TTS Service - Converted from HTTP to WebSocket Streaming
Balanced efficient text-to-speech using Zonos model with real-time audio streaming
"""

import asyncio
import json
import logging
import sys
import os
import time
import wave
import io
from typing import Dict, Any, List, Optional
import websockets

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import WebSocket base client and service registry
try:
    # Try both possible import paths
    try:
        from ws_service_registry import ServiceRegistration, ServiceCapabilities
    except ImportError:
        sys.path.append(os.path.dirname(__file__))
        from ws_service_registry import ServiceRegistration, ServiceCapabilities
    WEBSOCKET_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️  WebSocket infrastructure import error: {e}")
    WEBSOCKET_AVAILABLE = False
    
    # Create fallback classes when WebSocket infrastructure is not available
    class ServiceRegistration:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class ServiceCapabilities:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

# Import your existing Zonos TTS implementation
try:
    # Try the correct import path for Zonos TTS
    from voicebot_orchestrator.zonos_tts import ZonosTTS, create_zonos_tts
    TTS_AVAILABLE = True
    logging.info("✅ Zonos TTS imported successfully")
except ImportError as e:
    logging.warning(f"⚠️  Zonos TTS import error: {e}")
    TTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Zonos voice configurations
ZONOS_VOICES = {
    "neutral": {"name": "Zonos Neutral", "speed": 1.0},
    "professional": {"name": "Zonos Professional", "speed": 1.0},
    "friendly": {"name": "Zonos Friendly", "speed": 1.0},
    "confident": {"name": "Zonos Confident", "speed": 1.0},
    "warm": {"name": "Zonos Warm", "speed": 1.0},
    "clear": {"name": "Zonos Clear", "speed": 1.0}
}

class WebSocketZonosTTSService:
    def __init__(self, orchestrator_url="ws://localhost:9001"):
        self.orchestrator_url = orchestrator_url
        self.websocket = None
        self.running = False
        self.sessions = {}
        
        # Service configuration
        self.service_id = "tts_zonos_ws"
        self.service_type = "tts"
        self.voices = ZONOS_VOICES
        
        # Initialize TTS if available
        if TTS_AVAILABLE:
            try:
                self.zonos_tts = create_zonos_tts(voice="neutral")
                logging.info("✅ Zonos TTS engine initialized successfully")
            except Exception as e:
                logging.error(f"❌ Failed to initialize Zonos TTS: {e}")
                self.zonos_tts = None
        else:
            self.zonos_tts = None
            logging.warning("⚠️  Running in simulation mode - no TTS engine available")
        
        # Service registration info
        if WEBSOCKET_AVAILABLE:
            self.registration = ServiceRegistration(
                service_id=self.service_id,
                service_type=self.service_type,
                service_name="WebSocket Zonos TTS Service",
                version="1.0.0",
                endpoint="ws://localhost:9002",
                websocket_port=9002,
                http_port=0,
                capabilities=ServiceCapabilities(
                    realtime=True,
                    streaming=True,
                    languages=['en'],
                    voice_models=list(self.voices.keys()),
                    max_concurrent=5,
                    latency_ms=800
                ),
                metadata={
                    "engine": "zonos",
                    "voices": len(self.voices),
                    "description": "balanced_efficient",
                    "implementation_type": "websocket_streaming"
                }
            )

    async def connect_to_orchestrator(self):
        """Connect to the WebSocket orchestrator"""
        try:
            self.websocket = await websockets.connect(self.orchestrator_url)
            logging.info(f"📡 Connected to orchestrator at {self.orchestrator_url}")
            return True
        except Exception as e:
            logging.error(f"❌ Failed to connect to orchestrator: {e}")
            return False
    
    async def register_service(self):
        """Register this service with the orchestrator"""
        if not WEBSOCKET_AVAILABLE:
            logging.warning("⚠️  WebSocket infrastructure not available, skipping registration")
            return False
            
        try:
            registration_msg = {
                "type": "service_registration",
                "data": {
                    "service_id": self.service_id,
                    "service_type": self.service_type,
                    "service_name": "WebSocket Zonos TTS Service",
                    "version": "1.0.0",
                    "endpoint": "ws://localhost:9002",
                    "websocket_port": 9002,
                    "capabilities": {
                        "realtime": True,
                        "streaming": True,
                        "languages": ['en'],
                        "voice_models": list(self.voices.keys()),
                        "max_concurrent": 5,
                        "latency_ms": 800
                    },
                    "metadata": {
                        "engine": "zonos",
                        "voices": len(self.voices),
                        "description": "balanced_efficient",
                        "implementation_type": "websocket_streaming"
                    }
                },
                "session_id": self.service_id,
                "timestamp": time.time()
            }
            
            await self.websocket.send(json.dumps(registration_msg))
            logging.info(f"📝 Service registration sent for {self.service_id}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to register service: {e}")
            return False

    async def handle_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        try:
            message_type = data.get("type")
            session_id = data.get("session_id", "unknown")
            
            if message_type == "tts_request":
                await self.handle_tts_request(session_id, data.get("data", {}))
            elif message_type == "ping":
                await self.send_pong(session_id)
            else:
                logging.warning(f"[TTS-WS] Unknown message type: {message_type}")
                
        except Exception as e:
            logging.error(f"[TTS-WS] Message handling error: {e}")
            await self.send_error(session_id, f"Message handling error: {e}")

    async def handle_tts_request(self, session_id: str, tts_data: Dict[str, Any]):
        """Handle TTS synthesis request"""
        try:
            text = tts_data.get("text", "")
            voice = tts_data.get("voice", "neutral")
            
            if not text:
                await self.send_error(session_id, "No text provided for TTS")
                return
            
            # Simulate TTS processing if no engine available
            if not self.zonos_tts:
                logging.info(f"🎵 Simulating Zonos TTS for: '{text[:50]}...' (voice: {voice})")
                await asyncio.sleep(0.8)  # Simulate processing time
                
                # Send simulated audio response
                response = {
                    "type": "audio_output",
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "data": {
                        "audio_format": "wav",
                        "sample_rate": 22050,
                        "channels": 1,
                        "duration": len(text) * 0.1,
                        "voice": voice,
                        "text": text,
                        "simulated": True
                    },
                    "metadata": {"engine": "zonos", "service_id": self.service_id}
                }
            else:
                # Use real TTS engine
                logging.info(f"🎵 Generating Zonos TTS for: '{text[:50]}...' (voice: {voice})")
                
                # Generate audio using Zonos TTS
                audio_data = await self.generate_tts(text, voice)
                
                response = {
                    "type": "audio_output",
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "data": {
                        "audio_data": audio_data.hex() if audio_data else None,  # Convert bytes to hex string
                        "audio_format": "wav",
                        "sample_rate": 22050,
                        "channels": 1,
                        "voice": voice,
                        "text": text,
                        "audio_size": len(audio_data) if audio_data else 0
                    },
                    "metadata": {"engine": "zonos", "service_id": self.service_id}
                }
            
            await self.websocket.send(json.dumps(response))
            logging.info(f"✅ TTS response sent for session {session_id}")
            
        except Exception as e:
            logging.error(f"❌ TTS request error: {e}")
            await self.send_error(session_id, f"TTS generation error: {e}")

    async def generate_tts(self, text: str, voice: str) -> bytes:
        """Generate TTS audio using Zonos engine"""
        try:
            if voice not in self.voices:
                voice = "neutral"
            
            # Use the Zonos TTS service to generate audio (async method)
            if hasattr(self.zonos_tts, 'synthesize_speech'):
                audio_data = await self.zonos_tts.synthesize_speech(text, voice=voice)
            else:
                # Fallback to sync method if available
                audio_data = self.zonos_tts.generate_speech(text, voice)
            return audio_data
            
        except Exception as e:
            logging.error(f"❌ Zonos TTS generation error: {e}")
            raise

    async def send_error(self, session_id: str, error_message: str):
        """Send error message"""
        try:
            error_response = {
                "type": "error",
                "session_id": session_id,
                "timestamp": time.time(),
                "data": {"error": error_message, "service": "tts_zonos"},
                "metadata": {"service_id": self.service_id}
            }
            
            await self.websocket.send(json.dumps(error_response))
            
        except Exception as e:
            logging.error(f"❌ Failed to send error message: {e}")

    async def send_pong(self, session_id: str):
        """Send pong response"""
        try:
            pong_response = {
                "type": "pong",
                "session_id": session_id,
                "timestamp": time.time(),
                "data": {"service_id": self.service_id, "status": "healthy"},
                "metadata": {"service_id": self.service_id}
            }
            
            await self.websocket.send(json.dumps(pong_response))
            
        except Exception as e:
            logging.error(f"❌ Failed to send pong: {e}")

    async def send_health_update(self):
        """Send periodic health updates"""
        while self.running:
            try:
                if self.websocket:
                    health_data = {
                        "type": "service_health",
                        "data": {
                            "service_id": self.service_id,
                            "status": "healthy" if self.zonos_tts else "unhealthy",
                            "active_sessions": len(self.sessions),
                            "voices": len(self.voices),
                            "engine": "zonos",
                            "implementation": self.registration.metadata.get("implementation_type", "unknown"),
                            "timestamp": time.time()
                        }
                    }
                    await self.websocket.send(json.dumps(health_data))
                
                await asyncio.sleep(30)  # Send health update every 30 seconds
                
            except Exception as e:
                logging.error(f"❌ Health update error: {e}")
                break

    async def message_loop(self):
        """Main message handling loop with keep-alive"""
        print("[DEBUG] Starting message loop with keep-alive")
        print(f"[DEBUG] self.running = {self.running}")
        
        try:
            while self.running:
                print("[DEBUG] In while loop iteration")
                try:
                    # Wait for messages with timeout to allow for keep-alive checks
                    print("[DEBUG] Waiting for messages...")
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                    try:
                        data = json.loads(message)
                        print(f"[DEBUG] Received message: {json.dumps(data)}")
                        await self.handle_message(data)
                    except json.JSONDecodeError:
                        logging.error(f"[TTS-WS] Invalid JSON received: {message}")
                    except Exception as e:
                        logging.error(f"[TTS-WS] Error processing message: {e}")
                        
                except asyncio.TimeoutError:
                    # No message received, continue loop (keep-alive)
                    print("[DEBUG] Keep-alive timeout, continuing...")
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logging.info("[TTS-WS] WebSocket connection closed")
                    print("[DEBUG] WebSocket connection closed")
                    break
                    
        except Exception as e:
            logging.error(f"[TTS-WS] Message loop error: {e}")
            print(f"[DEBUG] Message loop error: {e}")
        finally:
            self.running = False
            print("[DEBUG] Message loop ended")

    async def run(self):
        """Main service run method"""
        try:
            # Connect to orchestrator
            connected = await self.connect_to_orchestrator()
            if not connected:
                return False
                
            # Register service
            registered = await self.register_service()
            if not registered:
                return False
                
            self.running = True
            print("[DEBUG] Starting health monitoring task")
            health_task = asyncio.create_task(self.send_health_update())
            
            # Start message handling loop
            print("[DEBUG] Starting message loop...")
            await self.message_loop()
            
            # Cleanup
            health_task.cancel()
            try:
                await health_task
            except asyncio.CancelledError:
                pass
                
            return True
            
        except Exception as e:
            logging.error(f"❌ Service error: {e}")
            return False
        finally:
            self.running = False
            if self.websocket:
                await self.websocket.close()
            print("[DEBUG] Service shut down complete")

async def main():
    service = WebSocketZonosTTSService()
    
    print("[DEBUG] Starting service with debug logging")
    print("🎵 Starting WebSocket Zonos TTS Service...")
    print("📡 Connecting to orchestrator at ws://localhost:9001")
    print("🔄 Converting HTTP TTS service to WebSocket streaming")
    print("------------------------------------------------------------")
    
    try:
        success = await service.run()
        if success:
            print("✅ WebSocket TTS service completed successfully")
        else:
            print("❌ WebSocket TTS service failed to start properly")
    except KeyboardInterrupt:
        print("\n⏹️  Service interrupted by user")
        service.running = False
    except Exception as e:
        print(f"💥 Service crashed: {e}")
        logging.error(f"Service crashed: {e}")
    
    # Keep the main process alive (this shouldn't be reached with keep-alive message loop)
    print("🔄 Service process completed")

if __name__ == "__main__":
    asyncio.run(main())
