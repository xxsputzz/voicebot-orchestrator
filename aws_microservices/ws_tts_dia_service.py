#!/usr/bin/env python3
"""
WebSocket Dia TTS Service - Converted from HTTP to WebSocket Streaming
Advanced text-to-speech using Dia model with real-time audio streaming
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
    from ws_service_registry import ServiceRegistration, ServiceCapabilities
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("Warning: WebSocket infrastructure not available")

# Import your existing Dia TTS implementation
try:
    from voicebot_orchestrator.dia_tts import DiaTTS
    from voicebot_orchestrator.enhanced_real_tts import EnhancedRealTTS
    TTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️  Dia TTS import error: {e}")
    TTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Dia voice configurations
DIA_VOICES = {
    "female_natural": {"name": "Dia Female Natural", "speed": 1.0, "pitch": 1.0},
    "female_expressive": {"name": "Dia Female Expressive", "speed": 1.0, "pitch": 1.0},
    "female_calm": {"name": "Dia Female Calm", "speed": 0.9, "pitch": 0.95},
    "male_natural": {"name": "Dia Male Natural", "speed": 1.0, "pitch": 1.0},
    "male_deep": {"name": "Dia Male Deep", "speed": 0.95, "pitch": 0.9},
    "male_clear": {"name": "Dia Male Clear", "speed": 1.05, "pitch": 1.05},
    "neutral_professional": {"name": "Dia Neutral Professional", "speed": 1.0, "pitch": 1.0},
    "neutral_friendly": {"name": "Dia Neutral Friendly", "speed": 1.1, "pitch": 1.05},
    "child_friendly": {"name": "Dia Child Friendly", "speed": 1.2, "pitch": 1.15},
    "elderly_wise": {"name": "Dia Elderly Wise", "speed": 0.85, "pitch": 0.9}
}

class WebSocketDiaTTSService:
    """WebSocket-enabled Dia TTS service with advanced voice synthesis"""
    
    def __init__(self):
        self.service_id = "tts_dia_ws"
        self.service_name = "WebSocket Dia TTS"
        self.websocket = None
        self.orchestrator_url = "ws://localhost:9001"  # Service port
        self.running = False
        
        # Initialize TTS engine
        self.tts_engine = None
        self.enhanced_tts = None
        
        # Audio streaming parameters
        self.sample_rate = 22050  # Dia standard
        self.chunk_size = 2048    # Audio streaming chunk size
        
        # Session management
        self.sessions = {}  # session_id -> session_data
        
        # Service registration info
        self.registration = ServiceRegistration(
            service_id=self.service_id,
            service_name=self.service_name,
            service_type="tts",
            version="1.0.0",
            endpoint="localhost",
            websocket_port=8007,
            http_port=8007,
            capabilities=ServiceCapabilities(
                realtime=True,
                streaming=True,
                languages=["en"],
                voice_models=list(DIA_VOICES.keys()) if 'DIA_VOICES' in globals() else [],
                max_concurrent=4,
                latency_ms=1000  # Premium quality, slightly higher latency
            ),
            metadata={
                "engine": "dia",
                "sample_rate": self.sample_rate,
                "supported_languages": ["en"],
                "supported_voices": list(DIA_VOICES.keys()) if 'DIA_VOICES' in globals() else [],
                "chunk_size": self.chunk_size,
                "speed": "balanced",
                "quality": "premium",
                "input_types": ["text/plain"],
                "output_types": ["audio/wav", "audio/raw"],
                "latency": "medium",
                "status": "starting",
                "voice_count": len(DIA_VOICES) if 'DIA_VOICES' in globals() else 10
            }
        )
    
    async def initialize_tts(self):
        """Initialize the Dia TTS engine"""
        if not TTS_AVAILABLE:
            logging.warning("⚠️  Dia TTS dependencies not available, using mock audio")
            self.registration.metadata["implementation_type"] = "mock"
            self.registration.status = "ready"
            return True
        
        try:
            logging.info("[Dia-WS] Initializing Dia TTS engine...")
            
            # Try to initialize Dia TTS
            try:
                self.tts_engine = DiaTTS()
                logging.info("✅ Direct Dia TTS initialized")
            except Exception as e:
                logging.warning(f"⚠️  Direct Dia failed: {e}, trying Enhanced Real TTS...")
                try:
                    self.enhanced_tts = EnhancedRealTTS()
                    logging.info("✅ Enhanced Real TTS initialized with Dia backend")
                except Exception as e2:
                    logging.error(f"❌ Enhanced TTS failed: {e2}")
                    raise e2
            
            self.registration.status = "ready"
            self.registration.metadata["implementation_type"] = "real"
            logging.info("✅ Dia TTS initialized successfully!")
            return True
            
        except Exception as e:
            logging.error(f"❌ Dia TTS initialization failed: {e}")
            self.registration.status = "degraded"
            self.registration.metadata["error"] = str(e)
            self.registration.metadata["implementation_type"] = "mock"
            return False
    
    async def connect_to_orchestrator(self):
        """Connect to WebSocket orchestrator and register service"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries and not self.running:
            try:
                logging.info(f"[Dia-WS] Connecting to orchestrator: {self.orchestrator_url}")
                
                self.websocket = await websockets.connect(
                    self.orchestrator_url,
                    ping_interval=20,
                    ping_timeout=10
                )
                
                # Register with orchestrator
                registration_msg = {
                    "type": "service_registration",
                    "data": self.registration.to_dict()
                }
                await self.websocket.send(json.dumps(registration_msg))
                
                # Start message handling
                self.running = True
                logging.info("✅ Connected to orchestrator and registered Dia TTS service")
                
                return True
                
            except Exception as e:
                retry_count += 1
                logging.error(f"❌ Connection attempt {retry_count} failed: {e}")
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                
        return False
    
    async def handle_message(self, message_data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        try:
            message_type = message_data.get("type")
            data = message_data.get("data", {})
            session_id = message_data.get("session_id")
            
            if message_type == "text_to_speech":
                await self.handle_text_to_speech(data, session_id)
            elif message_type == "tts_request":
                await self.handle_tts_request(data, session_id)
            elif message_type == "stream_tts":
                await self.handle_stream_tts(data, session_id)
            elif message_type == "session_start":
                await self.handle_session_start(data, session_id)
            elif message_type == "session_end":
                await self.handle_session_end(data, session_id)
            elif message_type == "voice_list":
                await self.handle_voice_list(data, session_id)
            else:
                logging.warning(f"[Dia-WS] Unknown message type: {message_type}")
                
        except Exception as e:
            logging.error(f"[Dia-WS] Error handling message: {e}")
            await self.send_error(session_id, f"Message handling error: {e}")
    
    async def handle_text_to_speech(self, data: Dict[str, Any], session_id: str):
        """Handle text-to-speech conversion with streaming audio"""
        try:
            text = data.get("text", "").strip()
            voice = data.get("voice", "female_natural")
            speed = data.get("speed", 1.0)
            pitch = data.get("pitch", 1.0)
            stream_audio = data.get("stream_audio", True)
            
            if not text:
                await self.send_error(session_id, "No text provided for TTS conversion")
                return
            
            # Validate voice
            if voice not in DIA_VOICES:
                logging.warning(f"[Dia-WS] Unknown voice '{voice}', using 'female_natural'")
                voice = "female_natural"
            
            # Initialize session if needed
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    "start_time": time.time(),
                    "audio_chunks": [],
                    "metadata": data.get("metadata", {}),
                    "voice_settings": {
                        "voice": voice,
                        "speed": speed,
                        "pitch": pitch
                    }
                }
            
            session = self.sessions[session_id]
            
            if stream_audio:
                await self.stream_tts_audio(text, voice, speed, pitch, session_id, session)
            else:
                await self.generate_tts_audio(text, voice, speed, pitch, session_id, session)
                
        except Exception as e:
            logging.error(f"[Dia-WS] Error handling TTS request: {e}")
            await self.send_error(session_id, f"TTS processing error: {e}")
    
    async def stream_tts_audio(self, text: str, voice: str, speed: float, pitch: float, session_id: str, session: Dict):
        """Generate streaming TTS audio using Dia"""
        start_time = time.time()
        
        try:
            # Send streaming start notification
            await self.websocket.send(json.dumps({
                "type": "tts_stream_start",
                "session_id": session_id,
                "data": {
                    "service": self.service_id,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "voice": voice,
                    "speed": speed,
                    "pitch": pitch,
                    "engine": "dia",
                    "expected_duration": len(text) * 0.08  # Dia is slightly slower than Kokoro
                }
            }))
            
            if (self.tts_engine or self.enhanced_tts) and TTS_AVAILABLE:
                # Use real Dia TTS
                try:
                    await self._generate_real_dia_audio(text, voice, speed, pitch, session_id, session)
                except Exception as e:
                    logging.error(f"[Dia-WS] Real Dia TTS error: {e}")
                    # Fallback to mock
                    await self._generate_mock_streaming_audio(text, voice, session_id)
            else:
                # Use mock streaming audio
                await self._generate_mock_streaming_audio(text, voice, session_id)
            
            processing_time = time.time() - start_time
            
            # Send completion notification
            await self.websocket.send(json.dumps({
                "type": "tts_stream_complete",
                "session_id": session_id,
                "data": {
                    "service": self.service_id,
                    "processing_time_seconds": processing_time,
                    "text_length": len(text),
                    "voice": voice,
                    "speed": speed,
                    "pitch": pitch,
                    "engine": "dia"
                }
            }))
            
            logging.info(f"[Dia-WS] TTS streaming completed in {processing_time:.2f}s for text: '{text[:50]}...'")
            
        except Exception as e:
            logging.error(f"[Dia-WS] Error in streaming TTS: {e}")
            await self.send_error(session_id, f"TTS streaming error: {e}")
    
    async def _generate_real_dia_audio(self, text: str, voice: str, speed: float, pitch: float, session_id: str, session: Dict):
        """Generate real Dia TTS audio with streaming"""
        try:
            # Use the appropriate TTS engine
            if self.tts_engine:
                # Direct Dia TTS
                audio_data = await self.tts_engine.synthesize(
                    text=text,
                    voice=voice,
                    speed=speed,
                    pitch=pitch
                )
            elif self.enhanced_tts:
                # Enhanced Real TTS with Dia backend
                audio_data = await self.enhanced_tts.generate_speech(
                    text=text,
                    voice_id=voice,
                    speed=speed,
                    pitch=pitch
                )
            else:
                raise Exception("No TTS engine available")
            
            # Stream audio in chunks
            if audio_data:
                await self._stream_audio_chunks(audio_data, session_id)
            else:
                raise Exception("No audio data generated")
                
        except Exception as e:
            logging.error(f"[Dia-WS] Real Dia audio generation error: {e}")
            raise
    
    async def _stream_audio_chunks(self, audio_data: bytes, session_id: str):
        """Stream audio data in chunks"""
        try:
            # Stream audio in chunks
            total_chunks = len(audio_data) // self.chunk_size + (1 if len(audio_data) % self.chunk_size else 0)
            
            for i in range(0, len(audio_data), self.chunk_size):
                chunk = audio_data[i:i + self.chunk_size]
                chunk_number = i // self.chunk_size + 1
                
                # Send audio chunk
                import base64
                chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                
                await self.websocket.send(json.dumps({
                    "type": "tts_audio_chunk",
                    "session_id": session_id,
                    "data": {
                        "audio_data": chunk_b64,
                        "chunk_number": chunk_number,
                        "total_chunks": total_chunks,
                        "sample_rate": self.sample_rate,
                        "format": "wav",
                        "engine": "dia",
                        "is_final": chunk_number == total_chunks
                    }
                }))
                
                # Small delay for streaming
                await asyncio.sleep(0.015)  # Slightly longer delay than Kokoro
                
        except Exception as e:
            logging.error(f"[Dia-WS] Error streaming audio chunks: {e}")
            raise
    
    async def _generate_mock_streaming_audio(self, text: str, voice: str, session_id: str):
        """Generate mock streaming audio for testing"""
        try:
            # Generate mock audio (simple sine wave or silence)
            import struct
            import math
            
            duration = len(text) * 0.12  # Slightly longer than Kokoro for realism
            num_samples = int(self.sample_rate * duration)
            
            # Generate richer sine wave as mock audio (Dia is higher quality)
            frequency = 523.25  # C5 note (higher than Kokoro's A4)
            audio_data = b''
            
            for i in range(num_samples):
                # Add some harmonic richness
                fundamental = math.sin(2 * math.pi * frequency * i / self.sample_rate)
                harmonic = math.sin(2 * math.pi * frequency * 2 * i / self.sample_rate) * 0.3
                sample = int(32767 * (fundamental + harmonic) * 0.2)
                audio_data += struct.pack('<h', sample)
            
            # Stream mock audio
            await self._stream_audio_chunks(audio_data, session_id)
            
        except Exception as e:
            logging.error(f"[Dia-WS] Error generating mock audio: {e}")
            raise
    
    async def handle_tts_request(self, data: Dict[str, Any], session_id: str):
        """Handle TTS request (backward compatibility)"""
        await self.handle_text_to_speech(data, session_id)
    
    async def handle_stream_tts(self, data: Dict[str, Any], session_id: str):
        """Handle streaming TTS request"""
        data["stream_audio"] = True
        await self.handle_text_to_speech(data, session_id)
    
    async def handle_voice_list(self, data: Dict[str, Any], session_id: str):
        """Handle voice list request"""
        voice_list = {
            voice_id: {
                "name": voice_info["name"],
                "speed": voice_info["speed"],
                "pitch": voice_info.get("pitch", 1.0),
                "engine": "dia",
                "quality": "premium",
                "latency": "medium"
            }
            for voice_id, voice_info in DIA_VOICES.items()
        }
        
        response = {
            "type": "voice_list_response",
            "session_id": session_id,
            "data": {
                "service": self.service_id,
                "engine": "dia",
                "voices": voice_list,
                "total_voices": len(voice_list)
            }
        }
        await self.websocket.send(json.dumps(response))
    
    async def handle_session_start(self, data: Dict[str, Any], session_id: str):
        """Handle session start"""
        self.sessions[session_id] = {
            "start_time": time.time(),
            "audio_chunks": [],
            "metadata": data.get("metadata", {}),
            "voice_preference": data.get("voice", "female_natural"),
            "speed_preference": data.get("speed", 1.0),
            "pitch_preference": data.get("pitch", 1.0)
        }
        
        response = {
            "type": "session_started",
            "session_id": session_id,
            "data": {
                "service": self.service_id,
                "ready": True,
                "engine": "dia",
                "available_voices": list(DIA_VOICES.keys()),
                "sample_rate": self.sample_rate,
                "capabilities": {
                    "streaming": True,
                    "multiple_voices": True,
                    "speed_control": True,
                    "pitch_control": True,
                    "premium_quality": True
                }
            }
        }
        await self.websocket.send(json.dumps(response))
        logging.info(f"[Dia-WS] Session started: {session_id}")
    
    async def handle_session_end(self, data: Dict[str, Any], session_id: str):
        """Handle session end"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Calculate session stats
            duration = time.time() - session["start_time"]
            audio_chunks_generated = len(session["audio_chunks"])
            
            # Clean up session
            del self.sessions[session_id]
            
            response = {
                "type": "session_ended",
                "session_id": session_id,
                "data": {
                    "service": self.service_id,
                    "duration": duration,
                    "audio_chunks_generated": audio_chunks_generated
                }
            }
            await self.websocket.send(json.dumps(response))
            logging.info(f"[Dia-WS] Session ended: {session_id} (duration: {duration:.2f}s)")
    
    async def send_error(self, session_id: str, error_message: str):
        """Send error message"""
        if self.websocket:
            error_response = {
                "type": "error",
                "session_id": session_id,
                "data": {
                    "service": self.service_id,
                    "error": error_message,
                    "timestamp": time.time()
                }
            }
            await self.websocket.send(json.dumps(error_response))
    
    async def send_health_update(self):
        """Send periodic health updates"""
        while self.running:
            try:
                if self.websocket:
                    health_data = {
                        "type": "service_health",
                        "data": {
                            "service_id": self.service_id,
                            "status": "healthy" if (self.tts_engine or self.enhanced_tts or not TTS_AVAILABLE) else "degraded",
                            "active_sessions": len(self.sessions),
                            "engine": "dia",
                            "available_voices": len(DIA_VOICES),
                            "sample_rate": self.sample_rate,
                            "implementation": self.registration.metadata.get("implementation_type", "unknown"),
                            "timestamp": time.time()
                        }
                    }
                    await self.websocket.send(json.dumps(health_data))
                
                await asyncio.sleep(30)  # Send health update every 30 seconds
                
            except Exception as e:
                logging.error(f"[Dia-WS] Error sending health update: {e}")
                break
    
    async def message_loop(self):
        """Main message handling loop with keep-alive"""
        try:
            while self.running:
                try:
                    # Wait for messages with timeout to allow for keep-alive checks
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                    try:
                        data = json.loads(message)
                        await self.handle_message(data)
                    except json.JSONDecodeError:
                        logging.error(f"[WS] Invalid JSON received: {message}")
                    except Exception as e:
                        logging.error(f"[WS] Error processing message: {e}")
                        
                except asyncio.TimeoutError:
                    # No message received, continue loop (keep-alive)
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logging.info("[WS] WebSocket connection closed")
                    break
                    
        except Exception as e:
            logging.error(f"[WS] Message loop error: {e}")
        finally:
            self.running = False
    
    async def run(self):
        """Run the WebSocket Dia TTS service"""
        # Initialize TTS engine
        await self.initialize_tts()  # Don't fail if TTS init fails, use mock instead
        
        # Connect to orchestrator
        if not await self.connect_to_orchestrator():
            logging.error("[Dia-WS] Failed to connect to orchestrator")
            return False
        
        try:
            # Start health monitoring task
            health_task = asyncio.create_task(self.send_health_update())
            
            # Start message handling
            await self.message_loop()
            
        except KeyboardInterrupt:
            logging.info("[Dia-WS] Received interrupt, shutting down...")
        except Exception as e:
            logging.error(f"[Dia-WS] Service error: {e}")
        finally:
            self.running = False
            if self.websocket:
                await self.websocket.close()
            
            # Clean up sessions
            self.sessions.clear()
            
            logging.info("[Dia-WS] Service shut down complete")
        
        return True

def safe_print(text):
    """Safe print function that handles Unicode characters for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)

async def main():
    """Main entry point"""
    safe_print("🎭 Starting WebSocket Dia TTS Service...")
    safe_print("📡 Connecting to orchestrator at ws://localhost:9001")
    safe_print("💎 Premium quality TTS with advanced voice synthesis")
    safe_print("🗣️  Supporting 10 Dia voices with pitch control")
    safe_print("-" * 60)
    
    service = WebSocketDiaTTSService()
    
    success = await service.run()
    
    if success:
        safe_print("✅ WebSocket Dia TTS service completed successfully")
    else:
        safe_print("❌ WebSocket Dia TTS service encountered errors")
        sys.exit(1)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the service
    asyncio.run(main())
