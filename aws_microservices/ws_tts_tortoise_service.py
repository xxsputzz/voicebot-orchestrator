#!/usr/bin/env python3
"""
WebSocket Tortoise TTS Service - Converted from HTTP to WebSocket Streaming
Neural TTS with GPU acceleration and real-time audio streaming
"""

import asyncio
import json
import logging
import sys
import os
import time
import base64
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path
import websockets

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import WebSocket base client and service registry
try:
    from ws_service_registry import ServiceRegistration, ServiceCapabilities
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("Warning: WebSocket infrastructure not available")

# Import existing TTS implementations
try:
    # Tortoise TTS
    from tortoise_gpu_manager import get_gpu_manager, cleanup_tortoise_gpu, emergency_gpu_cleanup
    GPU_MANAGER_AVAILABLE = True
except ImportError:
    logging.warning("⚠️  GPU manager not available, using basic GPU handling")
    GPU_MANAGER_AVAILABLE = False
    
    # Fallback functions
    def get_gpu_manager():
        return None
    
    def cleanup_tortoise_gpu():
        pass
    
    def emergency_gpu_cleanup():
        pass

try:
    from tortoise_timeout_config import get_timeout_manager
    from tortoise_tts_implementation_real import create_tortoise_tts
    TORTOISE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️  Tortoise TTS import error: {e}")
    TORTOISE_AVAILABLE = False

try:
    # Kokoro TTS - Fast, real-time TTS
    from voicebot_orchestrator.real_kokoro_tts import KokoroTTS
    KOKORO_AVAILABLE = True
    logging.info("✅ Kokoro TTS available")
except ImportError:
    try:
        from voicebot_orchestrator.tts import KokoroTTS
        KOKORO_AVAILABLE = True
        logging.info("✅ Kokoro TTS available (fallback)")
    except ImportError as e:
        logging.warning(f"⚠️  Kokoro TTS import error: {e}")
        KOKORO_AVAILABLE = False

try:
    # Zonos TTS - High-quality neural TTS
    from voicebot_orchestrator.zonos_tts import ZonosTTS
    ZONOS_AVAILABLE = True
    logging.info("✅ Zonos TTS available")
except ImportError as e:
    logging.warning(f"⚠️  Zonos TTS import error: {e}")
    ZONOS_AVAILABLE = False

# Check if any TTS engine is available
TTS_AVAILABLE = TORTOISE_AVAILABLE or KOKORO_AVAILABLE or ZONOS_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class WebSocketTTSService:
    """WebSocket-enabled TTS service supporting multiple engines: Tortoise, Kokoro, and Zonos"""
    
    def __init__(self, voice: str = "rainbow", device: str = "auto", engine: str = "auto"):
        self.service_id = "tts_unified_ws"
        self.service_name = "WebSocket Unified TTS"
        self.websocket = None
        self.orchestrator_url = "ws://localhost:9001"  # Service port
        self.running = False
        
        # TTS engines
        self.tts_engines = {}
        self.current_engine = engine
        self.voice = voice
        self.device = device
        self.force_gpu_mode = device == "cuda"
        
        # GPU and timeout managers (for Tortoise)
        if TORTOISE_AVAILABLE:
            self.gpu_manager = get_gpu_manager()
            self.timeout_manager = get_timeout_manager()
        else:
            self.gpu_manager = None
            self.timeout_manager = None
        
        # Session management for streaming
        self.sessions = {}  # session_id -> session_data
        
        # Engine-specific voice mappings
        self.engine_voices = {
            "tortoise": [
                "rainbow", "angie", "tom", "freeman", "halle", "sarah", "deniro", "lj",
                "myself", "pat", "geralt", "daniel", "emma", "applejack", "train_daws",
                "train_dreams", "train_grace", "train_kennard", "train_lescault",
                "train_mouse", "weaver", "mol", "snakes", "tookyourdrugs", "dotrice",
                "p336", "tim_reynolds", "myself_1", "daws", "emma_1", "grace"
            ],
            "kokoro": [
                "af_bella", "af_nicole", "af_sarah", "am_adam", "am_michael", 
                "bf_emma", "bf_isabella", "bm_george", "bm_lewis"
            ],
            "zonos": [
                "en_male_1", "en_male_2", "en_female_1", "en_female_2",
                "neural_voice_1", "neural_voice_2"
            ]
        }
        
        # Get all available voices
        self.available_voices = []
        for engine, voices in self.engine_voices.items():
            if self._is_engine_available(engine):
                self.available_voices.extend(voices)
        
        # Engine capabilities
        self.engine_capabilities = {
            "tortoise": {
                "quality": "highest",
                "speed": "slow", 
                "gpu_acceleration": True,
                "streaming": True,
                "voice_count": len(self.engine_voices["tortoise"])
            },
            "kokoro": {
                "quality": "good",
                "speed": "fastest",
                "gpu_acceleration": False,
                "streaming": True,
                "voice_count": len(self.engine_voices["kokoro"])
            },
            "zonos": {
                "quality": "high",
                "speed": "fast",
                "gpu_acceleration": True,
                "streaming": True,
                "voice_count": len(self.engine_voices["zonos"])
            }
        }
        
        # Service registration info
        self.registration = ServiceRegistration(
            service_id=self.service_id,
            service_name=self.service_name,
            service_type="tts",
            version="2.0.0",
            endpoint="localhost",
            websocket_port=8004,
            http_port=8004,
            capabilities=ServiceCapabilities(
                realtime=True,
                streaming=True,
                languages=["en"],  # Tortoise primarily English
                voice_models=self.available_voices[:10] if hasattr(self, 'available_voices') else [],
                max_concurrent=3,  # Lower for high-quality TTS
                latency_ms=2000  # Higher latency for quality
            ),
            metadata={
                "current_engine": engine,
                "voice": voice,
                "device": device,
                "available_engines": self._get_available_engines() if hasattr(self, '_get_available_engines') else ["tortoise"],
                "available_voices": self.available_voices[:10] if hasattr(self, 'available_voices') else [],
                "total_voices": len(self.available_voices) if hasattr(self, 'available_voices') else 29,
                "engine_capabilities": self.engine_capabilities if hasattr(self, 'engine_capabilities') else {},
                "gpu_acceleration": device == "cuda",
                "unlimited_timeout": True,
                "input_types": ["text/plain"],
                "output_types": ["audio/wav", "audio/base64"],
                "status": "starting"
            }
        )
    
    def _is_engine_available(self, engine: str) -> bool:
        """Check if a TTS engine is available"""
        if engine == "tortoise":
            return TORTOISE_AVAILABLE
        elif engine == "kokoro":
            return KOKORO_AVAILABLE
        elif engine == "zonos":
            return ZONOS_AVAILABLE
        return False
    
    def _get_available_engines(self) -> List[str]:
        """Get list of available TTS engines"""
        engines = []
        if TORTOISE_AVAILABLE:
            engines.append("tortoise")
        if KOKORO_AVAILABLE:
            engines.append("kokoro")
        if ZONOS_AVAILABLE:
            engines.append("zonos")
        return engines
    
    def _get_engine_for_voice(self, voice: str) -> str:
        """Determine which engine to use for a given voice"""
        for engine, voices in self.engine_voices.items():
            if voice in voices and self._is_engine_available(engine):
                return engine
        
        # Fallback to auto-select best available engine
        return self._auto_select_engine()
    
    def _auto_select_engine(self) -> str:
        """Auto-select the best available TTS engine"""
        # Priority: Kokoro (fastest) -> Zonos (balanced) -> Tortoise (highest quality)
        if KOKORO_AVAILABLE:
            return "kokoro"
        elif ZONOS_AVAILABLE:
            return "zonos"
        elif TORTOISE_AVAILABLE:
            return "tortoise"
        return "mock"
    
    def _get_default_voice_for_engine(self, engine: str) -> str:
        """Get the default voice for a given engine"""
        defaults = {
            "tortoise": "rainbow",
            "kokoro": "af_bella", 
            "zonos": "en_female_1"
        }
        return defaults.get(engine, self.voice)
    
    async def initialize_tts(self):
        """Initialize available TTS engines"""
        try:
            logging.info(f"[TTS-WS] Initializing TTS engines...")
            
            # Initialize Tortoise TTS
            if TORTOISE_AVAILABLE:
                try:
                    logging.info("   Initializing Tortoise TTS...")
                    # Import torch for device detection
                    import torch
                    device_available = self.device
                    if device_available == "auto":
                        device_available = 'cuda' if torch.cuda.is_available() else 'cpu'
                    elif device_available == "cuda" and not torch.cuda.is_available():
                        logging.warning("⚠️  CUDA requested but not available for Tortoise, using CPU")
                        device_available = 'cpu'
                    
                    # Initialize GPU manager
                    actual_device = self.gpu_manager.initialize_device(
                        device=device_available,
                        force_gpu=self.force_gpu_mode
                    )
                    
                    # Create TTS service with GPU management
                    with self.gpu_manager.gpu_context() as managed_device:
                        self.tts_engines["tortoise"] = create_tortoise_tts(device=managed_device)
                    
                    logging.info(f"   ✅ Tortoise TTS initialized on {actual_device}")
                    
                except Exception as e:
                    logging.error(f"   ❌ Tortoise TTS initialization failed: {e}")
                    TORTOISE_AVAILABLE = False
            
            # Initialize Kokoro TTS
            if KOKORO_AVAILABLE:
                try:
                    logging.info("   Initializing Kokoro TTS...")
                    self.tts_engines["kokoro"] = KokoroTTS()
                    logging.info("   ✅ Kokoro TTS initialized")
                except Exception as e:
                    logging.error(f"   ❌ Kokoro TTS initialization failed: {e}")
                    KOKORO_AVAILABLE = False
            
            # Initialize Zonos TTS
            if ZONOS_AVAILABLE:
                try:
                    logging.info("   Initializing Zonos TTS...")
                    self.tts_engines["zonos"] = ZonosTTS()
                    logging.info("   ✅ Zonos TTS initialized")
                except Exception as e:
                    logging.error(f"   ❌ Zonos TTS initialization failed: {e}")
                    ZONOS_AVAILABLE = False
            
            # Auto-select engine if set to auto
            if self.current_engine == "auto":
                self.current_engine = self._auto_select_engine()
            
            # Verify selected engine is available
            if not self._is_engine_available(self.current_engine):
                self.current_engine = self._auto_select_engine()
            
            available_engines = self._get_available_engines()
            if available_engines:
                self.registration.status = "ready"
                self.registration.metadata["implementation_type"] = "real"
                self.registration.metadata["available_engines"] = available_engines
                self.registration.metadata["current_engine"] = self.current_engine
                logging.info(f"✅ TTS engines initialized! Available: {available_engines}, Using: {self.current_engine}")
                return True
            else:
                logging.warning("⚠️  No TTS engines available, using mock")
                self.registration.status = "ready"
                self.registration.metadata["implementation_type"] = "mock"
                self.registration.metadata["available_engines"] = []
                self.current_engine = "mock"
                return True
            
        except Exception as e:
            logging.error(f"❌ TTS initialization failed: {e}")
            self.registration.status = "error"
            self.registration.metadata["error"] = str(e)
            self.registration.metadata["implementation_type"] = "mock"
            self.current_engine = "mock"
            return False
    
    async def connect_to_orchestrator(self):
        """Connect to WebSocket orchestrator and register service"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries and not self.running:
            try:
                logging.info(f"[TTS-WS] Connecting to orchestrator: {self.orchestrator_url}")
                
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
                logging.info("✅ Connected to orchestrator and registered TTS service")
                
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
            
            if message_type == "text_input":
                await self.handle_text_to_speech(data, session_id)
            elif message_type == "tts_request":
                await self.handle_tts_request(data, session_id)
            elif message_type == "stream_tts_request":
                await self.handle_stream_tts_request(data, session_id)
            elif message_type == "voice_change":
                await self.handle_voice_change(data, session_id)
            elif message_type == "session_start":
                await self.handle_session_start(data, session_id)
            elif message_type == "session_end":
                await self.handle_session_end(data, session_id)
            else:
                logging.warning(f"[TTS-WS] Unknown message type: {message_type}")
                
        except Exception as e:
            logging.error(f"[TTS-WS] Error handling message: {e}")
            await self.send_error(session_id, f"Message handling error: {e}")
    
    async def handle_text_to_speech(self, data: Dict[str, Any], session_id: str):
        """Handle text-to-speech conversion"""
        try:
            text = data.get("text", "").strip()
            voice = data.get("voice", self.voice)
            stream_audio = data.get("stream_audio", True)
            output_format = data.get("output_format", "base64")
            
            if not text:
                await self.send_error(session_id, "No text provided for TTS")
                return
            
            # Initialize session if needed
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    "voice": voice,
                    "start_time": time.time(),
                    "metadata": data.get("metadata", {}),
                    "audio_chunks": []
                }
            
            session = self.sessions[session_id]
            session["voice"] = voice  # Update voice for session
            
            if stream_audio:
                await self.stream_tts_synthesis(text, voice, session_id, session, output_format)
            else:
                await self.generate_tts_audio(text, voice, session_id, session, output_format)
                
        except Exception as e:
            logging.error(f"[TTS-WS] Error handling text-to-speech: {e}")
            await self.send_error(session_id, f"TTS processing error: {e}")
    
    async def handle_tts_request(self, data: Dict[str, Any], session_id: str):
        """Handle TTS request (backward compatibility)"""
        await self.handle_text_to_speech(data, session_id)
    
    async def handle_stream_tts_request(self, data: Dict[str, Any], session_id: str):
        """Handle streaming TTS request"""
        data["stream_audio"] = True
        await self.handle_text_to_speech(data, session_id)
    
    async def stream_tts_synthesis(self, text: str, voice: str, session_id: str, session: Dict, output_format: str = "base64"):
        """Generate streaming TTS audio chunk by chunk"""
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
                    "output_format": output_format
                }
            }))
            
            if self.tts_service and TTS_AVAILABLE:
                # Use real TTS with streaming simulation
                try:
                    # Generate full audio first (since Tortoise doesn't do real streaming)
                    audio_data = await self._generate_real_tts(text, voice)
                    
                    # Simulate streaming by chunking the audio
                    if audio_data:
                        await self._stream_audio_chunks(audio_data, session_id, output_format)
                    else:
                        raise Exception("No audio data generated")
                        
                except Exception as e:
                    logging.error(f"[TTS-WS] Real TTS streaming error: {e}")
                    # Fallback to mock
                    await self._generate_mock_streaming_audio(text, voice, session_id, output_format)
            else:
                # Use mock streaming audio
                await self._generate_mock_streaming_audio(text, voice, session_id, output_format)
            
            processing_time = time.time() - start_time
            
            # Send completion notification
            await self.websocket.send(json.dumps({
                "type": "tts_stream_complete",
                "session_id": session_id,
                "data": {
                    "processing_time_seconds": processing_time,
                    "text_length": len(text),
                    "voice": voice,
                    "is_final": True
                }
            }))
            
            logging.info(f"[TTS-WS] Streaming TTS completed in {processing_time:.2f}s for '{text[:50]}...' (voice: {voice})")
            
        except Exception as e:
            logging.error(f"[TTS-WS] Error in streaming TTS: {e}")
            await self.send_error(session_id, f"Streaming TTS error: {e}")
    
    async def _generate_real_tts(self, text: str, voice: str) -> bytes:
        """Generate real TTS audio using Tortoise"""
        try:
            # Use GPU context if available
            if self.gpu_manager:
                with self.gpu_manager.gpu_context() as device:
                    # Generate with Tortoise TTS
                    audio_path = await self.tts_service.generate_speech(
                        text=text,
                        voice=voice,
                        output_path=None,  # Return bytes instead of file
                        preset="fast"  # Use fast preset for streaming
                    )
                    
                    # Read audio file if path returned
                    if audio_path and os.path.exists(audio_path):
                        with open(audio_path, 'rb') as f:
                            audio_data = f.read()
                        # Clean up temp file
                        os.unlink(audio_path)
                        return audio_data
            else:
                # Generate without GPU management
                audio_path = await self.tts_service.generate_speech(
                    text=text,
                    voice=voice,
                    output_path=None,
                    preset="fast"
                )
                
                if audio_path and os.path.exists(audio_path):
                    with open(audio_path, 'rb') as f:
                        audio_data = f.read()
                    os.unlink(audio_path)
                    return audio_data
            
            return None
            
        except Exception as e:
            logging.error(f"[TTS-WS] Error generating real TTS: {e}")
            return None
    
    async def _stream_audio_chunks(self, audio_data: bytes, session_id: str, output_format: str = "base64"):
        """Stream audio data in chunks"""
        chunk_size = 1024 * 4  # 4KB chunks
        total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            chunk_index = i // chunk_size
            is_final = (chunk_index == total_chunks - 1)
            
            # Encode chunk based on output format
            if output_format == "base64":
                chunk_data = base64.b64encode(chunk).decode('utf-8')
            else:
                chunk_data = chunk.hex()
            
            # Send audio chunk
            await self.websocket.send(json.dumps({
                "type": "audio_chunk",
                "session_id": session_id,
                "data": {
                    "audio_data": chunk_data,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "format": output_format,
                    "is_final": is_final
                }
            }))
            
            # Small delay for realistic streaming
            await asyncio.sleep(0.01)
    
    async def _generate_mock_streaming_audio(self, text: str, voice: str, session_id: str, output_format: str = "base64"):
        """Generate mock streaming audio for testing"""
        # Create mock audio data (simple sine wave or silence)
        import struct
        sample_rate = 22050
        duration = max(1.0, len(text) * 0.1)  # Rough duration based on text length
        
        # Generate simple sine wave as mock audio
        samples = []
        for i in range(int(sample_rate * duration)):
            t = i / sample_rate
            frequency = 440.0  # A4 note
            amplitude = 0.3
            sample = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
            samples.append(struct.pack('<h', sample))
        
        audio_data = b''.join(samples)
        
        # Add WAV header for proper audio format
        wav_header = self._create_wav_header(len(audio_data), sample_rate)
        full_audio = wav_header + audio_data
        
        # Stream the mock audio
        await self._stream_audio_chunks(full_audio, session_id, output_format)
    
    def _create_wav_header(self, data_length: int, sample_rate: int = 22050) -> bytes:
        """Create WAV file header"""
        # WAV header format
        header = struct.pack('<4sL4s', b'RIFF', 36 + data_length, b'WAVE')
        header += struct.pack('<4sLHHLLHH', b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        header += struct.pack('<4sL', b'data', data_length)
        return header
    
    async def generate_tts_audio(self, text: str, voice: str, session_id: str, session: Dict, output_format: str = "base64"):
        """Generate complete TTS audio (non-streaming)"""
        start_time = time.time()
        
        try:
            if self.tts_service and TTS_AVAILABLE:
                # Use real TTS
                audio_data = await self._generate_real_tts(text, voice)
                if not audio_data:
                    raise Exception("Failed to generate real TTS audio")
            else:
                # Generate mock audio
                audio_data = await self._generate_mock_audio(text, voice)
            
            # Encode audio data
            if output_format == "base64":
                encoded_audio = base64.b64encode(audio_data).decode('utf-8')
            else:
                encoded_audio = audio_data.hex()
            
            processing_time = time.time() - start_time
            
            # Send complete audio response
            response = {
                "type": "tts_audio_complete",
                "session_id": session_id,
                "data": {
                    "audio_data": encoded_audio,
                    "format": output_format,
                    "voice": voice,
                    "text": text,
                    "processing_time_seconds": processing_time,
                    "audio_length_bytes": len(audio_data)
                }
            }
            
            await self.websocket.send(json.dumps(response))
            logging.info(f"[TTS-WS] TTS audio generated in {processing_time:.2f}s: {len(audio_data)} bytes")
            
        except Exception as e:
            logging.error(f"[TTS-WS] Error generating TTS audio: {e}")
            await self.send_error(session_id, f"TTS generation error: {e}")
    
    async def _generate_mock_audio(self, text: str, voice: str) -> bytes:
        """Generate mock audio data for testing"""
        # Simple mock: return small audio data
        import math
        sample_rate = 22050
        duration = max(1.0, len(text) * 0.08)
        
        samples = []
        for i in range(int(sample_rate * duration)):
            t = i / sample_rate
            frequency = 440.0 + hash(voice) % 200  # Voice affects frequency slightly
            amplitude = 0.2
            sample = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
            samples.append(struct.pack('<h', sample))
        
        audio_data = b''.join(samples)
        wav_header = self._create_wav_header(len(audio_data), sample_rate)
        return wav_header + audio_data
    
    async def handle_voice_change(self, data: Dict[str, Any], session_id: str):
        """Handle voice change request"""
        try:
            new_voice = data.get("voice")
            if not new_voice:
                await self.send_error(session_id, "No voice specified")
                return
            
            if new_voice not in self.available_voices:
                await self.send_error(session_id, f"Voice '{new_voice}' not available. Available voices: {', '.join(self.available_voices[:10])}")
                return
            
            # Update session voice
            if session_id in self.sessions:
                self.sessions[session_id]["voice"] = new_voice
            
            # Update service default voice
            self.voice = new_voice
            self.registration.metadata["voice"] = new_voice
            
            # Send confirmation
            response = {
                "type": "voice_changed",
                "session_id": session_id,
                "data": {
                    "service": self.service_id,
                    "new_voice": new_voice,
                    "available_voices": self.available_voices
                }
            }
            await self.websocket.send(json.dumps(response))
            logging.info(f"[TTS-WS] Voice changed to '{new_voice}' for session {session_id}")
            
        except Exception as e:
            logging.error(f"[TTS-WS] Error changing voice: {e}")
            await self.send_error(session_id, f"Voice change error: {e}")
    
    async def handle_session_start(self, data: Dict[str, Any], session_id: str):
        """Handle session start"""
        voice = data.get("voice", self.voice)
        
        self.sessions[session_id] = {
            "voice": voice,
            "start_time": time.time(),
            "metadata": data.get("metadata", {}),
            "audio_chunks": [],
            "preferences": data.get("preferences", {})
        }
        
        response = {
            "type": "session_started",
            "session_id": session_id,
            "data": {
                "service": self.service_id,
                "ready": True,
                "voice": voice,
                "available_voices": self.available_voices,
                "capabilities": {
                    "streaming": True,
                    "voice_change": True,
                    "gpu_acceleration": self.registration.metadata.get("actual_device") == "cuda"
                }
            }
        }
        await self.websocket.send(json.dumps(response))
        logging.info(f"[TTS-WS] Session started: {session_id} (voice: {voice})")
    
    async def handle_session_end(self, data: Dict[str, Any], session_id: str):
        """Handle session end"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            duration = time.time() - session["start_time"]
            audio_chunks_count = len(session.get("audio_chunks", []))
            
            # Clean up session
            del self.sessions[session_id]
            
            response = {
                "type": "session_ended",
                "session_id": session_id,
                "data": {
                    "service": self.service_id,
                    "duration": duration,
                    "audio_chunks_generated": audio_chunks_count
                }
            }
            await self.websocket.send(json.dumps(response))
            logging.info(f"[TTS-WS] Session ended: {session_id} (duration: {duration:.2f}s)")
    
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
                    # Check GPU status if available
                    gpu_status = "unknown"
                    if self.gpu_manager and TTS_AVAILABLE:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                gpu_status = f"cuda:{torch.cuda.current_device()}"
                            else:
                                gpu_status = "cpu"
                        except:
                            gpu_status = "error"
                    
                    health_data = {
                        "type": "service_health",
                        "data": {
                            "service_id": self.service_id,
                            "status": "healthy" if self.tts_service or not TTS_AVAILABLE else "degraded",
                            "active_sessions": len(self.sessions),
                            "voice": self.voice,
                            "device": gpu_status,
                            "available_voices_count": len(self.available_voices),
                            "implementation": self.registration.metadata.get("implementation_type", "unknown"),
                            "timestamp": time.time()
                        }
                    }
                    await self.websocket.send(json.dumps(health_data))
                
                await asyncio.sleep(30)  # Send health update every 30 seconds
                
            except Exception as e:
                logging.error(f"[TTS-WS] Error sending health update: {e}")
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
    
    async def cleanup(self):
        """Clean up TTS resources"""
        try:
            if self.tts_service and TTS_AVAILABLE:
                logging.info("[TTS-WS] Cleaning up TTS service...")
                
                if self.gpu_manager:
                    self.gpu_manager.cleanup_all()
                else:
                    # Manual cleanup
                    try:
                        cleanup_tortoise_gpu()
                    except Exception as cleanup_error:
                        logging.warning(f"[TTS-WS] Cleanup warning: {cleanup_error}")
                        try:
                            emergency_gpu_cleanup()
                        except Exception as emergency_error:
                            logging.error(f"[TTS-WS] Emergency cleanup failed: {emergency_error}")
                
                self.tts_service = None
                logging.info("[TTS-WS] TTS cleanup complete")
        except Exception as e:
            logging.error(f"[TTS-WS] Error during cleanup: {e}")
    
    async def run(self):
        """Run the WebSocket TTS service"""
        # Initialize TTS engine
        await self.initialize_tts()  # Don't fail if TTS init fails, use mock instead
        
        # Connect to orchestrator
        if not await self.connect_to_orchestrator():
            logging.error("[TTS-WS] Failed to connect to orchestrator")
            return False
        
        try:
            # Start health monitoring task
            health_task = asyncio.create_task(self.send_health_update())
            
            # Start message handling
            await self.message_loop()
            
        except KeyboardInterrupt:
            logging.info("[TTS-WS] Received interrupt, shutting down...")
        except Exception as e:
            logging.error(f"[TTS-WS] Service error: {e}")
        finally:
            self.running = False
            if self.websocket:
                await self.websocket.close()
            
            # Clean up sessions
            self.sessions.clear()
            
            # Clean up TTS resources
            await self.cleanup()
            
            logging.info("[TTS-WS] Service shut down complete")
        
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
    safe_print("🔊 Starting WebSocket Tortoise TTS Service...")
    safe_print("📡 Connecting to orchestrator at ws://localhost:9001")
    safe_print("🔄 Converting HTTP TTS service to WebSocket streaming")
    safe_print("🎙️  29 voices available with GPU acceleration")
    safe_print("-" * 60)
    
    # Add math import for mock audio generation
    import math
    import struct
    
    service = WebSocketTTSService(
        voice="rainbow",  # Default voice
        device="auto"     # Auto-detect GPU/CPU
    )
    
    success = await service.run()
    
    if success:
        safe_print("✅ WebSocket TTS service completed successfully")
    else:
        safe_print("❌ WebSocket TTS service encountered errors")
        sys.exit(1)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add required imports
    import math
    import struct
    
    # Run the service
    asyncio.run(main())
