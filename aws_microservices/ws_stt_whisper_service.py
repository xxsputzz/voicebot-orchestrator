#!/usr/bin/env python3
"""
WebSocket Whisper STT Service - Converted from HTTP to WebSocket Streaming
Speech-to-Text using OpenAI Whisper with real-time streaming capabilities
"""

import asyncio
import json
import logging
import tempfile
import os
import sys
import time
import base64
import subprocess
import socket
from pathlib import Path
from typing import Dict, Any, Optional
import websockets
from dataclasses import dataclass

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import WebSocket base client and service registry
try:
    from ws_service_registry import ServiceRegistration, ServiceCapabilities
    WEBSOCKET_AVAILABLE = True
    
    # Import orchestrator utils
    try:
        from aws_microservices.orchestrator_utils import start_orchestrator_if_needed
    except ImportError:
        # Fallback import
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from orchestrator_utils import start_orchestrator_if_needed
        
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("Warning: WebSocket infrastructure not available")
    
    # Create fallback function
    def start_orchestrator_if_needed():
        print("Warning: Orchestrator auto-start not available")
        return False

# Import your existing STT implementation
try:
    # Use the working FasterWhisperSTT from tests directory
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests'))
    print(f"[DEBUG] Added to sys.path: {os.path.join(os.path.dirname(__file__), '..', 'tests')}")
    
    from faster_whisper_stt import FasterWhisperSTT, FASTER_WHISPER_AVAILABLE
    print(f"[DEBUG] FASTER_WHISPER_AVAILABLE: {FASTER_WHISPER_AVAILABLE}")
    
    if not FASTER_WHISPER_AVAILABLE:
        print("[DEBUG] faster-whisper not available, trying direct import...")
        try:
            from faster_whisper import WhisperModel
            print("[DEBUG] Direct faster_whisper import successful!")
        except ImportError as direct_error:
            print(f"[DEBUG] Direct faster_whisper import failed: {direct_error}")
    
    class WhisperSTT:
        """Wrapper to use FasterWhisperSTT with same interface"""
        def __init__(self, model_name="base", device="cpu"):
            self.faster_whisper = FasterWhisperSTT(
                model_name=model_name,
                device=device,
                compute_type="auto"
            )
            self._use_real = True
            
        async def transcribe_file(self, file_path):
            return await self.faster_whisper.transcribe_file(file_path)
            
        async def transcribe_audio(self, audio_data):
            return await self.faster_whisper.transcribe_audio(audio_data)
    
    logging.info("✅ Using working FasterWhisperSTT implementation")
    
except ImportError as e:
    logging.warning(f"⚠️  FasterWhisperSTT import error: {e}")
    print(f"[DEBUG] FasterWhisperSTT import error: {e}")
    try:
        from voicebot_orchestrator.real_whisper_stt import WhisperSTT
        logging.info("✅ Using real Whisper STT implementation")
    except ImportError as e:
        logging.warning(f"⚠️  Import error: {e}")
        try:
            from voicebot_orchestrator.stt import WhisperSTT
            logging.warning("⚠️  Using mock STT implementation")
        except ImportError as e2:
            logging.error(f"❌ Failed to import STT: {e2}")
            raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def check_port_available(host: str, port: int) -> bool:
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0  # Port is available if connection failed
    except Exception:
        return False

def is_orchestrator_running() -> bool:
    """Check if orchestrator is already running on expected ports"""
    return (
        not check_port_available('localhost', 9000) and 
        not check_port_available('localhost', 9001) and
        not check_port_available('localhost', 8080)
    )

def start_orchestrator_if_needed() -> bool:
    """Start orchestrator if it's not already running. Returns True if orchestrator is available."""
    if is_orchestrator_running():
        logging.info("✅ Orchestrator is already running")
        return True
    
    logging.info("🚀 Starting WebSocket Orchestrator (single instance)...")
    
    try:
        orchestrator_script = Path(parent_dir) / "ws_orchestrator_service.py"
        if not orchestrator_script.exists():
            logging.error(f"❌ Orchestrator script not found: {orchestrator_script}")
            return False
        
        # Start orchestrator in background
        process = subprocess.Popen([
            sys.executable, str(orchestrator_script)
        ], cwd=parent_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for orchestrator to start (check ports become unavailable)
        max_wait = 10  # 10 seconds max wait
        wait_interval = 0.5
        waited = 0
        
        while waited < max_wait:
            if is_orchestrator_running():
                logging.info("✅ Orchestrator started successfully (shared instance)")
                return True
            time.sleep(wait_interval)
            waited += wait_interval
        
        logging.error("❌ Orchestrator failed to start within timeout")
        return False
        
    except Exception as e:
        logging.error(f"❌ Error starting orchestrator: {e}")
        return False

class WebSocketSTTService:
    """WebSocket-enabled STT service that converts existing HTTP service to streaming WebSocket"""
    
    def __init__(self, model_name: str = "base", device: str = "cpu"):
        self.service_id = "stt_whisper_ws"
        self.service_name = "WebSocket Whisper STT"
        self.websocket = None
        self.orchestrator_url = "ws://localhost:9001"  # Service port
        self.running = False
        
        # Initialize Whisper STT
        self.whisper_stt = None
        self.model_name = model_name
        self.device = device
        
        # Session management for streaming
        self.sessions = {}  # session_id -> session_data
        
        # Service registration info
        self.registration = ServiceRegistration(
            service_id=self.service_id,
            service_name=self.service_name,
            service_type="stt",
            version="1.0.0",
            endpoint="localhost",
            websocket_port=8001,
            http_port=8001,
            capabilities=ServiceCapabilities(
                realtime=True,
                streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"],
                max_concurrent=5,
                latency_ms=200
            ),
            metadata={
                "model": model_name,
                "device": device,
                "implementation": "whisper",
                "supported_languages": "100+",
                "supported_formats": ["wav", "mp3", "flac", "m4a", "ogg"],
                "input_types": ["audio/wav", "audio/mp3", "audio/base64"],
                "output_types": ["text/plain"],
                "status": "starting"
            }
        )
    
    async def initialize_stt(self):
        """Initialize the STT engine"""
        try:
            logging.info(f"[STT-WS] Initializing Whisper STT (model: {self.model_name}, device: {self.device})")
            
            self.whisper_stt = WhisperSTT(
                model_name=self.model_name,
                device=self.device
            )
            
            # Check if we're using real or mock implementation
            if hasattr(self.whisper_stt, '_use_real') and self.whisper_stt._use_real:
                logging.info("✅ Whisper STT initialized with REAL transcription!")
                self.registration.metadata["implementation_type"] = "real"
            else:
                logging.warning("⚠️  Whisper STT using MOCK transcription")
                self.registration.metadata["implementation_type"] = "mock"
            
            self.registration.status = "ready"
            return True
            
        except Exception as e:
            logging.error(f"❌ STT initialization failed: {e}")
            self.registration.status = "error"
            self.registration.metadata["error"] = str(e)
            return False
    
    async def connect_to_orchestrator(self):
        """Connect to WebSocket orchestrator and register service"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries and not self.running:
            try:
                logging.info(f"[STT-WS] Connecting to orchestrator: {self.orchestrator_url}")
                
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
                logging.info("✅ Connected to orchestrator and registered STT service")
                
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
            
            print(f"[STT-WS] DEBUG: handle_message called with type='{message_type}', session_id='{session_id}'")
            logging.info(f"[STT-WS] DEBUG: handle_message called with type='{message_type}', session_id='{session_id}'")
            
            print(f"[STT-WS] DEBUG: Checking message_type == 'audio_chunk': {message_type == 'audio_chunk'}")
            print(f"[STT-WS] DEBUG: message_type repr: {repr(message_type)}")
            
            if message_type == "audio_chunk":
                print(f"[STT-WS] DEBUG: Processing audio_chunk message")
                await self.handle_audio_chunk(data, session_id)
            elif message_type == "audio_file":
                print(f"[STT-WS] DEBUG: Processing audio_file message") 
                await self.handle_audio_file(data, session_id)
            elif message_type == "transcribe_request":
                print(f"[STT-WS] DEBUG: Processing transcribe_request message")
                await self.handle_transcribe_request(data, session_id)
            elif message_type == "session_start":
                print(f"[STT-WS] DEBUG: Processing session_start message")
                await self.handle_session_start(data, session_id)
            elif message_type == "session_end":
                print(f"[STT-WS] DEBUG: Processing session_end message")
                await self.handle_session_end(data, session_id)
            else:
                print(f"[STT-WS] WARNING: Unknown message type: {message_type}")
                logging.warning(f"[STT-WS] Unknown message type: {message_type}")
                
        except Exception as e:
            print(f"[STT-WS] ERROR: Error handling message: {e}")
            logging.error(f"[STT-WS] Error handling message: {e}")
            await self.send_error(session_id, f"Message handling error: {e}")
    
    async def handle_audio_chunk(self, data: Dict[str, Any], session_id: str):
        """Handle streaming audio chunks"""
        print(f"[STT-WS] DEBUG: handle_audio_chunk ENTRY - session: {session_id}")
        print(f"[STT-WS] DEBUG: data keys: {list(data.keys()) if data else 'None'}")
        
        try:
            print(f"[STT-WS] DEBUG: handle_audio_chunk starting processing for session {session_id}")
            logging.info(f"[STT-WS] DEBUG: handle_audio_chunk called for session {session_id}")
            
            audio_base64 = data.get("audio_data")
            chunk_index = data.get("chunk_index", 0)
            is_final = data.get("is_final", False)
            
            print(f"[STT-WS] DEBUG: Audio data length: {len(audio_base64) if audio_base64 else 0}, chunk_index: {chunk_index}, is_final: {is_final}")
            logging.info(f"[STT-WS] DEBUG: Audio data length: {len(audio_base64) if audio_base64 else 0}, chunk_index: {chunk_index}, is_final: {is_final}")
            
            if not audio_base64:
                print(f"[STT-WS] ERROR: No audio data in chunk for session {session_id}")
                logging.warning(f"[STT-WS] DEBUG: No audio data in chunk for session {session_id}")
                await self.send_error(session_id, "No audio data in chunk")
                return
            
            # Initialize session if needed
            if session_id not in self.sessions:
                print(f"[STT-WS] DEBUG: Creating new session {session_id}")
                logging.info(f"[STT-WS] DEBUG: Creating new session {session_id}")
                self.sessions[session_id] = {
                    "audio_chunks": [],
                    "start_time": time.time(),
                    "language": data.get("language", "auto"),
                    "task": data.get("task", "transcribe")
                }
            else:
                print(f"[STT-WS] DEBUG: Using existing session {session_id}")
            
            session = self.sessions[session_id]
            print(f"[STT-WS] DEBUG: Adding chunk to session {session_id}")
            session["audio_chunks"].append({
                "data": audio_base64,
                "index": chunk_index,
                "timestamp": time.time()
            })
            
            print(f"[STT-WS] DEBUG: Session {session_id} now has {len(session['audio_chunks'])} chunks")
            logging.info(f"[STT-WS] DEBUG: Session {session_id} now has {len(session['audio_chunks'])} chunks")
            
            # If this is the final chunk or we have enough data, process it
            chunk_count = len(session["audio_chunks"])
            should_process = is_final or chunk_count >= 5
            print(f"[STT-WS] DEBUG: Processing decision - is_final: {is_final}, chunks: {chunk_count}, should_process: {should_process}")
            
            if should_process:
                print(f"[STT-WS] DEBUG: STARTING process_accumulated_audio for session {session_id} (is_final: {is_final}, chunks: {chunk_count})")
                logging.info(f"[STT-WS] DEBUG: Processing accumulated audio for session {session_id} (is_final: {is_final}, chunks: {chunk_count})")
                await self.process_accumulated_audio(session_id, is_final)
                print(f"[STT-WS] DEBUG: COMPLETED process_accumulated_audio for session {session_id}")
            else:
                print(f"[STT-WS] DEBUG: Not processing yet - waiting for more chunks or final flag")
                
        except Exception as e:
            logging.error(f"[STT-WS] Error handling audio chunk: {e}")
            await self.send_error(session_id, f"Audio chunk processing error: {e}")
    
    async def handle_audio_file(self, data: Dict[str, Any], session_id: str):
        """Handle complete audio file transcription"""
        try:
            audio_base64 = data.get("audio_data")
            language = data.get("language", "auto")
            task = data.get("task", "transcribe")
            
            if not audio_base64:
                await self.send_error(session_id, "No audio data provided")
                return
            
            start_time = time.time()
            
            # Decode and save audio to temporary file
            audio_data = base64.b64decode(audio_base64)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # Transcribe the audio
                text = await self.whisper_stt.transcribe_file(temp_path)
                processing_time = time.time() - start_time
                
                # Send transcription result
                response = {
                    "type": "transcription_result",
                    "session_id": session_id,
                    "data": {
                        "text": text,
                        "language_detected": "auto",  # Mock for now
                        "processing_time_seconds": processing_time,
                        "confidence": 0.95,  # Mock confidence
                        "is_final": True,
                        "segments": []
                    }
                }
                
                await self.websocket.send(json.dumps(response))
                logging.info(f"[STT-WS] Transcription completed in {processing_time:.2f}s: '{text[:100]}...'")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logging.error(f"[STT-WS] Error handling audio file: {e}")
            await self.send_error(session_id, f"Audio file processing error: {e}")
    
    async def process_accumulated_audio(self, session_id: str, is_final: bool = False):
        """Process accumulated audio chunks for a session"""
        try:
            logging.info(f"[STT-WS] DEBUG: process_accumulated_audio called for session {session_id}, is_final: {is_final}")
            
            session = self.sessions.get(session_id)
            if not session or not session["audio_chunks"]:
                logging.warning(f"[STT-WS] DEBUG: No session or chunks found for session {session_id}")
                return
            
            start_time = time.time()
            logging.info(f"[STT-WS] DEBUG: Processing {len(session['audio_chunks'])} chunks for session {session_id}")
            
            # Combine all audio chunks
            combined_audio = b""
            for chunk in session["audio_chunks"]:
                audio_data = base64.b64decode(chunk["data"])
                combined_audio += audio_data
            
            logging.info(f"[STT-WS] DEBUG: Combined audio size: {len(combined_audio)} bytes")
            
            # Save combined audio as proper WAV file with headers
            import wave
            import struct
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_path = temp_file.name
            
            # Create proper WAV file with headers
            try:
                print(f"[STT-WS] DEBUG: Creating WAV file with {len(combined_audio)} bytes of raw audio")
                
                # Convert bytes to 16-bit signed integers
                if len(combined_audio) % 2 != 0:
                    # Pad with zero byte if odd length
                    combined_audio += b'\x00'
                
                # Create WAV file with proper headers
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit (2 bytes per sample)
                    wav_file.setframerate(16000)  # 16kHz sample rate
                    wav_file.writeframes(combined_audio)
                
                print(f"[STT-WS] DEBUG: Successfully created WAV file: {temp_path}")
                
            except Exception as wav_error:
                print(f"[STT-WS] ERROR: Failed to create WAV file: {wav_error}")
                # Fallback: try direct write (for already formatted WAV data)
                with open(temp_path, 'wb') as f:
                    f.write(combined_audio)
                print(f"[STT-WS] DEBUG: Used fallback direct write for WAV file")
            
            logging.info(f"[STT-WS] DEBUG: Saved audio to temp file: {temp_path}")
            
            try:
                # Transcribe the combined audio
                print(f"[STT-WS] DEBUG: Starting Whisper transcription for session {session_id}")
                logging.info(f"[STT-WS] DEBUG: Starting Whisper transcription for session {session_id}")
                
                text = await self.whisper_stt.transcribe_file(temp_path)
                processing_time = time.time() - start_time
                
                print(f"[STT-WS] DEBUG: Transcription result: '{text}' (time: {processing_time:.2f}s)")
                logging.info(f"[STT-WS] DEBUG: Transcription completed: '{text}' (time: {processing_time:.2f}s)")
                
                # Send partial or final transcription - FIXED MESSAGE TYPE
                message_type = "transcript_partial" if not is_final else "transcript_final"
                response = {
                    "type": message_type,
                    "session_id": session_id,
                    "data": {
                        "text": text,
                        "language_detected": session["language"],
                        "processing_time_seconds": processing_time,
                        "confidence": 0.95,
                        "is_final": is_final,
                        "chunk_count": len(session["audio_chunks"])
                    }
                }
                
                print(f"[STT-WS] DEBUG: Preparing to send {message_type} response: {text[:50]}...")
                logging.info(f"[STT-WS] DEBUG: Sending {message_type} response: {text[:50]}...")
                
                try:
                    await self.websocket.send(json.dumps(response))
                    print(f"[STT-WS] DEBUG: {message_type} response sent successfully!")
                    logging.info(f"[STT-WS] DEBUG: {message_type} response sent successfully")
                except Exception as send_error:
                    print(f"[STT-WS] ERROR: Failed to send response: {send_error}")
                    logging.error(f"[STT-WS] ERROR: Failed to send response: {send_error}")
                    raise
                
                # Clear processed chunks if not final
                if not is_final:
                    session["audio_chunks"] = []
                
                logging.info(f"[STT-WS] {'Final' if is_final else 'Partial'} transcription: '{text[:100]}...'")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logging.error(f"[STT-WS] Error processing accumulated audio: {e}")
            await self.send_error(session_id, f"Audio processing error: {e}")
    
    async def handle_transcribe_request(self, data: Dict[str, Any], session_id: str):
        """Handle direct transcription request (backward compatibility)"""
        await self.handle_audio_file(data, session_id)
    
    async def handle_session_start(self, data: Dict[str, Any], session_id: str):
        """Handle session start"""
        self.sessions[session_id] = {
            "audio_chunks": [],
            "start_time": time.time(),
            "language": data.get("language", "auto"),
            "task": data.get("task", "transcribe"),
            "metadata": data.get("metadata", {})
        }
        
        response = {
            "type": "session_started",
            "session_id": session_id,
            "data": {
                "service": self.service_id,
                "ready": True
            }
        }
        await self.websocket.send(json.dumps(response))
        logging.info(f"[STT-WS] Session started: {session_id}")
    
    async def handle_session_end(self, data: Dict[str, Any], session_id: str):
        """Handle session end"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Process any remaining audio chunks
            if session["audio_chunks"]:
                await self.process_accumulated_audio(session_id, is_final=True)
            
            # Clean up session
            del self.sessions[session_id]
            
            response = {
                "type": "session_ended", 
                "session_id": session_id,
                "data": {
                    "service": self.service_id,
                    "duration": time.time() - session["start_time"]
                }
            }
            await self.websocket.send(json.dumps(response))
            logging.info(f"[STT-WS] Session ended: {session_id}")
    
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
                            "status": "healthy" if self.whisper_stt else "unhealthy",
                            "active_sessions": len(self.sessions),
                            "model": self.model_name,
                            "device": self.device,
                            "implementation": self.registration.metadata.get("implementation_type", "unknown"),
                            "timestamp": time.time()
                        }
                    }
                    await self.websocket.send(json.dumps(health_data))
                
                await asyncio.sleep(30)  # Send health update every 30 seconds
                
            except Exception as e:
                logging.error(f"[STT-WS] Error sending health update: {e}")
                break
    
    async def message_loop(self):
        """Main message handling loop with keep-alive"""
        print("[DEBUG] Starting message loop with keep-alive")
        print(f"[DEBUG] self.running = {self.running}")
        try:
            while self.running and self.websocket:
                print("[DEBUG] In while loop iteration")
                try:
                    # Wait for messages with timeout to allow for keep-alive checks
                    print("[DEBUG] Waiting for messages...")
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                    print(f"[DEBUG] Received message: {message}")
                    try:
                        logging.info(f"[STT-WS] DEBUG: Raw message received: {message[:200]}...")
                        data = json.loads(message)
                        logging.info(f"[STT-WS] DEBUG: Parsed message type: {data.get('type')}, session: {data.get('session_id')}")
                        await self.handle_message(data)
                    except json.JSONDecodeError:
                        print(f"[ERROR] Invalid JSON received: {message}")
                    except Exception as e:
                        print(f"[ERROR] Error processing message: {e}")
                        
                except asyncio.TimeoutError:
                    # No message received, continue loop (keep-alive)
                    print("[DEBUG] Keep-alive timeout, continuing...")
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("[DEBUG] WebSocket connection closed")
                    break
                except asyncio.CancelledError:
                    print("[DEBUG] WebSocket operation cancelled")
                    break
                except Exception as e:
                    print(f"[ERROR] Unexpected error in message receive: {e}")
                    import traceback
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
                    break
                    
        except asyncio.CancelledError:
            print("[DEBUG] Message loop cancelled")
        except Exception as e:
            print(f"[ERROR] Message loop outer error: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
        finally:
            print("[DEBUG] Message loop ended")
            self.running = False
    
    async def run(self):
        """Run the WebSocket STT service"""
        print("[DEBUG] Starting service initialization")
        
        # Initialize STT engine
        if not await self.initialize_stt():
            print("[ERROR] Failed to initialize STT engine")
            return False
        
        print("[DEBUG] STT engine initialized successfully")
        
        # Connect to orchestrator
        if not await self.connect_to_orchestrator():
            print("[ERROR] Failed to connect to orchestrator")
            return False
            
        print("[DEBUG] Connected to orchestrator successfully")
        
        health_task = None
        try:
            # Start health monitoring task
            print("[DEBUG] Starting health monitoring task")
            health_task = asyncio.create_task(self.send_health_update())
            
            # Start message handling
            print("[DEBUG] Starting message loop...")
            await self.message_loop()
            print("[DEBUG] Message loop completed!")
            
        except KeyboardInterrupt:
            print("[DEBUG] Received interrupt, shutting down...")
        except asyncio.CancelledError:
            print("[DEBUG] Service was cancelled, shutting down...")
        except Exception as e:
            print(f"[ERROR] Service error: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
        finally:
            self.running = False
            
            # Cancel health task
            if health_task and not health_task.done():
                print("[DEBUG] Cancelling health monitoring task...")
                health_task.cancel()
                try:
                    await health_task
                except asyncio.CancelledError:
                    pass
            
            # Close WebSocket connection
            if self.websocket:
                print("[DEBUG] Closing WebSocket connection...")
                try:
                    await self.websocket.close()
                except Exception:
                    pass
            
            # Clean up sessions
            self.sessions.clear()
            
            print("[DEBUG] Service shut down complete")
        
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
    safe_print("🎙️  Starting WebSocket Whisper STT Service...")
    
    # Auto-start orchestrator if needed
    if not start_orchestrator_if_needed():
        safe_print("❌ Cannot start without orchestrator")
        return False
    
    safe_print("📡 Connecting to orchestrator at ws://localhost:9001")
    safe_print("🔄 Converting HTTP STT service to WebSocket streaming")
    safe_print("-" * 60)
    
    service = WebSocketSTTService(
        model_name="base",  # Can be: tiny, base, small, medium, large
        device="cpu"  # Use cpu for stability
    )
    
    success = await service.run()
    
    if success:
        safe_print("✅ WebSocket STT service completed successfully")
    else:
        safe_print("❌ WebSocket STT service encountered errors")
        sys.exit(1)
        
    # Keep the main process alive (this shouldn't be reached with keep-alive message loop)
    safe_print("⚠️  Service exited unexpectedly from message loop")
    sys.exit(1)

if __name__ == "__main__":
    # Configure logging with more verbose output
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("[DEBUG] Starting service with debug logging")
    
    # Run the service
    asyncio.run(main())
