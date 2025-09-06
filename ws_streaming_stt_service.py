#!/usr/bin/env python3
"""
WebSocket Streaming STT Service

Real-time speech-to-text service that connects to the WebSocket orchestrator
and provides streaming transcription capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
import wave
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, AsyncGenerator
import websockets
from websockets.exceptions import ConnectionClosed, InvalidMessage

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import service registry components
from ws_service_registry import (
    ServiceRegistration, 
    ServiceCapabilities, 
    ServiceType,
    WebSocketServiceClient
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamingSTTService(WebSocketServiceClient):
    """WebSocket-based streaming STT service"""
    
    def __init__(self, model_name: str = "whisper-base", port: int = 8003):
        # Initialize service registration
        capabilities = ServiceCapabilities(
            realtime=True,
            streaming=True,
            languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
            max_concurrent=5
        )
        
        registration = ServiceRegistration(
            service_id=f"streaming_stt_{port}",
            service_type=ServiceType.STT.value,
            service_name="Streaming Whisper STT Service",
            version="1.0.0", 
            endpoint=f"ws://localhost:{port}",
            websocket_port=port,
            http_port=port + 100,  # HTTP port for health checks
            capabilities=capabilities,
            metadata={
                "model": model_name,
                "streaming": True,
                "chunk_size": 1024,
                "sample_rate": 16000
            }
        )
        
        super().__init__(registration)
        self.model_name = model_name
        self.port = port
        self.stt_engine = None
        self.active_sessions = {}
        
    async def initialize(self):
        """Initialize the STT engine"""
        try:
            # Try to import real Whisper implementation
            try:
                from voicebot_orchestrator.real_whisper_stt import WhisperSTT
                self.stt_engine = WhisperSTT(model_name=self.model_name)
                logger.info(f"✅ Initialized real Whisper STT with model: {self.model_name}")
            except ImportError:
                # Fallback to mock implementation for development
                from voicebot_orchestrator.stt import WhisperSTT
                self.stt_engine = WhisperSTT()
                logger.warning("⚠️  Using mock STT implementation")
                
            await self.stt_engine.initialize()
            logger.info("STT engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize STT engine: {e}")
            raise
    
    async def process_audio_stream(self, audio_data: bytes, session_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process streaming audio data and yield transcription results.
        
        Args:
            audio_data: Raw audio bytes
            session_id: Session identifier
            
        Yields:
            Transcription results with timestamps and confidence
        """
        try:
            # Initialize session if needed
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    'buffer': b'',
                    'chunk_count': 0,
                    'start_time': time.time()
                }
            
            session = self.active_sessions[session_id]
            session['buffer'] += audio_data
            session['chunk_count'] += 1
            
            # Process audio in chunks for streaming
            chunk_size = 1024 * 16  # 16KB chunks for real-time processing
            
            if len(session['buffer']) >= chunk_size:
                # Create temporary audio file for processing
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    try:
                        # Write audio data as WAV file
                        with wave.open(temp_file.name, 'wb') as wav_file:
                            wav_file.setnchannels(1)  # Mono
                            wav_file.setsampwidth(2)  # 16-bit
                            wav_file.setframerate(16000)  # 16kHz
                            wav_file.writeframes(session['buffer'][:chunk_size])
                        
                        # Transcribe the audio chunk
                        result = await self.stt_engine.transcribe_audio_async(temp_file.name)
                        
                        if result and result.get('text', '').strip():
                            # Calculate timing information
                            chunk_duration = len(session['buffer'][:chunk_size]) / (16000 * 2)  # seconds
                            timestamp = time.time() - session['start_time']
                            
                            # Yield streaming result
                            yield {
                                "type": "transcription",
                                "session_id": session_id,
                                "text": result['text'].strip(),
                                "confidence": result.get('confidence', 0.9),
                                "timestamp": timestamp,
                                "chunk_id": session['chunk_count'],
                                "is_partial": True,  # Streaming results are partial
                                "language": result.get('language', 'en')
                            }
                        
                        # Remove processed data from buffer
                        session['buffer'] = session['buffer'][chunk_size:]
                        
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass
                            
        except Exception as e:
            logger.error(f"Error processing audio stream for session {session_id}: {e}")
            yield {
                "type": "error",
                "session_id": session_id,
                "message": f"STT processing error: {str(e)}",
                "timestamp": time.time()
            }
    
    async def finalize_session(self, session_id: str) -> Dict[str, Any]:
        """
        Finalize a streaming session and return complete transcription.
        
        Args:
            session_id: Session to finalize
            
        Returns:
            Final transcription result
        """
        if session_id not in self.active_sessions:
            return {"type": "error", "message": "Session not found"}
        
        try:
            session = self.active_sessions[session_id]
            
            # Process any remaining audio data
            if len(session['buffer']) > 0:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    try:
                        # Write remaining audio data
                        with wave.open(temp_file.name, 'wb') as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(16000)
                            wav_file.writeframes(session['buffer'])
                        
                        # Get final transcription
                        result = await self.stt_engine.transcribe_audio_async(temp_file.name)
                        
                        # Clean up session
                        del self.active_sessions[session_id]
                        
                        return {
                            "type": "final_transcription",
                            "session_id": session_id,
                            "text": result.get('text', '').strip(),
                            "confidence": result.get('confidence', 0.9),
                            "total_chunks": session['chunk_count'],
                            "duration": time.time() - session['start_time'],
                            "language": result.get('language', 'en')
                        }
                        
                    finally:
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass
                            
        except Exception as e:
            logger.error(f"Error finalizing session {session_id}: {e}")
            return {
                "type": "error",
                "session_id": session_id,
                "message": f"Finalization error: {str(e)}"
            }
        
        # Clean up session even on error
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            
        return {"type": "error", "message": "No audio data to process"}
    
    async def handle_service_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle incoming messages from the orchestrator.
        
        Args:
            message: Incoming message
            
        Returns:
            Response message or None for streaming responses
        """
        try:
            message_type = message.get('type')
            session_id = message.get('session_id', str(uuid.uuid4()))
            
            if message_type == 'start_stream':
                # Initialize streaming session
                logger.info(f"Starting STT stream for session: {session_id}")
                return {
                    "type": "stream_started",
                    "session_id": session_id,
                    "service": "stt",
                    "capabilities": {
                        "streaming": True,
                        "languages": self.registration.capabilities.languages,
                        "sample_rate": 16000,
                        "chunk_size": 1024
                    }
                }
            
            elif message_type == 'audio_chunk':
                # Process streaming audio
                audio_data = message.get('data')
                if audio_data:
                    # Convert base64 or handle binary data
                    if isinstance(audio_data, str):
                        import base64
                        audio_bytes = base64.b64decode(audio_data)
                    else:
                        audio_bytes = audio_data
                    
                    # Stream transcription results back to orchestrator
                    async for result in self.process_audio_stream(audio_bytes, session_id):
                        await self.send_to_orchestrator(result)
                    
                return None  # Streaming response, no immediate reply
            
            elif message_type == 'end_stream':
                # Finalize the streaming session
                logger.info(f"Ending STT stream for session: {session_id}")
                return await self.finalize_session(session_id)
            
            elif message_type == 'health_check':
                return {
                    "type": "health_response",
                    "status": "healthy",
                    "active_sessions": len(self.active_sessions),
                    "service": "stt"
                }
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
                return {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return {
                "type": "error",
                "message": f"Message processing error: {str(e)}"
            }

async def main():
    """Main entry point for the streaming STT service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WebSocket Streaming STT Service")
    parser.add_argument("--port", type=int, default=8003, help="Service port")
    parser.add_argument("--model", default="whisper-base", help="Whisper model name")
    parser.add_argument("--orchestrator-host", default="localhost", help="Orchestrator host")
    parser.add_argument("--orchestrator-port", type=int, default=9001, help="Orchestrator service port")
    
    args = parser.parse_args()
    
    # Create and initialize the service
    service = StreamingSTTService(model_name=args.model, port=args.port)
    
    try:
        # Initialize STT engine
        await service.initialize()
        
        # Connect to orchestrator and run service
        orchestrator_uri = f"ws://{args.orchestrator_host}:{args.orchestrator_port}/service"
        await service.connect_and_serve(orchestrator_uri)
        
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        raise
    finally:
        # Cleanup
        if service.stt_engine and hasattr(service.stt_engine, 'cleanup'):
            await service.stt_engine.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
