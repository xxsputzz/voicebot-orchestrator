#!/usr/bin/env python3
"""
WebSocket Streaming TTS Service

Real-time text-to-speech service that connects to the WebSocket orchestrator
and provides chunk-by-chunk audio streaming.
"""

import asyncio
import json
import logging
import sys
import time
import uuid
import tempfile
import os
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

class StreamingTTSService(WebSocketServiceClient):
    """WebSocket-based streaming TTS service with chunk streaming"""
    
    def __init__(self, model_name: str = "tortoise", port: int = 8015):
        # Initialize service registration
        capabilities = ServiceCapabilities(
            streaming=True,
            languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
            voice_models=self._get_voice_models_for_engine(model_name),
            max_concurrent=3  # TTS is resource intensive
        )
        
        registration = ServiceRegistration(
            service_id=f"streaming_tts_{port}",
            service_type=ServiceType.TTS.value,
            service_name=f"Streaming {model_name.title()} TTS Service",
            version="1.0.0",
            endpoint=f"ws://localhost:{port}",
            websocket_port=port,
            http_port=port + 100,  # HTTP port for health checks
            capabilities=capabilities,
            metadata={
                "engine": model_name,
                "streaming": True,
                "chunk_size": 1024,
                "sample_rate": 22050,
                "audio_format": "wav"
            }
        )
        
        super().__init__(registration)
        self.model_name = model_name
        self.port = port
        self.tts_engine = None
        self.active_syntheses = {}
        
    def _get_voice_models_for_engine(self, engine_name: str) -> List[str]:
        """Get available voice models for the specified engine"""
        if engine_name == "tortoise":
            return ["angie", "tom", "pat", "william", "deniro", "freeman", "halle", "geralt", "myself"]
        elif engine_name == "kokoro":
            return ["af_bella", "af_nicole", "af_sarah", "am_adam", "am_michael"]
        elif engine_name == "coqui":
            return ["female_1", "male_1", "female_2", "male_2"]
        else:
            return ["default"]
            
    async def initialize(self):
        """Initialize the TTS engine"""
        try:
            # Import based on model name
            if self.model_name == "tortoise":
                try:
                    from tortoise_tts_implementation_real import TortoiseTTSReal
                    self.tts_engine = TortoiseTTSReal()
                    logger.info("✅ Initialized Tortoise TTS Real implementation")
                except ImportError:
                    from tortoise_tts_implementation import TortoiseTTS
                    self.tts_engine = TortoiseTTS()
                    logger.warning("⚠️  Using Tortoise TTS fallback implementation")
                    
            elif self.model_name == "kokoro":
                try:
                    # Import kokoro TTS (assuming implementation exists)
                    from kokoro_tts import KokoroTTS
                    self.tts_engine = KokoroTTS()
                    logger.info("✅ Initialized Kokoro TTS")
                except ImportError:
                    logger.error("Kokoro TTS not available, using fallback")
                    from tortoise_tts_implementation import TortoiseTTS
                    self.tts_engine = TortoiseTTS()
                    
            elif self.model_name == "coqui":
                try:
                    # Import Coqui TTS (assuming implementation exists)
                    from coqui_tts import CoquiTTS
                    self.tts_engine = CoquiTTS()
                    logger.info("✅ Initialized Coqui TTS")
                except ImportError:
                    logger.error("Coqui TTS not available, using fallback")
                    from tortoise_tts_implementation import TortoiseTTS
                    self.tts_engine = TortoiseTTS()
                    
            else:
                # Default to Tortoise
                from tortoise_tts_implementation import TortoiseTTS
                self.tts_engine = TortoiseTTS()
                logger.warning(f"⚠️  Unknown TTS engine {self.model_name}, using Tortoise fallback")
            
            await self.tts_engine.initialize()
            logger.info("TTS engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            raise
    
    async def stream_tts_synthesis(self, text: str, voice: str, synthesis_id: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream TTS synthesis chunk by chunk.
        
        Args:
            text: Text to synthesize
            voice: Voice model to use
            synthesis_id: Synthesis session identifier
            **kwargs: Additional parameters (speed, pitch, etc.)
            
        Yields:
            Audio chunk streaming results
        """
        try:
            # Initialize synthesis session
            if synthesis_id not in self.active_syntheses:
                self.active_syntheses[synthesis_id] = {
                    'start_time': time.time(),
                    'chunk_count': 0,
                    'total_duration': 0
                }
            
            session = self.active_syntheses[synthesis_id]
            
            # Prepare synthesis parameters
            tts_params = {
                'voice': voice,
                'speed': kwargs.get('speed', 1.0),
                'temperature': kwargs.get('temperature', 0.7),
                'enable_streaming': True
            }
            
            logger.info(f"Starting TTS synthesis for session {synthesis_id}: '{text[:50]}...'")
            
            # Start synthesis with streaming
            try:
                # Method 1: Direct streaming if supported
                if hasattr(self.tts_engine, 'synthesize_streaming'):
                    async for audio_chunk in self.tts_engine.synthesize_streaming(text, **tts_params):
                        session['chunk_count'] += 1
                        
                        # Convert audio chunk to base64 for WebSocket transmission
                        import base64
                        if isinstance(audio_chunk, bytes):
                            audio_data = base64.b64encode(audio_chunk).decode('utf-8')
                        else:
                            # Handle numpy arrays or other formats
                            import numpy as np
                            if isinstance(audio_chunk, np.ndarray):
                                audio_bytes = audio_chunk.tobytes()
                            else:
                                audio_bytes = bytes(audio_chunk)
                            audio_data = base64.b64encode(audio_bytes).decode('utf-8')
                        
                        yield {
                            "type": "audio_chunk",
                            "synthesis_id": synthesis_id,
                            "audio_data": audio_data,
                            "chunk_id": session['chunk_count'],
                            "timestamp": time.time(),
                            "format": "wav",
                            "sample_rate": 22050,
                            "finished": False
                        }
                
                # Method 2: Fallback - synthesize full audio and chunk it
                else:
                    # Generate full audio first
                    output_path = await self.tts_engine.synthesize_audio_async(text, **tts_params)
                    
                    if output_path and os.path.exists(output_path):
                        # Read and chunk the audio file
                        chunk_size = 4096  # 4KB chunks
                        
                        with open(output_path, 'rb') as audio_file:
                            while True:
                                chunk = audio_file.read(chunk_size)
                                if not chunk:
                                    break
                                
                                session['chunk_count'] += 1
                                
                                # Convert to base64
                                import base64
                                audio_data = base64.b64encode(chunk).decode('utf-8')
                                
                                yield {
                                    "type": "audio_chunk",
                                    "synthesis_id": synthesis_id,
                                    "audio_data": audio_data,
                                    "chunk_id": session['chunk_count'],
                                    "timestamp": time.time(),
                                    "format": "wav",
                                    "sample_rate": 22050,
                                    "finished": False
                                }
                                
                                # Small delay between chunks for streaming effect
                                await asyncio.sleep(0.01)
                        
                        # Clean up temporary file
                        try:
                            os.unlink(output_path)
                        except:
                            pass
                    else:
                        raise Exception("TTS synthesis failed - no output generated")
                
                # Send completion signal
                session['total_duration'] = time.time() - session['start_time']
                yield {
                    "type": "synthesis_complete",
                    "synthesis_id": synthesis_id,
                    "total_chunks": session['chunk_count'],
                    "duration": session['total_duration'],
                    "timestamp": time.time(),
                    "finished": True
                }
                
            except Exception as synthesis_error:
                logger.error(f"TTS synthesis error for session {synthesis_id}: {synthesis_error}")
                yield {
                    "type": "error",
                    "synthesis_id": synthesis_id,
                    "message": f"TTS synthesis error: {str(synthesis_error)}",
                    "timestamp": time.time()
                }
            
        except Exception as e:
            logger.error(f"Error in TTS streaming for synthesis {synthesis_id}: {e}")
            yield {
                "type": "error",
                "synthesis_id": synthesis_id,
                "message": f"TTS streaming error: {str(e)}",
                "timestamp": time.time()
            }
        
        finally:
            # Clean up session
            if synthesis_id in self.active_syntheses:
                del self.active_syntheses[synthesis_id]
    
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
            synthesis_id = message.get('synthesis_id', str(uuid.uuid4()))
            
            if message_type == 'synthesize_speech':
                # Handle streaming speech synthesis
                text = message.get('text', '')
                voice = message.get('voice', 'angie')
                
                if not text:
                    return {
                        "type": "error",
                        "message": "No text provided for synthesis"
                    }
                
                logger.info(f"Starting TTS synthesis for session: {synthesis_id}")
                
                # Extract parameters
                speed = message.get('speed', 1.0)
                temperature = message.get('temperature', 0.7)
                
                # Stream audio chunks back to orchestrator
                async for result in self.stream_tts_synthesis(
                    text, voice, synthesis_id,
                    speed=speed,
                    temperature=temperature
                ):
                    await self.send_to_orchestrator(result)
                
                return None  # Streaming response, no immediate reply
            
            elif message_type == 'get_voices':
                # Return available voice models
                return {
                    "type": "voices_list",
                    "voices": self.registration.capabilities.voice_models,
                    "engine": self.model_name
                }
            
            elif message_type == 'cancel_synthesis':
                # Cancel ongoing synthesis
                if synthesis_id in self.active_syntheses:
                    del self.active_syntheses[synthesis_id]
                
                return {
                    "type": "synthesis_cancelled",
                    "synthesis_id": synthesis_id
                }
            
            elif message_type == 'health_check':
                return {
                    "type": "health_response",
                    "status": "healthy",
                    "active_syntheses": len(self.active_syntheses),
                    "service": "tts",
                    "engine": self.model_name,
                    "available_voices": len(self.registration.capabilities.voice_models)
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
    """Main entry point for the streaming TTS service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WebSocket Streaming TTS Service")
    parser.add_argument("--port", type=int, default=8015, help="Service port")
    parser.add_argument("--engine", default="tortoise", help="TTS engine name")
    parser.add_argument("--orchestrator-host", default="localhost", help="Orchestrator host")
    parser.add_argument("--orchestrator-port", type=int, default=9001, help="Orchestrator service port")
    
    args = parser.parse_args()
    
    # Create and initialize the service
    service = StreamingTTSService(model_name=args.engine, port=args.port)
    
    try:
        # Initialize TTS engine
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
        if service.tts_engine and hasattr(service.tts_engine, 'cleanup'):
            await service.tts_engine.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
