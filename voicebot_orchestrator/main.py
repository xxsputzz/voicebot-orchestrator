"""
Main FastAPI application for voicebot orchestrator.
"""
import asyncio
import logging
import time
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json

from .config import settings
from .session_manager import SessionManager
from .stt import WhisperSTT
from .llm import MistralLLM
from .tts import KokoroTTS

# Sprint 3: Import metrics and analytics
try:
    from .metrics import metrics_collector, measure_async_latency
    from .analytics import analytics_engine
    METRICS_ENABLED = True
except ImportError:
    # Mock for development without dependencies
    METRICS_ENABLED = False
    class MockMetrics:
        def record_request(self, *args): pass
        def record_component_latency(self, *args): pass
        def update_active_sessions(self, *args): pass
        def start_metrics_server(self, *args): pass
    
    class MockAnalytics:
        def record_session(self, *args): pass
    
    metrics_collector = MockMetrics()
    analytics_engine = MockAnalytics()
    
    def measure_async_latency(component):
        def decorator(func):
            return func
        return decorator

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Initialize components
app = FastAPI(
    title="Voicebot Orchestrator",
    description="Real-time voicebot orchestration system",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
session_manager = SessionManager(
    timeout=settings.session_timeout,
    max_sessions=settings.max_concurrent_sessions
)

stt_service = WhisperSTT(
    model_name=settings.whisper_model,
    device=settings.whisper_device
)

llm_service = MistralLLM(
    model_path=settings.mistral_model_path,
    max_tokens=settings.mistral_max_tokens,
    temperature=settings.mistral_temperature
)

tts_service = KokoroTTS(
    voice=settings.kokoro_voice,
    language=settings.kokoro_language,
    speed=settings.kokoro_speed
)


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting Voicebot Orchestrator")
    logger.info(f"Server configuration: {settings.host}:{settings.port}")
    
    # Sprint 3: Start metrics server
    if METRICS_ENABLED:
        try:
            metrics_collector.start_metrics_server(8000)
            logger.info("Metrics server started on port 8000")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Shutting down Voicebot Orchestrator")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Voicebot Orchestrator API", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Sprint 3: Record API request
    start_time = time.time()
    
    health_data = {
        "status": "healthy",
        "services": {
            "stt": "ready",
            "llm": "ready", 
            "tts": "ready"
        },
        "metrics_enabled": METRICS_ENABLED
    }
    
    if METRICS_ENABLED:
        latency = time.time() - start_time
        metrics_collector.record_request("health", "/health", latency)
    
    return health_data


@app.get("/sessions")
async def list_sessions():
    """List active sessions."""
    start_time = time.time()
    
    active_sessions = await session_manager.list_active_sessions()
    result = {"active_sessions": active_sessions, "count": len(active_sessions)}
    
    # Sprint 3: Update session metrics
    if METRICS_ENABLED:
        metrics_collector.update_active_sessions(len(active_sessions))
        latency = time.time() - start_time
        metrics_collector.record_request("system", "/sessions", latency)
    
    return result


@app.post("/sessions/{session_id}")
async def create_session(session_id: str, metadata: Dict[str, Any] = None):
    """Create a new session."""
    try:
        session_data = await session_manager.create_session(session_id, metadata)
        return {
            "session_id": session_data.session_id,
            "state": session_data.state.value,
            "created_at": session_data.created_at
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/sessions/{session_id}")
async def end_session(session_id: str):
    """End a session."""
    success = await session_manager.end_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session ended successfully"}


@app.websocket("/ws/{session_id}")
async def call_session(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time voice conversation.
    
    Args:
        websocket: WebSocket connection
        session_id: Unique session identifier
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for session: {session_id}")
    
    # Sprint 3: Session start metrics
    session_start_time = time.time()
    message_count = 0
    total_stt_time = 0.0
    total_llm_time = 0.0
    total_tts_time = 0.0
    error_count = 0
    
    # Get or create session
    session = await session_manager.get_session(session_id)
    if not session:
        try:
            session = await session_manager.create_session(session_id)
        except ValueError as e:
            await websocket.close(code=1008, reason=str(e))
            return
    
    # Semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(2)
    
    try:
        while True:
            # Receive audio data
            audio_bytes = await websocket.receive_bytes()
            logger.debug(f"Received audio data: {len(audio_bytes)} bytes")
            
            async with semaphore:
                try:
                    # Update session activity
                    await session_manager.update_activity(session_id)
                    message_count += 1
                    
                    # Sprint 3: Measure component latencies
                    stt_start = time.time()
                    text = await stt_service.transcribe_audio(audio_bytes)
                    stt_latency = time.time() - stt_start
                    total_stt_time += stt_latency
                    
                    if METRICS_ENABLED:
                        metrics_collector.record_component_latency("stt", session_id, stt_latency)
                    
                    logger.info(f"STT result: {text}")
                    
                    # Validate input
                    if not await llm_service.validate_input(text):
                        error_response = "I'm sorry, I cannot process that input."
                        
                        tts_start = time.time()
                        audio_out = await tts_service.synthesize_speech(error_response)
                        tts_latency = time.time() - tts_start
                        total_tts_time += tts_latency
                        
                        if METRICS_ENABLED:
                            metrics_collector.record_component_latency("tts", session_id, tts_latency)
                            metrics_collector.record_pipeline_error("llm", "ValidationError")
                        
                        error_count += 1
                        await websocket.send_bytes(audio_out)
                        continue
                    
                    # Get conversation history
                    session = await session_manager.get_session(session_id)
                    conversation_history = session.conversation_history if session else []
                    
                    # Language Model inference
                    llm_start = time.time()
                    response_text = await llm_service.generate_response(text, conversation_history)
                    llm_latency = time.time() - llm_start
                    total_llm_time += llm_latency
                    
                    if METRICS_ENABLED:
                        metrics_collector.record_component_latency("llm", session_id, llm_latency)
                    
                    logger.info(f"LLM response: {response_text}")
                    
                    # Text-to-Speech
                    tts_start = time.time()
                    audio_out = await tts_service.synthesize_speech(response_text)
                    tts_latency = time.time() - tts_start
                    total_tts_time += tts_latency
                    
                    if METRICS_ENABLED:
                        metrics_collector.record_component_latency("tts", session_id, tts_latency)
                    
                    logger.debug(f"TTS output: {len(audio_out)} bytes")
                    
                    # Add to conversation history
                    await session_manager.add_to_history(session_id, text, response_text)
                    
                    # Send response
                    await websocket.send_bytes(audio_out)
                    
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    error_count += 1
                    if METRICS_ENABLED:
                        metrics_collector.record_pipeline_error("pipeline", type(e).__name__)
                    error_response = "I'm sorry, there was an error processing your request."
                    try:
                        audio_out = await tts_service.synthesize_speech(error_response)
                        await websocket.send_bytes(audio_out)
                    except Exception:
                        # If TTS also fails, send text error
                        await websocket.send_text(json.dumps({"error": error_response}))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Sprint 3: Record session analytics
        session_duration = time.time() - session_start_time
        
        if METRICS_ENABLED:
            metrics_collector.record_handle_time(session_duration)
            
            # Record session analytics
            session_analytics = {
                "session_id": session_id,
                "duration": session_duration,
                "stt_latency": total_stt_time / max(message_count, 1),
                "llm_latency": total_llm_time / max(message_count, 1),
                "tts_latency": total_tts_time / max(message_count, 1),
                "total_latency": (total_stt_time + total_llm_time + total_tts_time) / max(message_count, 1),
                "message_count": message_count,
                "error_count": error_count,
                "cache_hits": 0,  # TODO: Track from cache service
                "cache_misses": 0,  # TODO: Track from cache service
                "first_call_resolution": error_count == 0 and message_count > 0,
                "customer_satisfaction": 4.2,  # TODO: Get from survey
                "word_count": message_count * 10,  # Estimate
                "intent": "banking_support",  # TODO: Extract from LLM
                "outcome": "resolved" if error_count == 0 else "error"
            }
            
            analytics_engine.record_session(session_analytics)
        
        # Clean up session
        await session_manager.end_session(session_id)
        logger.info(f"Session {session_id} ended (duration: {session_duration:.2f}s, messages: {message_count})")


@app.post("/stt/test")
async def test_stt(file_path: str):
    """Test STT functionality with an audio file."""
    try:
        result = await stt_service.transcribe_file(file_path)
        return {"transcription": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/tts/test")
async def test_tts(text: str, voice: str = None, language: str = None):
    """Test TTS functionality with text input."""
    try:
        # Update voice parameters if provided
        if voice or language:
            tts_service.set_voice_parameters(voice=voice, language=language)
        
        # Validate text
        if not await tts_service.validate_text(text):
            raise HTTPException(status_code=400, detail="Invalid text input")
        
        audio_data = await tts_service.synthesize_speech(text)
        
        return {
            "message": "TTS synthesis successful",
            "audio_size": len(audio_data),
            "text": text
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "voicebot_orchestrator.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=True
    )
