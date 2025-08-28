"""
Orchestrator Core Microservice

Central coordination service for the voicebot orchestration platform.
Handles session management, workflow orchestration, and inter-service communication.
"""

import asyncio
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

# FastAPI and WebSocket imports (would be real in production)
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    # Mock implementations for restricted environment
    class FastAPI:
        def __init__(self, **kwargs): pass
        def get(self, path): return lambda f: f
        def post(self, path): return lambda f: f
        def websocket(self, path): return lambda f: f
        def add_middleware(self, *args, **kwargs): pass
    
    class HTTPException(Exception):
        def __init__(self, status_code, detail): pass
    
    class WebSocket:
        async def accept(self): pass
        async def send_json(self, data): pass
        async def receive_json(self): return {}
        async def close(self): pass
    
    class WebSocketDisconnect(Exception): pass
    
    class CORSMiddleware: pass
    
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def Field(**kwargs): return None
    
    class uvicorn:
        @staticmethod
        def run(*args, **kwargs): pass

# Internal imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from voicebot_orchestrator.session_manager import SessionManager as VoicebotSessionManager
from voicebot_orchestrator.enhanced_llm import create_enhanced_llm
from voicebot_orchestrator.analytics import AnalyticsEngine


# Configuration
PORT = int(os.getenv("ORCHESTRATOR_PORT", "8000"))
HOST = os.getenv("ORCHESTRATOR_HOST", "0.0.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Logging setup
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class SessionStartRequest(BaseModel):
    session_id: str = Field(..., min_length=3, max_length=64)
    phone_number: Optional[str] = None
    customer_id: Optional[str] = None
    domain: str = Field(default="banking")
    config: Optional[Dict[str, Any]] = None


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    created_at: str
    last_activity: str
    domain: str
    customer_data: Optional[Dict[str, Any]] = None


class CallProcessingRequest(BaseModel):
    session_id: str
    audio_data: str  # Base64 encoded audio
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CallProcessingResponse(BaseModel):
    session_id: str
    response_audio: str  # Base64 encoded audio response
    transcript: str
    response_text: str
    processing_time: float
    cache_hit: bool
    adapter_used: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    timestamp: str
    services: Dict[str, str]


# Global state
app_state = {
    "session_manager": None,
    "llm": None,
    "analytics": None,
    "active_connections": {},
    "service_urls": {
        "stt_service": os.getenv("STT_SERVICE_URL", "http://localhost:8001"),
        "llm_service": os.getenv("LLM_SERVICE_URL", "http://localhost:8002"),
        "tts_service": os.getenv("TTS_SERVICE_URL", "http://localhost:8003"),
        "cache_service": os.getenv("CACHE_SERVICE_URL", "http://localhost:8004"),
        "analytics_service": os.getenv("ANALYTICS_SERVICE_URL", "http://localhost:8005")
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup
    logger.info("Starting Orchestrator Core microservice...")
    
    try:
        # Initialize services
        app_state["session_manager"] = VoicebotSessionManager()
        app_state["llm"] = create_enhanced_llm(
            enable_cache=True,
            enable_adapters=True
        )
        app_state["analytics"] = AnalyticsEngine()
        
        # Setup banking domain
        await app_state["llm"].setup_banking_domain()
        
        logger.info("Orchestrator Core initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Orchestrator Core...")
    
    # Close active WebSocket connections
    for connection in app_state["active_connections"].values():
        try:
            await connection.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket connection: {e}")
    
    logger.info("Orchestrator Core shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Voicebot Orchestrator Core",
    description="Central coordination service for voicebot operations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancer and monitoring."""
    try:
        # Check service availability
        services_status = {}
        for service_name, service_url in app_state["service_urls"].items():
            try:
                # In real implementation, would make HTTP request to service health endpoint
                services_status[service_name] = "healthy"
            except Exception:
                services_status[service_name] = "unhealthy"
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            environment=ENVIRONMENT,
            timestamp=datetime.utcnow().isoformat(),
            services=services_status
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    try:
        # In real implementation, would return Prometheus format metrics
        metrics_data = {
            "sessions_active": len(app_state.get("active_connections", {})),
            "sessions_total": 0,  # Would get from analytics
            "response_time_avg": 0.250,  # Would calculate from recent requests
            "cache_hit_rate": 0.35,  # Would get from cache service
            "error_rate": 0.02  # Would calculate from error logs
        }
        
        return {"metrics": metrics_data, "timestamp": datetime.utcnow().isoformat()}
    
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")


@app.post("/sessions", response_model=SessionStatusResponse)
async def create_session(request: SessionStartRequest):
    """Create a new voicebot session."""
    try:
        logger.info(f"Creating session: {request.session_id}")
        
        # Create session via session manager
        session = await app_state["session_manager"].create_session(
            session_id=request.session_id,
            customer_data={
                "phone_number": request.phone_number,
                "customer_id": request.customer_id,
                "domain": request.domain
            }
        )
        
        # Log to analytics
        await app_state["analytics"].log_event(
            event_type="session_created",
            session_id=request.session_id,
            metadata={
                "domain": request.domain,
                "environment": ENVIRONMENT,
                "microservice": "orchestrator-core"
            }
        )
        
        return SessionStatusResponse(
            session_id=request.session_id,
            status="active",
            created_at=datetime.utcnow().isoformat(),
            last_activity=datetime.utcnow().isoformat(),
            domain=request.domain,
            customer_data={
                "phone_number": request.phone_number,
                "customer_id": request.customer_id
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to create session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")


@app.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session(session_id: str):
    """Get session status and information."""
    try:
        session_status = await app_state["session_manager"].get_session_status(session_id)
        
        if not session_status:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionStatusResponse(
            session_id=session_id,
            status=session_status.get("status", "unknown"),
            created_at=session_status.get("created_at", ""),
            last_activity=session_status.get("last_activity", ""),
            domain=session_status.get("domain", "banking"),
            customer_data=session_status.get("customer_data")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Session retrieval failed: {str(e)}")


@app.post("/sessions/{session_id}/process", response_model=CallProcessingResponse)
async def process_call(session_id: str, request: CallProcessingRequest):
    """Process voice call through the complete pipeline."""
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"Processing call for session: {session_id}")
        
        # 1. STT - Convert audio to text
        if request.audio_data:
            # Call STT microservice
            transcript = await _call_stt_service(request.audio_data)
        elif request.message:
            transcript = request.message
        else:
            raise ValueError("Either audio_data or message must be provided")
        
        # 2. LLM - Generate response with caching and adapters
        session_status = await app_state["session_manager"].get_session_status(session_id)
        domain = session_status.get("domain", "banking") if session_status else "banking"
        
        response_text = await app_state["llm"].generate_response(
            user_input=transcript,
            domain_context=domain
        )
        
        # 3. TTS - Convert response to audio
        response_audio = await _call_tts_service(response_text)
        
        # 4. Analytics logging
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        await app_state["analytics"].log_event(
            event_type="call_processed",
            session_id=session_id,
            metadata={
                "transcript": transcript,
                "response": response_text,
                "processing_time": processing_time,
                "domain": domain
            }
        )
        
        # Get performance metrics
        llm_metrics = app_state["llm"].get_performance_metrics()
        
        return CallProcessingResponse(
            session_id=session_id,
            response_audio=response_audio,
            transcript=transcript,
            response_text=response_text,
            processing_time=processing_time,
            cache_hit=llm_metrics["cache_hits"] > 0,
            adapter_used=domain + "-lora" if llm_metrics["adapter_enhanced_calls"] > 0 else None
        )
        
    except Exception as e:
        logger.error(f"Failed to process call for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Call processing failed: {str(e)}")


@app.websocket("/sessions/{session_id}/stream")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time call streaming."""
    await websocket.accept()
    app_state["active_connections"][session_id] = websocket
    
    try:
        logger.info(f"WebSocket connection established for session: {session_id}")
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            if data.get("type") == "audio_chunk":
                # Process audio chunk
                response = await _process_audio_chunk(session_id, data)
                await websocket.send_json(response)
                
            elif data.get("type") == "text_message":
                # Process text message
                response = await _process_text_message(session_id, data)
                await websocket.send_json(response)
                
            elif data.get("type") == "ping":
                # Heartbeat response
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        if session_id in app_state["active_connections"]:
            del app_state["active_connections"][session_id]


# Helper functions
async def _call_stt_service(audio_data: str) -> str:
    """Call STT microservice."""
    # In real implementation, would make HTTP request to STT service
    # For now, return mock transcript
    return "What is my account balance?"


async def _call_tts_service(text: str) -> str:
    """Call TTS microservice."""
    # In real implementation, would make HTTP request to TTS service
    # For now, return mock audio data
    return "base64_encoded_audio_data"


async def _process_audio_chunk(session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Process incoming audio chunk."""
    # Mock implementation
    return {
        "type": "audio_response",
        "session_id": session_id,
        "response_audio": "base64_audio_response",
        "timestamp": datetime.utcnow().isoformat()
    }


async def _process_text_message(session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Process incoming text message."""
    message = data.get("message", "")
    
    # Get session domain
    session_status = await app_state["session_manager"].get_session_status(session_id)
    domain = session_status.get("domain", "banking") if session_status else "banking"
    
    # Generate LLM response
    response_text = await app_state["llm"].generate_response(
        user_input=message,
        domain_context=domain
    )
    
    return {
        "type": "text_response",
        "session_id": session_id,
        "original_message": message,
        "response_text": response_text,
        "timestamp": datetime.utcnow().isoformat()
    }


def main():
    """Main entry point for orchestrator-core microservice."""
    logger.info(f"Starting Orchestrator Core on {HOST}:{PORT}")
    
    # Configuration
    config = {
        "host": HOST,
        "port": PORT,
        "log_level": LOG_LEVEL.lower(),
        "access_log": True,
        "reload": ENVIRONMENT == "development"
    }
    
    # Start the server
    uvicorn.run("voicebot_orchestrator.microservices.orchestrator_core:app", **config)


if __name__ == "__main__":
    main()
