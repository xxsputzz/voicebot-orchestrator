"""
LLM Microservice

Dedicated service for language model operations with semantic caching and LoRA adapters.
Handles text generation, domain adaptation, and performance optimization.
"""

import asyncio
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

# Mock FastAPI for restricted environment
class FastAPI:
    def __init__(self, **kwargs): pass
    def post(self, path): return lambda f: f
    def get(self, path): return lambda f: f

class HTTPException(Exception):
    def __init__(self, status_code, detail): pass

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

from voicebot_orchestrator.enhanced_llm import create_enhanced_llm

# Configuration
PORT = int(os.getenv("LLM_SERVICE_PORT", "8002"))
HOST = os.getenv("LLM_SERVICE_HOST", "0.0.0.0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)


# Request/Response models
class GenerationRequest(BaseModel):
    text: str
    session_id: str
    domain: str = "banking"
    conversation_history: Optional[List[Dict[str, str]]] = None
    max_tokens: int = 512
    temperature: float = 0.7
    use_cache: bool = True


class GenerationResponse(BaseModel):
    response_text: str
    session_id: str
    processing_time: float
    cache_hit: bool
    adapter_used: Optional[str]
    token_count: int
    confidence: float


# App state
app_state = {
    "llm": None,
    "stats": {
        "requests_processed": 0,
        "cache_hits": 0,
        "adapter_uses": 0,
        "avg_processing_time": 0.0,
        "errors": 0
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Service lifespan management."""
    logger.info("Starting LLM Service...")
    
    # Initialize enhanced LLM
    app_state["llm"] = create_enhanced_llm(
        enable_cache=True,
        enable_adapters=True
    )
    
    # Setup banking domain
    await app_state["llm"].setup_banking_domain()
    
    logger.info("LLM Service initialized")
    yield
    logger.info("LLM Service shutdown")


app = FastAPI(title="LLM Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "llm-service",
        "version": "1.0.0",
        "model_loaded": app_state["llm"] is not None,
        "cache_enabled": app_state["llm"].enable_cache if app_state["llm"] else False,
        "adapters_enabled": app_state["llm"].enable_adapters if app_state["llm"] else False,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text response using LLM."""
    start_time = datetime.utcnow()
    
    try:
        # Get initial cache/adapter stats
        initial_metrics = app_state["llm"].get_performance_metrics()
        initial_cache_hits = initial_metrics["cache_hits"]
        initial_adapter_calls = initial_metrics["adapter_enhanced_calls"]
        
        # Generate response
        response_text = await app_state["llm"].generate_response(
            user_input=request.text,
            conversation_history=request.conversation_history,
            use_cache=request.use_cache,
            domain_context=request.domain
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Check if cache was hit
        final_metrics = app_state["llm"].get_performance_metrics()
        cache_hit = final_metrics["cache_hits"] > initial_cache_hits
        adapter_used = None
        
        if final_metrics["adapter_enhanced_calls"] > initial_adapter_calls:
            adapter_used = f"{request.domain}-lora"
        
        # Update service stats
        app_state["stats"]["requests_processed"] += 1
        if cache_hit:
            app_state["stats"]["cache_hits"] += 1
        if adapter_used:
            app_state["stats"]["adapter_uses"] += 1
        
        app_state["stats"]["avg_processing_time"] = (
            app_state["stats"]["avg_processing_time"] + processing_time
        ) / 2
        
        return GenerationResponse(
            response_text=response_text,
            session_id=request.session_id,
            processing_time=processing_time,
            cache_hit=cache_hit,
            adapter_used=adapter_used,
            token_count=len(response_text.split()),  # Simple approximation
            confidence=0.9 if adapter_used else 0.8
        )
        
    except Exception as e:
        app_state["stats"]["errors"] += 1
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    if app_state["llm"]:
        llm_metrics = app_state["llm"].get_performance_metrics()
        cache_stats = app_state["llm"].get_cache_stats()
        adapter_status = app_state["llm"].get_adapter_status()
        
        return {
            "service_stats": app_state["stats"],
            "llm_metrics": llm_metrics,
            "cache_stats": cache_stats,
            "adapter_status": adapter_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        return app_state["stats"]


@app.post("/cache/clear")
async def clear_cache():
    """Clear the semantic cache."""
    try:
        success = app_state["llm"].clear_cache()
        return {
            "status": "success" if success else "failed",
            "message": "Cache cleared" if success else "Cache clear failed",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/adapters")
async def list_adapters():
    """List available LoRA adapters."""
    try:
        if app_state["llm"] and app_state["llm"].adapter_manager:
            status = app_state["llm"].get_adapter_status()
            return status
        else:
            return {"available_adapters": [], "loaded_adapters": [], "active_adapter": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point."""
    uvicorn.run("voicebot_orchestrator.microservices.llm_service:app", 
                host=HOST, port=PORT, log_level=LOG_LEVEL.lower())


if __name__ == "__main__":
    main()
