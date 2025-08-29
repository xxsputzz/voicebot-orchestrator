"""
LLM Microservice for AWS Deployment
Runs on GPU-optimized instances (g4dn.xlarge)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import logging
import uvicorn
import torch
import gc
from typing import Dict, Any, List, Optional
import time

# Import your existing LLM implementation
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from voicebot_orchestrator.enhanced_llm import EnhancedMistralLLM

app = FastAPI(title="LLM Microservice", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LLM instance
llm_service = None
current_model_type = None

class GenerateRequest(BaseModel):
    text: str
    model_type: Optional[str] = "mistral"
    use_cache: Optional[bool] = True
    domain_context: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

class GenerateResponse(BaseModel):
    response: str
    processing_time_seconds: float
    model_used: str
    cache_hit: bool
    tokens_generated: int

@app.on_event("startup")
async def startup_event():
    """Initialize LLM service on startup"""
    global llm_service, current_model_type
    logging.info("🧠 Initializing LLM Microservice...")
    
    try:
        # Initialize with default Mistral model
        llm_service = EnhancedMistralLLM(
            model_path="mistral:latest",
            temperature=0.7,
            enable_cache=True,
            enable_adapters=True
        )
        current_model_type = "mistral"
        
        # Setup banking domain optimization
        await asyncio.to_thread(llm_service.setup_banking_domain)
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"✅ LLM Service ready on GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logging.warning("⚠️ LLM running on CPU (slower performance)")
        
        logging.info("✅ LLM Microservice ready!")
        
    except Exception as e:
        logging.error(f"❌ LLM initialization failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global llm_service
    if llm_service:
        try:
            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            llm_service = None
            logging.info("🛑 LLM Microservice shutdown complete")
        except Exception as e:
            logging.error(f"⚠️ Cleanup warning: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_memory = None
    
    if gpu_available:
        gpu_memory = {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    return {
        "status": "healthy",
        "service": "llm",
        "timestamp": time.time(),
        "ready": llm_service is not None,
        "model_type": current_model_type,
        "gpu_available": gpu_available,
        "gpu_memory": gpu_memory
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest) -> GenerateResponse:
    """
    Generate response using LLM
    
    Args:
        request: Generation request with text and parameters
        
    Returns:
        Generated response with metadata
    """
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service not ready")
    
    start_time = time.time()
    
    try:
        # Validate input
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        if len(request.text) > 2000:
            raise HTTPException(status_code=400, detail="Text input too long (max 2000 characters)")
        
        # Check if model switch is needed
        if request.model_type != current_model_type:
            await switch_model(request.model_type)
        
        # Generate response
        response = await llm_service.generate_response(
            user_input=request.text,
            conversation_history=request.conversation_history,
            use_cache=request.use_cache,
            domain_context=request.domain_context
        )
        
        processing_time = time.time() - start_time
        
        # Get cache statistics
        cache_stats = llm_service.get_cache_stats()
        cache_hit = cache_stats and cache_stats.get("last_query_was_cache_hit", False)
        
        return GenerateResponse(
            response=response,
            processing_time_seconds=round(processing_time, 3),
            model_used=current_model_type,
            cache_hit=cache_hit,
            tokens_generated=len(response.split())  # Approximate token count
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"❌ Generation failed after {processing_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

async def switch_model(model_type: str):
    """Switch LLM model type"""
    global llm_service, current_model_type
    
    if model_type == current_model_type:
        return
    
    logging.info(f"🔄 Switching LLM model: {current_model_type} → {model_type}")
    
    try:
        # Cleanup current model
        if llm_service:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del llm_service
            gc.collect()
        
        # Initialize new model
        if model_type == "gpt-oss":
            model_path = "gpt-oss:20b"
        else:
            model_path = "mistral:latest"
        
        llm_service = EnhancedMistralLLM(
            model_path=model_path,
            temperature=0.7,
            enable_cache=True,
            enable_adapters=True
        )
        
        current_model_type = model_type
        logging.info(f"✅ Switched to {model_type}")
        
    except Exception as e:
        logging.error(f"❌ Model switch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model switch failed: {str(e)}")

@app.get("/performance")
async def get_performance_metrics():
    """Get LLM performance metrics"""
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service not ready")
    
    try:
        metrics = llm_service.get_performance_metrics()
        cache_stats = llm_service.get_cache_stats()
        adapter_status = llm_service.get_adapter_status()
        
        return {
            "performance": metrics,
            "cache": cache_stats,
            "adapters": adapter_status,
            "gpu_memory": {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.post("/clear_cache")
async def clear_cache():
    """Clear semantic cache"""
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service not ready")
    
    try:
        success = llm_service.clear_cache()
        return {"cache_cleared": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.get("/info")
async def service_info():
    """Get LLM service information"""
    return {
        "service": "llm",
        "current_model": current_model_type,
        "available_models": ["mistral", "gpt-oss"],
        "features": {
            "semantic_cache": True,
            "lora_adapters": True,
            "domain_optimization": True,
            "banking_domain": True
        },
        "limits": {
            "max_input_length": 2000,
            "max_tokens": 1024,
            "temperature_range": [0.1, 2.0]
        }
    }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the service
    uvicorn.run(
        "llm_service:app",
        host="0.0.0.0",
        port=8002,
        workers=1,  # Single worker for GPU models
        log_level="info"
    )
