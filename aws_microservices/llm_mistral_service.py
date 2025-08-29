"""
Mistral LLM Microservice - Independent Service
Language model using Mistral only
Port: 8021
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

app = FastAPI(title="Mistral LLM Microservice", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LLM instance - Mistral only
llm_service = None

class GenerateRequest(BaseModel):
    text: str
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
    """Initialize Mistral LLM service on startup"""
    global llm_service
    logging.info("🧠 Initializing Mistral LLM Microservice...")
    
    try:
        # Initialize with Mistral model only
        llm_service = EnhancedMistralLLM(
            model_path="mistral:latest",
            temperature=0.7,
            enable_cache=True,
            enable_adapters=True
        )
        
        # Setup banking domain optimization
        await asyncio.to_thread(llm_service.setup_banking_domain)
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"✅ Mistral LLM Service ready on GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logging.warning("⚠️ Mistral LLM running on CPU (slower performance)")
        
        logging.info("✅ Mistral LLM Microservice ready!")
        
    except Exception as e:
        logging.error(f"❌ Mistral LLM initialization failed: {e}")
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
            logging.info("🛑 Mistral LLM Microservice shutdown complete")
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
        "service": "llm_mistral",
        "model": "mistral",
        "timestamp": time.time(),
        "ready": llm_service is not None,
        "gpu_available": gpu_available,
        "gpu_memory": gpu_memory,
        "features": ["semantic_cache", "lora_adapters", "banking_domain"]
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest) -> GenerateResponse:
    """
    Generate response using Mistral LLM
    """
    if not llm_service:
        raise HTTPException(status_code=503, detail="Mistral LLM service not ready")
    
    start_time = time.time()
    
    try:
        # Validate input
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        if len(request.text) > 2000:
            raise HTTPException(status_code=400, detail="Text input too long (max 2000 characters)")
        
        # Generate response using Mistral
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
            model_used="mistral",
            cache_hit=cache_hit,
            tokens_generated=len(response.split())  # Approximate token count
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"❌ Mistral generation failed after {processing_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Mistral generation failed: {str(e)}")

@app.get("/performance")
async def get_performance_metrics():
    """Get Mistral LLM performance metrics"""
    if not llm_service:
        raise HTTPException(status_code=503, detail="Mistral LLM service not ready")
    
    try:
        metrics = llm_service.get_performance_metrics()
        cache_stats = llm_service.get_cache_stats()
        adapter_status = llm_service.get_adapter_status()
        
        return {
            "model": "mistral",
            "performance": metrics,
            "cache": cache_stats,
            "adapters": adapter_status,
            "gpu_memory": {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Mistral metrics: {str(e)}")

@app.post("/clear_cache")
async def clear_cache():
    """Clear Mistral semantic cache"""
    if not llm_service:
        raise HTTPException(status_code=503, detail="Mistral LLM service not ready")
    
    try:
        success = llm_service.clear_cache()
        return {"cache_cleared": success, "model": "mistral"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear Mistral cache: {str(e)}")

@app.get("/info")
async def service_info():
    """Get Mistral LLM service information"""
    return {
        "service": "llm_mistral",
        "model": "mistral",
        "port": 8021,
        "version": "latest",
        "features": {
            "semantic_cache": True,
            "lora_adapters": True,
            "domain_optimization": True,
            "banking_domain": True,
            "conversation_history": True
        },
        "limits": {
            "max_input_length": 2000,
            "max_tokens": 1024,
            "temperature_range": [0.1, 2.0]
        },
        "performance": {
            "typical_response_time": "2-5 seconds",
            "tokens_per_second": "20-50",
            "cache_hit_rate": "15-30%"
        },
        "independent": True,
        "description": "Dedicated Mistral LLM service for conversational AI"
    }

@app.get("/status")
async def get_status():
    """Get detailed Mistral service status"""
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3
        }
    
    cache_stats = None
    if llm_service:
        try:
            cache_stats = llm_service.get_cache_stats()
        except:
            pass
    
    return {
        "service_name": "Mistral LLM",
        "model": "mistral",
        "status": "running" if llm_service else "stopped",
        "ready": llm_service is not None,
        "gpu_info": gpu_info,
        "cache_stats": cache_stats,
        "advantages": [
            "Fast response generation",
            "Good conversation quality",
            "Efficient memory usage",
            "Banking domain optimized",
            "LoRA adapter support"
        ],
        "use_cases": [
            "Customer service chat",
            "Banking conversations",
            "General Q&A",
            "Real-time dialogue",
            "Domain-specific tasks"
        ]
    }

@app.get("/adapters")
async def get_adapter_info():
    """Get LoRA adapter information"""
    if not llm_service:
        raise HTTPException(status_code=503, detail="Mistral LLM service not ready")
    
    try:
        adapter_status = llm_service.get_adapter_status()
        return {
            "model": "mistral",
            "adapters": adapter_status,
            "supported_domains": ["banking", "general"],
            "adapter_benefits": [
                "Domain-specific optimization",
                "Improved accuracy",
                "Faster convergence",
                "Task specialization"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get adapter info: {str(e)}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the Mistral LLM service
    uvicorn.run(
        "llm_mistral_service:app",
        host="0.0.0.0",
        port=8021,
        workers=1,  # Single worker for GPU models
        log_level="info"
    )
