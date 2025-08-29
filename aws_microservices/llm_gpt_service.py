"""
GPT LLM Microservice - Independent Service
Language model using GPT (Open Source) only
Port: 8022
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
from voicebot_orchestrator.enhanced_llm import EnhancedMistralLLM  # We'll adapt this for GPT

app = FastAPI(title="GPT LLM Microservice", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LLM instance - GPT only
llm_service = None

class GenerateRequest(BaseModel):
    text: str
    use_cache: Optional[bool] = True
    domain_context: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    frequency_penalty: Optional[float] = 0.0

class GenerateResponse(BaseModel):
    response: str
    processing_time_seconds: float
    model_used: str
    cache_hit: bool
    tokens_generated: int

@app.on_event("startup")
async def startup_event():
    """Initialize GPT LLM service on startup"""
    global llm_service
    logging.info("🧠 Initializing GPT LLM Microservice...")
    
    try:
        # Initialize with GPT model (using adapted enhanced LLM)
        llm_service = EnhancedMistralLLM(
            model_path="gpt-oss:20b",  # GPT open-source model
            temperature=0.7,
            enable_cache=True,
            enable_adapters=True
        )
        
        # Setup domain optimization for GPT
        await asyncio.to_thread(llm_service.setup_banking_domain)
        
        # Check GPU availability (highly recommended for GPT)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"✅ GPT LLM Service ready on GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logging.warning("⚠️ GPT LLM running on CPU (significantly slower performance)")
        
        logging.info("✅ GPT LLM Microservice ready!")
        
    except Exception as e:
        logging.error(f"❌ GPT LLM initialization failed: {e}")
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
            logging.info("🛑 GPT LLM Microservice shutdown complete")
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
        "service": "llm_gpt",
        "model": "gpt-oss",
        "timestamp": time.time(),
        "ready": llm_service is not None,
        "gpu_available": gpu_available,
        "gpu_memory": gpu_memory,
        "features": ["semantic_cache", "lora_adapters", "advanced_reasoning"]
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest) -> GenerateResponse:
    """
    Generate response using GPT LLM
    """
    if not llm_service:
        raise HTTPException(status_code=503, detail="GPT LLM service not ready")
    
    start_time = time.time()
    
    try:
        # Validate input
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        if len(request.text) > 3000:  # GPT can handle longer context
            raise HTTPException(status_code=400, detail="Text input too long (max 3000 characters)")
        
        # Generate response using GPT
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
            model_used="gpt-oss",
            cache_hit=cache_hit,
            tokens_generated=len(response.split())  # Approximate token count
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"❌ GPT generation failed after {processing_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"GPT generation failed: {str(e)}")

@app.post("/chat")
async def chat_completion(request: GenerateRequest):
    """
    Chat completion endpoint (GPT-style interface)
    """
    if not llm_service:
        raise HTTPException(status_code=503, detail="GPT LLM service not ready")
    
    start_time = time.time()
    
    try:
        # Convert conversation history to GPT format
        messages = []
        if request.conversation_history:
            for turn in request.conversation_history:
                if "user" in turn:
                    messages.append({"role": "user", "content": turn["user"]})
                if "assistant" in turn:
                    messages.append({"role": "assistant", "content": turn["assistant"]})
        
        # Add current message
        messages.append({"role": "user", "content": request.text})
        
        # Generate response
        response = await llm_service.generate_response(
            user_input=request.text,
            conversation_history=request.conversation_history,
            use_cache=request.use_cache,
            domain_context=request.domain_context
        )
        
        processing_time = time.time() - start_time
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-oss-20b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(request.text.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(request.text.split()) + len(response.split())
            },
            "processing_time_seconds": round(processing_time, 3)
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"❌ GPT chat completion failed after {processing_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"GPT chat completion failed: {str(e)}")

@app.get("/performance")
async def get_performance_metrics():
    """Get GPT LLM performance metrics"""
    if not llm_service:
        raise HTTPException(status_code=503, detail="GPT LLM service not ready")
    
    try:
        metrics = llm_service.get_performance_metrics()
        cache_stats = llm_service.get_cache_stats()
        adapter_status = llm_service.get_adapter_status()
        
        return {
            "model": "gpt-oss",
            "performance": metrics,
            "cache": cache_stats,
            "adapters": adapter_status,
            "gpu_memory": {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get GPT metrics: {str(e)}")

@app.post("/clear_cache")
async def clear_cache():
    """Clear GPT semantic cache"""
    if not llm_service:
        raise HTTPException(status_code=503, detail="GPT LLM service not ready")
    
    try:
        success = llm_service.clear_cache()
        return {"cache_cleared": success, "model": "gpt-oss"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear GPT cache: {str(e)}")

@app.get("/info")
async def service_info():
    """Get GPT LLM service information"""
    return {
        "service": "llm_gpt",
        "model": "gpt-oss-20b",
        "port": 8022,
        "version": "open-source",
        "features": {
            "semantic_cache": True,
            "lora_adapters": True,
            "domain_optimization": True,
            "banking_domain": True,
            "conversation_history": True,
            "chat_completion": True,
            "advanced_reasoning": True
        },
        "limits": {
            "max_input_length": 3000,
            "max_tokens": 1024,
            "temperature_range": [0.1, 2.0],
            "context_window": "4096 tokens"
        },
        "performance": {
            "typical_response_time": "5-15 seconds",
            "tokens_per_second": "10-30",
            "cache_hit_rate": "20-40%",
            "reasoning_quality": "high"
        },
        "independent": True,
        "description": "Dedicated GPT (Open Source) LLM service for advanced conversational AI"
    }

@app.get("/status")
async def get_status():
    """Get detailed GPT service status"""
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
        "service_name": "GPT LLM",
        "model": "gpt-oss-20b",
        "status": "running" if llm_service else "stopped",
        "ready": llm_service is not None,
        "gpu_info": gpu_info,
        "cache_stats": cache_stats,
        "advantages": [
            "Advanced reasoning capabilities",
            "Superior language understanding",
            "Complex problem solving",
            "High-quality responses",
            "Chat completion API compatible"
        ],
        "use_cases": [
            "Complex customer queries",
            "Advanced problem solving",
            "Creative content generation",
            "Technical documentation",
            "Multi-step reasoning tasks"
        ],
        "requirements": [
            "High GPU memory (recommended 16GB+)",
            "Longer processing time",
            "Higher computational cost",
            "Advanced use cases"
        ]
    }

@app.get("/models")
async def get_model_info():
    """Get GPT model information"""
    return {
        "available_models": ["gpt-oss-20b"],
        "current_model": "gpt-oss-20b",
        "model_details": {
            "name": "GPT Open Source 20B",
            "parameters": "20 billion",
            "context_window": 4096,
            "training_data": "Diverse internet text",
            "capabilities": [
                "Text generation",
                "Conversation",
                "Code generation",
                "Analysis and reasoning",
                "Creative writing"
            ]
        },
        "performance_tiers": {
            "cpu": "Very slow (not recommended)",
            "gpu_8gb": "Slow but functional",
            "gpu_16gb": "Good performance",
            "gpu_24gb+": "Optimal performance"
        }
    }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the GPT LLM service
    uvicorn.run(
        "llm_gpt_service:app",
        host="0.0.0.0",
        port=8022,
        workers=1,  # Single worker for GPU models
        log_level="info"
    )
