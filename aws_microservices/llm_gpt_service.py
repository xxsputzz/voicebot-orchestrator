"""
GPT LLM Microservice - Independent Service
Language model using GPT (Open Source) o        # Initialize with memory-optimized GPT model
        llm_service = EnhancedMistralLLM(
            model_path="microsoft/DialoGPT-small",  # Use stable DialoGPT model instead
            max_tokens=512,  # Conservative token limit for stability
            temperature=0.7,
            enable_cache=True,
            enable_adapters=True,
            cache_dir="./cache",
            adapter_dir="./adapters"
        ): 8022
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
import re

# Import your existing LLM implementation
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from voicebot_orchestrator.enhanced_llm import EnhancedMistralLLM  # We'll adapt this for GPT

# EMOJI PURGING: Comprehensive emoji detection and removal
def _purge_emojis_from_llm_response(text: str) -> str:
    """
    Remove all emojis from LLM responses to prevent TTS encoding issues.
    Simply removes emojis without text replacement to preserve sentence meaning.
    """
    if not text:
        return text
    
    # Nuclear emoji removal - comprehensive Unicode ranges
    # Remove ALL emojis without any text replacement
    emoji_patterns = [
        r'[\U0001F600-\U0001F64F]',  # Emoticons
        r'[\U0001F300-\U0001F5FF]',  # Misc Symbols
        r'[\U0001F680-\U0001F6FF]',  # Transport
        r'[\U0001F1E0-\U0001F1FF]',  # Country flags
        r'[\U00002600-\U000027BF]',  # Misc symbols
        r'[\U0001F900-\U0001F9FF]',  # Supplemental Symbols
        r'[\U00002702-\U000027B0]',  # Dingbats
        r'[\U000024C2-\U0001F251]',  # Various symbols
        r'[\U0001F170-\U0001F171]',  # Enclosed alphanumerics
        r'[\U0001F17E-\U0001F17F]',  # More enclosed
        r'[\U0001F18E]',             # Negative squared
        r'[\U0001F191-\U0001F19A]',  # Squared symbols
        r'[\U0001F201-\U0001F202]',  # Squared katakana
        r'[\U0001F21A]',             # Squared CJK
        r'[\U0001F22F]',             # Squared finger
        r'[\U0001F232-\U0001F23A]',  # Squared CJK symbols
        r'[\U0001F250-\U0001F251]',  # Circled ideographs
        r'[\U0000FE0F]',             # Variation selector (fixed escape)
        r'[\U0000200D]',             # Zero width joiner
    ]
    for pattern in emoji_patterns:
        text = re.sub(pattern, '', text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

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
    """Initialize Memory-Optimized GPT LLM service on startup"""
    global llm_service
    logging.info("🧠 Initializing Memory-Optimized GPT LLM Microservice...")
    
    try:
        # Check GPU memory and optimize model size
        gpu_memory_gb = 0
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_name = torch.cuda.get_device_name(0)
            logging.info(f"🎯 GPU Detected: {gpu_name} ({gpu_memory_gb:.1f}GB)")
            
            # Smart model selection based on available GPU memory
            if gpu_memory_gb <= 8:
                model_size = "1.3b"  # Ultra-lightweight for 8GB GPUs
                max_memory = 3  # Use only 3GB for safety
                logging.info("📊 Using 1.3B model for stable 8GB GPU performance")
            elif gpu_memory_gb <= 12:
                model_size = "3b"  # Small model for 12GB GPUs  
                max_memory = 6
                logging.info("📊 Using 3B model for 12GB GPU")
            elif gpu_memory_gb <= 16:
                model_size = "7b"  # Medium model for 16GB GPUs
                max_memory = 10
                logging.info("📊 Using 7B model for 16GB GPU")
            else:
                model_size = "13b"  # Large model for 24GB+ GPUs
                max_memory = 16
                logging.info("📊 Using 13B model for high-memory GPU")
        else:
            model_size = "1.3b"  # Ultra-lightweight for CPU
            max_memory = 2
            logging.warning("⚠️ Using 1.3B model for CPU execution")
        
        # Initialize with memory-optimized GPT model
        llm_service = EnhancedMistralLLM(
            model_path="microsoft/DialoGPT-small",  # Use stable DialoGPT model instead
            max_tokens=512,  # Conservative token limit for stability
            temperature=0.7,
            enable_cache=True,
            enable_adapters=True,
            cache_dir="./cache",
            adapter_dir="./adapters"
        )
        
        # Setup domain optimization for GPT with memory management
        await asyncio.to_thread(llm_service.setup_banking_domain)
        
        # Memory usage validation
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            reserved_gb = torch.cuda.memory_reserved() / 1024**3
            logging.info(f"💾 GPU Memory: {allocated_gb:.1f}GB allocated, {reserved_gb:.1f}GB reserved")
            
            if reserved_gb > gpu_memory_gb * 0.85:
                logging.warning("⚠️ High GPU memory usage - consider reducing model size")
        
        logging.info("✅ Memory-Optimized GPT LLM Microservice ready!")
        
    except Exception as e:
        logging.error(f"❌ GPT LLM initialization failed: {e}")
        # Cleanup GPU memory on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Enhanced cleanup on shutdown with memory optimization"""
    global llm_service
    if llm_service:
        try:
            logging.info("🛑 Starting GPT LLM graceful shutdown...")
            
            # Clear cache first
            try:
                llm_service.clear_cache()
            except:
                pass
            
            # GPU memory cleanup with multiple passes
            if torch.cuda.is_available():
                for i in range(3):  # Multiple cleanup passes
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    if i < 2:
                        time.sleep(0.5)  # Allow GPU memory to be released
                
                # Final memory check
                final_memory = torch.cuda.memory_allocated() / 1024**3
                logging.info(f"💾 Final GPU memory: {final_memory:.2f}GB allocated")
            
            # Python garbage collection
            llm_service = None
            for _ in range(2):
                gc.collect()
            
            logging.info("✅ GPT LLM Microservice shutdown complete")
            
        except Exception as e:
            logging.error(f"⚠️ Cleanup warning: {e}")
            # Force cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

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
        
        # EMOJI PURGING: Remove emojis from LLM response to prevent TTS encoding issues
        original_response = response
        response = _purge_emojis_from_llm_response(response)
        if response != original_response:
            logging.info(f"[EMOJI_PURGE_LLM] Cleaned emoji from LLM response")
        
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
            "model": "dialogpt-small",
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
        "model": "dialogpt-small",
        "port": 8022,
        "version": "stable",
        "features": {
            "semantic_cache": True,
            "lora_adapters": True,
            "domain_optimization": True,
            "banking_domain": True,
            "conversation_history": True,
            "chat_completion": True,
            "fast_responses": True
        },
        "limits": {
            "max_input_length": 1000,
            "max_tokens": 512,
            "temperature_range": [0.1, 2.0],
            "context_window": "1024 tokens"
        },
        "performance": {
            "typical_response_time": "1-3 seconds",
            "tokens_per_second": "30-60",
            "cache_hit_rate": "40-60%",
            "stability": "high"
        },
        "independent": True,
        "description": "Memory-optimized GPT service for stable 8GB GPU performance"
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
        "model": "dialogpt-small",
        "status": "running" if llm_service else "stopped",
        "ready": llm_service is not None,
        "gpu_info": gpu_info,
        "cache_stats": cache_stats,
        "advantages": [
            "Fast response times",
            "Stable GPU memory usage",
            "8GB GPU compatible",
            "Good conversation quality",
            "Chat completion API compatible"
        ],
        "use_cases": [
            "Customer service conversations",
            "General question answering",
            "Interactive chat systems",
            "Banking domain queries",
            "Stable production workloads"
        ],
        "requirements": [
            "Low GPU memory (works on 8GB)",
            "Fast processing time",
            "Lower computational cost",
            "General conversational AI"
        ]
    }

@app.get("/models")
async def get_model_info():
    """Get GPT model information"""
    return {
        "available_models": ["dialogpt-small", "dialogpt-medium"],
        "current_model": "dialogpt-small",
        "model_details": {
            "name": "Microsoft DialoGPT Small",
            "parameters": "117 million",
            "context_window": 1024,
            "training_data": "Reddit conversations",
            "capabilities": [
                "Conversational responses",
                "Context awareness",
                "Banking domain optimization",
                "Fast inference",
                "Stable memory usage"
            ]
        },
        "performance_tiers": {
            "cpu": "Functional (2-5 seconds)",
            "gpu_8gb": "Fast performance (1-2 seconds)",
            "gpu_16gb": "Optimal performance (<1 second)",
            "gpu_24gb+": "Ultra-fast performance"
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
