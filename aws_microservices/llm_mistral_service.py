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
import re
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from voicebot_orchestrator.enhanced_llm import EnhancedMistralLLM

# Import prompt loader for dynamic prompt injection
from prompt_loader import prompt_loader

# Import conversation manager for context tracking
from conversation_manager import ConversationManager

# EMOJI PURGING: Comprehensive emoji detection and removal (matching GPT service)
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

# Global conversation manager for context tracking
conversation_manager = ConversationManager()

class GenerateRequest(BaseModel):
    text: str
    use_cache: Optional[bool] = True
    domain_context: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    call_type: Optional[str] = None  # "inbound", "outbound", or None
    conversation_id: Optional[str] = None  # For conversation tracking
    customer_phone: Optional[str] = None   # For finding existing conversations

class GenerateResponse(BaseModel):
    response: str
    processing_time_seconds: float
    model_used: str
    cache_hit: bool
    tokens_generated: int
    conversation_id: Optional[str] = None  # Track conversation for client-side management
    conversation_id: Optional[str] = None  # Return conversation ID for tracking

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
        
        # Handle conversation context
        conversation_context = None
        conversation_id = request.conversation_id
        
        # Try to find existing conversation for customer
        if not conversation_id and request.customer_phone:
            conversation_id = conversation_manager.find_active_conversation(request.customer_phone)
        
        # Get or create conversation context
        if conversation_id:
            conversation_context = conversation_manager.get_conversation_context(conversation_id)
            if not conversation_context:
                # Conversation ID provided but not found, start new one
                conversation_id = conversation_manager.start_conversation(
                    request.customer_phone, 
                    request.call_type or "general"
                )
                conversation_context = conversation_manager.get_conversation_context(conversation_id)
        else:
            # Start new conversation
            conversation_id = conversation_manager.start_conversation(
                request.customer_phone, 
                request.call_type or "general"
            )
            conversation_context = conversation_manager.get_conversation_context(conversation_id)
        
        # Use conversation history from database if not provided in request
        conversation_history = request.conversation_history
        if not conversation_history and conversation_context:
            # Convert database format to LLM format
            db_history = conversation_context["recent_history"]
            conversation_history = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in db_history
            ]
        
        # Load system prompts with conversation awareness
        system_prompt = prompt_loader.get_system_prompt(call_type=request.call_type)
        
        # Add conversation context to system prompt
        if conversation_context and not conversation_context["is_first_interaction"]:
            context_note = f"""
CONVERSATION CONTEXT:
- This is a CONTINUING conversation (message #{conversation_context['message_count'] + 1})
- DO NOT introduce yourself again as Alex
- DO NOT repeat company name unless specifically relevant
- Continue naturally from the previous interaction
- Current conversation state: {conversation_context['conversation_state']}
"""
            system_prompt = system_prompt + context_note
        
        # Combine system prompt with domain context if provided
        enhanced_context = request.domain_context or ""
        if system_prompt:
            enhanced_context = system_prompt + "\n\n" + enhanced_context if enhanced_context else system_prompt
            call_type_info = f" (call_type: {request.call_type})" if request.call_type else ""
            logging.info(f"[PROMPT_INJECTION] Added system prompt ({len(system_prompt)} chars) to Mistral LLM{call_type_info}")
            
        # Add user message to conversation history
        conversation_manager.add_message(conversation_id, "user", request.text)
        
        # Generate response using Mistral with enhanced context
        response = await llm_service.generate_response(
            user_input=request.text,
            conversation_history=conversation_history,
            use_cache=request.use_cache,
            domain_context=enhanced_context
        )
        
        # EMOJI PURGING: Remove emojis from LLM response to prevent TTS encoding issues
        original_response = response
        response = _purge_emojis_from_llm_response(response)
        if response != original_response:
            logging.info(f"[EMOJI_PURGE_LLM] Cleaned emoji from Mistral LLM response")
        
        # Add assistant response to conversation history
        conversation_manager.add_message(conversation_id, "assistant", response)
        
        processing_time = time.time() - start_time
        
        # Get cache statistics
        cache_stats = llm_service.get_cache_stats()
        cache_hit = cache_stats and cache_stats.get("last_query_was_cache_hit", False)
        
        return GenerateResponse(
            response=response,
            processing_time_seconds=round(processing_time, 3),
            model_used="mistral",
            cache_hit=cache_hit,
            tokens_generated=len(response.split()),  # Approximate token count
            conversation_id=conversation_id  # Return conversation ID for tracking
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

# PROMPT MANAGEMENT ENDPOINTS
@app.get("/prompts")
async def get_prompts_info():
    """Get information about loaded prompts"""
    try:
        available_prompts = prompt_loader.get_available_prompts()
        system_prompt = prompt_loader.get_system_prompt()
        
        return {
            "model": "mistral",
            "prompts_directory": str(prompt_loader.get_prompts_directory()),
            "available_prompts": available_prompts,
            "total_prompts": len(available_prompts),
            "system_prompt_length": len(system_prompt),
            "system_prompt_preview": system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prompts info: {str(e)}")

@app.post("/prompts/reload")
async def reload_prompts():
    """Reload prompts from the docs/prompts folder"""
    try:
        prompts = prompt_loader.reload_prompts()
        return {
            "message": "Prompts reloaded successfully",
            "total_prompts": len(prompts),
            "loaded_prompts": list(prompts.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload prompts: {str(e)}")

@app.get("/prompts/{prompt_name}")
async def get_specific_prompt(prompt_name: str):
    """Get content of a specific prompt file"""
    try:
        prompts = prompt_loader.load_all_prompts()
        if prompt_name not in prompts:
            raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")
        
        return {
            "prompt_name": prompt_name,
            "content": prompts[prompt_name],
            "length": len(prompts[prompt_name])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prompt: {str(e)}")

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
