"""
Sprint 5: Enhanced LLM with Real Ollama AI Integration

Enhanced Mistral LLM that integrates real Ollama AI generation instead of pattern matching,
while maintaining semantic caching and LoRA adapter support for improved performance.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List, Optional, Any
from .ollama_enhanced_llm import OllamaEnhancedLLM, create_ollama_enhanced_llm


class EnhancedMistralLLM:
    """
    Enhanced Mistral LLM with real Ollama AI generation.
    
    This class now uses OllamaEnhancedLLM internally for real AI generation
    while maintaining backward compatibility with the existing interface.
    """
    
    def __init__(
        self,
        model_path: str = "mistral:latest",
        max_tokens: int = 512,
        temperature: float = 0.7,
        enable_cache: bool = True,
        enable_adapters: bool = True,
        cache_dir: str = "./cache",
        adapter_dir: str = "./adapters"
    ):
        """
        Initialize enhanced Mistral LLM with Ollama integration.
        
        Args:
            model_path: Ollama model name (e.g., "mistral:latest")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            enable_cache: Whether to enable semantic caching
            enable_adapters: Whether to enable LoRA adapters
            cache_dir: Directory for semantic cache
            adapter_dir: Directory for LoRA adapters
        """
        # Create the underlying Ollama-enhanced LLM
        self._ollama_llm = create_ollama_enhanced_llm(
            model_name=model_path,
            enable_cache=enable_cache,
            enable_adapters=enable_adapters,
            fallback_enabled=True
        )
        
        # Store configuration for compatibility
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_cache = enable_cache
        self.enable_adapters = enable_adapters
        
        self.logger = logging.getLogger(__name__)
        
        # Expose underlying properties for compatibility
        self.cache = self._ollama_llm.cache
        self.adapter_manager = self._ollama_llm.adapter_manager
    
    async def generate_response(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_cache: bool = True,
        domain_context: Optional[str] = None
    ) -> str:
        """
        Generate response with real AI and caching support.
        
        Args:
            user_input: User's input text
            conversation_history: Previous conversation context
            use_cache: Whether to use semantic cache
            domain_context: Domain context for adapter selection
            
        Returns:
            Generated response text
        """
        # Default to banking domain if not specified
        if not domain_context:
            domain_context = "banking"
        
        return await self._ollama_llm.generate_response(
            user_input=user_input,
            conversation_history=conversation_history,
            use_cache=use_cache,
            domain_context=domain_context,
            call_type="inbound"
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the enhanced LLM."""
        return self._ollama_llm.get_performance_metrics()
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get semantic cache statistics."""
        return self._ollama_llm.get_cache_stats()
    
    def get_adapter_status(self) -> Optional[Dict[str, Any]]:
        """Get LoRA adapter status."""
        return self._ollama_llm.get_adapter_status()
    
    def clear_cache(self) -> bool:
        """Clear semantic cache."""
        return self._ollama_llm.clear_cache()
    
    def setup_banking_domain(self) -> bool:
        """Set up banking domain optimization."""
        # This is handled automatically by the Ollama integration
        return True


# Factory function for easy setup
def create_enhanced_llm(
    model_path: str = "mistral:latest",
    enable_cache: bool = True,
    enable_adapters: bool = True
) -> EnhancedMistralLLM:
    """
    Create enhanced Mistral LLM with real Ollama AI integration.
    
    Args:
        model_path: Ollama model name (e.g., "mistral:latest")
        enable_cache: Enable semantic caching
        enable_adapters: Enable LoRA adapters
        
    Returns:
        Configured EnhancedMistralLLM instance with real AI
    """
    return EnhancedMistralLLM(
        model_path=model_path,
        max_tokens=512,
        temperature=0.7,
        enable_cache=enable_cache,
        enable_adapters=enable_adapters
    )

# Alias for backward compatibility
EnhancedLLM = EnhancedMistralLLM
