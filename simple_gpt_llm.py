"""
Simple High-Quality GPT LLM - Clean Direct Approach

This version uses a straightforward approach with GPT for clean, natural responses
without complex hybrid logic or over-engineering.
"""

import asyncio
import logging
import aiohttp
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# Import prompt system
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from aws_microservices.prompt_loader import PromptLoader

class SimpleGPTLLM:
    """
    Simple, clean GPT LLM implementation for natural banking conversations.
    
    Focuses on quality responses without complex optimization layers.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-oss:20b",
        fallback_model: str = "mistral:latest",
        ollama_host: str = "localhost",
        ollama_port: int = 11434
    ):
        """Initialize simple GPT LLM."""
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.ollama_url = f"http://{ollama_host}:{ollama_port}"
        
        # Initialize prompt system
        self.prompt_loader = PromptLoader()
        
        # Simple metrics
        self.total_requests = 0
        self.successful_responses = 0
        self.gpt_used = 0
        self.fallback_used = 0
        
        self.logger = logging.getLogger(__name__)
    
    async def generate_response(
        self,
        user_input: str,
        user_name: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        call_type: str = "inbound"
    ) -> str:
        """Generate clean, natural response using GPT."""
        self.total_requests += 1
        
        # Build clean prompt
        prompt = self._build_clean_prompt(user_input, user_name, conversation_history, call_type)
        
        # Try GPT first
        try:
            response = await self._call_ollama_clean(prompt, self.model_name)
            if response and len(response.strip()) > 20:  # Basic quality check
                self.gpt_used += 1
                self.successful_responses += 1
                return response.strip()
        except Exception as e:
            self.logger.warning(f"GPT model failed: {e}")
        
        # Fallback to Mistral if GPT fails
        try:
            response = await self._call_ollama_clean(prompt, self.fallback_model)
            if response and len(response.strip()) > 20:
                self.fallback_used += 1
                self.successful_responses += 1
                return response.strip()
        except Exception as e:
            self.logger.error(f"Fallback model failed: {e}")
        
        # Final fallback
        return self._create_simple_fallback(user_input, user_name)
    
    def _build_clean_prompt(
        self,
        user_input: str,
        user_name: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]],
        call_type: str
    ) -> str:
        """Build clean, straightforward prompt."""
        try:
            # Get base system prompt
            system_prompt = self.prompt_loader.get_system_prompt(call_type=call_type)
        except Exception as e:
            self.logger.warning(f"Using fallback system prompt: {e}")
            system_prompt = self._get_simple_system_prompt()
        
        # Add conversation context if available
        context = ""
        if conversation_history and len(conversation_history) > 0:
            context = "\nRecent conversation:\n"
            for exchange in conversation_history[-2:]:  # Last 2 exchanges
                if exchange.get('customer'):
                    context += f"Customer: {exchange['customer']}\n"
                if exchange.get('alex') or exchange.get('assistant'):
                    alex_response = exchange.get('alex') or exchange.get('assistant')
                    context += f"Alex: {alex_response}\n"
        
        # Build simple, effective prompt
        prompt = f"""{system_prompt}

{context}

Customer: {user_input}

Alex:"""
        
        return prompt
    
    async def _call_ollama_clean(self, prompt: str, model: str) -> str:
        """Clean Ollama API call with optimized settings."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 300,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "stop": ["Customer:", "Human:", "\n\nCustomer:", "\n\nHuman:"]
            }
        }
        
        timeout = aiohttp.ClientTimeout(total=60)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "").strip()
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
    
    def _create_simple_fallback(self, user_input: str, user_name: Optional[str]) -> str:
        """Create simple, natural fallback response."""
        name_part = f" {user_name}" if user_name else ""
        
        return f"Hi{name_part}! I'm Alex with Finally Payoff Debt. I heard you say '{user_input[:100]}'. I'm here to help you explore our loan options that could save you money on high-interest debt. What questions can I answer for you?"
    
    def _get_simple_system_prompt(self) -> str:
        """Simple fallback system prompt."""
        return """You are Alex, a friendly banking specialist working for Finally Payoff Debt. 
        You help customers with debt consolidation and personal loans. You're knowledgeable, 
        enthusiastic, and always try to provide helpful financial guidance. Keep responses 
        natural and conversational."""
    
    def get_stats(self) -> Dict:
        """Get simple performance statistics."""
        success_rate = (self.successful_responses / max(self.total_requests, 1)) * 100
        gpt_usage = (self.gpt_used / max(self.total_requests, 1)) * 100
        
        return {
            "total_requests": self.total_requests,
            "successful_responses": self.successful_responses,
            "success_rate": f"{success_rate:.1f}%",
            "gpt_usage": f"{gpt_usage:.1f}%",
            "fallback_usage": f"{((self.fallback_used / max(self.total_requests, 1)) * 100):.1f}%"
        }


def create_simple_gpt_llm(model_name: str = "gpt-oss:20b") -> SimpleGPTLLM:
    """Create simple GPT LLM."""
    return SimpleGPTLLM(model_name=model_name)
