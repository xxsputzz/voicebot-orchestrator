"""
Clean Natural GPT LLM - Conversational Banking Specialist

This version provides natural, conversational responses without script-heavy language.
"""

import asyncio
import logging
import aiohttp
import json
import os
import sys
from typing import Dict, List, Optional

# Import prompt system
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class NaturalGPTLLM:
    """
    Natural, conversational GPT LLM for banking conversations.
    
    Focuses on natural dialogue without heavy scripting.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-oss:20b",
        fallback_model: str = "mistral:latest"
    ):
        """Initialize natural GPT LLM."""
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.ollama_url = "http://localhost:11434"
        
        # Simple tracking
        self.total_requests = 0
        self.successful_responses = 0
        
        self.logger = logging.getLogger(__name__)
    
    async def generate_response(
        self,
        user_input: str,
        user_name: Optional[str] = None
    ) -> str:
        """Generate natural, conversational response."""
        self.total_requests += 1
        
        # Build natural conversation prompt
        prompt = self._build_natural_prompt(user_input, user_name)
        
        # Try GPT first
        try:
            response = await self._call_ollama(prompt, self.model_name)
            if response and len(response.strip()) > 15:
                self.successful_responses += 1
                return response.strip()
        except Exception as e:
            self.logger.warning(f"GPT failed, using fallback: {e}")
        
        # Fallback to Mistral
        try:
            response = await self._call_ollama(prompt, self.fallback_model)
            if response and len(response.strip()) > 15:
                self.successful_responses += 1
                return response.strip()
        except Exception as e:
            self.logger.error(f"Both models failed: {e}")
        
        # Final fallback
        return self._natural_fallback(user_input, user_name)
    
    def _build_natural_prompt(self, user_input: str, user_name: Optional[str]) -> str:
        """Build natural conversation prompt."""
        
        # Natural system prompt focused on conversation
        system_prompt = """You are Alex, a friendly banking specialist at Finally Payoff Debt. You help people consolidate high-interest debt through personal loans.

Your personality:
- Warm, approachable, and genuinely helpful
- Knowledgeable about debt consolidation and personal loans
- Always acknowledge what customers tell you specifically
- Ask relevant follow-up questions to better help them
- Keep responses conversational and natural (not scripted)

Your expertise:
- Personal loans for debt consolidation
- Interest rates typically 5.99% to 35.99% APR
- Loan amounts from $10,000 to $100,000
- Help people save money by consolidating high-interest credit card debt
- Work with various credit profiles

Remember: Be natural and conversational. Acknowledge specific details customers share (names, amounts, situations) and provide helpful guidance based on their individual needs."""
        
        # Build conversation
        conversation = f"""{system_prompt}

Customer: {user_input}

Alex:"""
        
        return conversation
    
    async def _call_ollama(self, prompt: str, model: str) -> str:
        """Call Ollama with natural conversation settings."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,  # More creative for natural conversation
                "num_predict": 200,   # Shorter, more natural responses
                "top_p": 0.9,
                "top_k": 50,
                "repeat_penalty": 1.2,
                "stop": ["Customer:", "Human:", "\n\nCustomer:", "\n\nHuman:", "System:"]
            }
        }
        
        timeout = aiohttp.ClientTimeout(total=45)
        
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
                    raise Exception(f"API error {response.status}: {error_text}")
    
    def _natural_fallback(self, user_input: str, user_name: Optional[str]) -> str:
        """Natural fallback response."""
        name_part = f" {user_name}" if user_name else ""
        
        if "debt" in user_input.lower():
            return f"Hi{name_part}! I'm Alex from Finally Payoff Debt. I understand you're dealing with debt - that can be really stressful. I specialize in helping people consolidate high-interest debt into lower-rate personal loans. What's your current debt situation?"
        elif "qualify" in user_input.lower():
            return f"Hi{name_part}! I'm Alex from Finally Payoff Debt. Great question about qualifying! We work with people across different credit profiles. Generally, we look for stable income and reasonable debt levels. What's your current monthly income?"
        elif "rate" in user_input.lower() or "apr" in user_input.lower():
            return f"Hi{name_part}! I'm Alex from Finally Payoff Debt. Our personal loan rates typically range from 5.99% to 35.99% APR, depending on your credit profile and other factors. Much lower than most credit cards! What's your current situation?"
        else:
            return f"Hi{name_part}! I'm Alex from Finally Payoff Debt. I'm here to help you with debt consolidation and personal loans. What questions can I answer for you today?"
    
    def get_stats(self) -> Dict:
        """Get performance stats."""
        success_rate = (self.successful_responses / max(self.total_requests, 1)) * 100
        return {
            "total_requests": self.total_requests,
            "success_rate": f"{success_rate:.1f}%"
        }


def create_natural_gpt_llm(model_name: str = "gpt-oss:20b") -> NaturalGPTLLM:
    """Create natural GPT LLM."""
    return NaturalGPTLLM(model_name=model_name)
