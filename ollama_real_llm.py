#!/usr/bin/env python3
"""
Ollama-Powered Real AI LLM Integration
Replaces pattern matching with real AI generation using local Ollama models
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
import sys
import os

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from aws_microservices.prompt_loader import prompt_loader

class OllamaRealAILLM:
    """
    Real AI-powered LLM using local Ollama models
    """
    
    def __init__(self, 
                 model_name: str = "mistral:latest",
                 ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.logger = logging.getLogger(__name__)
        
    async def generate_response(
        self, 
        user_input: str, 
        conversation_history: Optional[List[Dict]] = None,
        call_type: Optional[str] = None,
        domain_context: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Generate response using Ollama AI with dynamic prompt injection
        """
        
        # Load system prompts based on call type
        system_prompt = prompt_loader.get_system_prompt(call_type=call_type)
        
        # Build conversation context
        context = ""
        if conversation_history:
            for exchange in conversation_history[-5:]:  # Last 5 exchanges
                if "human" in exchange:
                    context += f"Customer: {exchange['human']}\n"
                if "assistant" in exchange:
                    context += f"Alex: {exchange['assistant']}\n"
        
        # Add domain context if provided
        if domain_context:
            system_prompt += f"\n\nAdditional Context: {domain_context}"
        
        # Create the full prompt for AI generation
        full_prompt = f"""{system_prompt}

Previous conversation:
{context}

Customer: {user_input}
Alex:"""

        self.logger.info(f"🤖 Generating AI response with {self.model_name}")
        self.logger.info(f"📝 Prompt length: {len(full_prompt)} characters")
        
        # Generate response using Ollama
        try:
            response = await self._call_ollama_api(full_prompt)
            
            # Clean up response
            response = response.strip()
            
            # Remove any accidental repetition of the prompt
            if "Customer:" in response:
                response = response.split("Customer:")[0].strip()
            if "Alex:" in response:
                response = response.replace("Alex:", "").strip()
                
            self.logger.info(f"✅ Generated {len(response)} character response")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Ollama generation failed: {e}")
            # Fallback to enhanced pattern response
            return self._fallback_response(user_input)
    
    async def _call_ollama_api(self, prompt: str) -> str:
        """
        Call Ollama API for text generation
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 300,  # Max tokens to generate
                "stop": ["Customer:", "\n\n"]  # Stop generation at these tokens
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "")
                else:
                    raise Exception(f"Ollama API error: {response.status}")
    
    def _fallback_response(self, user_input: str) -> str:
        """
        Fallback response if Ollama is unavailable
        """
        return f"Hi! I'm Alex with Finally Payoff Debt. I heard you say '{user_input}'. As your banking specialist, I'm here to help you with any financial questions you might have. How can I assist you today?"
    
    async def test_ollama_connection(self) -> bool:
        """
        Test if Ollama is running and model is available
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Test Ollama health
                async with session.get(f"{self.ollama_url}/api/tags") as response:
                    if response.status == 200:
                        models = await response.json()
                        available_models = [model['name'] for model in models.get('models', [])]
                        
                        if self.model_name in available_models:
                            self.logger.info(f"✅ Ollama connected with model {self.model_name}")
                            return True
                        else:
                            self.logger.warning(f"❌ Model {self.model_name} not found. Available: {available_models}")
                            return False
                    else:
                        self.logger.error(f"❌ Ollama not responding: {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"❌ Ollama connection failed: {e}")
            return False

# Integration function to replace the existing LLM
class OllamaEnhancedLLM:
    """
    Drop-in replacement for EnhancedMistralLLM that uses real AI
    """
    
    def __init__(self, 
                 model_path: str = "mistral:latest",  # Now this is the Ollama model name
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 enable_cache: bool = True,
                 enable_adapters: bool = True,
                 cache_dir: str = "./cache",
                 adapter_dir: str = "./adapters"):
        
        # Extract model name from path for Ollama
        if "mistral" in model_path.lower():
            ollama_model = "mistral:latest"
        elif "gpt" in model_path.lower():
            ollama_model = "gpt-oss:20b"
        else:
            ollama_model = "mistral:latest"  # Default
            
        self.ollama_llm = OllamaRealAILLM(model_name=ollama_model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_cache = enable_cache
        self._current_domain_context = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Initialized OllamaEnhancedLLM with {ollama_model}")
    
    async def generate_response(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_cache: bool = True,
        domain_context: Optional[str] = None
    ) -> str:
        """
        Generate response using real AI instead of pattern matching
        """
        
        # Store domain context
        if domain_context:
            self._current_domain_context = domain_context
        
        # Call Ollama for real AI generation
        response = await self.ollama_llm.generate_response(
            user_input=user_input,
            conversation_history=conversation_history,
            domain_context=domain_context,
            use_cache=use_cache
        )
        
        return response
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Return mock cache stats for compatibility
        """
        return {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_hit_rate": 0.0,
            "last_query_was_cache_hit": False
        }

# Demo function
async def demo_ollama_integration():
    """
    Demo the Ollama integration with real AI
    """
    print("🚀 Ollama Real AI Integration Demo")
    print("=" * 50)
    
    # Test both models if available
    models_to_test = ["mistral:latest", "gpt-oss:20b"]
    
    for model in models_to_test:
        print(f"\n🧠 Testing {model}")
        print("-" * 30)
        
        llm = OllamaRealAILLM(model_name=model)
        
        # Test connection
        is_connected = await llm.test_ollama_connection()
        if not is_connected:
            print(f"❌ {model} not available, skipping...")
            continue
        
        # Test inputs
        test_inputs = [
            "My name is Mike and I have $15,000 in credit card debt",
            "What's the difference between APR and interest rate?",
            "I'm self-employed, can I still get a personal loan?",
            "Are there any fees I should know about?",
        ]
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\n📝 Test {i}: '{test_input}'")
            
            try:
                response = await llm.generate_response(
                    user_input=test_input,
                    call_type="inbound"
                )
                print(f"🤖 AI Response: {response}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 20)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_ollama_integration())
