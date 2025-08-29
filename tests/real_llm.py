"""
Real Local LLM module using Ollama/Mistral.
"""
import asyncio
import json
import time
from typing import List, Dict, Any, Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
    print("✅ Ollama library available")
except ImportError:
    OLLAMA_AVAILABLE = False
    print("❌ Ollama library not available")


class RealOllamaLLM:
    """Real local LLM using Ollama/Mistral."""
    
    def __init__(self, model_name: str = "mistral", host: str = "localhost:11434", max_tokens: int = 512, temperature: float = 0.7) -> None:
        """
        Initialize real Ollama LLM.
        
        Args:
            model_name: Ollama model name (mistral, llama2, etc.)
            host: Ollama server host
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.host = host
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        
        # Performance tracking
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
        
        print(f"🧠 Initializing RealOllamaLLM with model: {model_name}, host: {host}")
    
    def _get_client(self):
        """Get Ollama client."""
        if self._client is None and OLLAMA_AVAILABLE:
            try:
                self._client = ollama.Client(host=self.host)
                print(f"✅ Connected to Ollama at {self.host}")
            except Exception as e:
                print(f"❌ Failed to connect to Ollama: {e}")
                self._client = None
        return self._client
    
    async def check_model_availability(self) -> bool:
        """Check if the model is available in Ollama."""
        client = self._get_client()
        if not client:
            return False
        
        try:
            models_response = await asyncio.to_thread(client.list)
            
            # Handle different response formats
            if isinstance(models_response, dict) and 'models' in models_response:
                available_models = [model.get('name', model.get('model', '')) for model in models_response['models']]
            elif isinstance(models_response, list):
                available_models = [model.get('name', model.get('model', '')) for model in models_response]
            else:
                print(f"⚠️ Unexpected models response format: {type(models_response)}")
                available_models = []
            
            print(f"📋 Available Ollama models: {available_models}")
            
            # Check if our model is available
            is_available = any(self.model_name in model for model in available_models)
            
            if not is_available:
                print(f"⚠️ Model '{self.model_name}' not found. Attempting to pull...")
                try:
                    await asyncio.to_thread(client.pull, self.model_name)
                    print(f"✅ Successfully pulled model '{self.model_name}'")
                    return True
                except Exception as e:
                    print(f"❌ Failed to pull model '{self.model_name}': {e}")
                    return False
            
            print(f"✅ Model '{self.model_name}' is available")
            return True
            
        except Exception as e:
            print(f"❌ Error checking model availability: {e}")
            # Try a simple generation test instead
            try:
                test_response = await asyncio.to_thread(
                    client.generate,
                    model=self.model_name,
                    prompt="Test",
                    options={'num_predict': 1}
                )
                return bool(test_response.get('response'))
            except Exception as test_e:
                print(f"❌ Model test also failed: {test_e}")
                return False
    
    async def generate_response(self, user_input: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate response using real Ollama/Mistral.
        
        Args:
            user_input: User's input text
            conversation_history: Previous conversation exchanges
            
        Returns:
            Generated response text
            
        Raises:
            ValueError: If input is invalid
        """
        if not user_input or not user_input.strip():
            raise ValueError("User input cannot be empty")
        
        if not OLLAMA_AVAILABLE:
            print("⚠️ Ollama not available, using mock response")
            return await self._mock_generate(user_input)
        
        client = self._get_client()
        if not client:
            print("⚠️ Ollama client not available, using mock response")
            return await self._mock_generate(user_input)
        
        # Check model availability
        model_available = await self.check_model_availability()
        if not model_available:
            print("⚠️ Model not available, using mock response")
            return await self._mock_generate(user_input)
        
        try:
            # Build conversation context
            context = self._build_context(user_input, conversation_history)
            
            start_time = time.time()
            
            # Generate response using Ollama
            response = await asyncio.to_thread(self._generate_with_ollama, context)
            
            generation_time = time.time() - start_time
            
            # Update metrics
            self.total_calls += 1
            self.total_time += generation_time
            
            print(f"🧠 Generated response in {generation_time:.3f}s")
            return response.strip()
            
        except Exception as e:
            print(f"❌ Ollama generation failed: {e}")
            return await self._mock_generate(user_input)
    
    def _generate_with_ollama(self, context: str) -> str:
        """Generate response using Ollama."""
        client = self._get_client()
        
        response = client.generate(
            model=self.model_name,
            prompt=context,
            options={
                'num_predict': self.max_tokens,
                'temperature': self.temperature,
                'top_p': 0.9,
                'top_k': 40,
                'stop': ['Human:', 'User:', '\\n\\nHuman:', '\\n\\nUser:']
            },
            stream=False
        )
        
        return response['response']
    
    def _build_context(self, user_input: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Build conversation context for the model."""
        context_parts = []
        
        # Add system prompt for banking assistant
        system_prompt = """You are a helpful and professional banking assistant. ALWAYS respond in English only.

You assist customers with:
- Account balance inquiries
- Transaction history and details
- Money transfers and payments
- Loan and mortgage information
- Credit card and investment services
- General banking questions and support

IMPORTANT: Always respond in English. Be polite, professional, and secure. Never ask for sensitive information like passwords or PINs. If you cannot help with something, politely explain and suggest appropriate alternatives.

Conversation:"""
        
        context_parts.append(system_prompt)
        
        # Add conversation history (last 5 exchanges)
        if conversation_history:
            recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
            for exchange in recent_history:
                human_msg = exchange.get('human', exchange.get('user_input', ''))
                assistant_msg = exchange.get('assistant', exchange.get('bot_response', ''))
                if human_msg and assistant_msg:
                    context_parts.append(f"Human: {human_msg}")
                    context_parts.append(f"Assistant: {assistant_msg}")
        
        # Add current user input
        context_parts.append(f"Human: {user_input}")
        context_parts.append("Assistant:")
        
        return "\\n".join(context_parts)
    
    async def _mock_generate(self, user_input: str) -> str:
        """Fallback mock generation."""
        print("⚠️ Using mock LLM response (Ollama not available)")
        
        # Simple keyword-based responses
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["balance", "account"]):
            return "I can help you check your account balance. Please verify your identity first, and I'll provide your current balance information."
        
        elif any(word in user_lower for word in ["transfer", "send", "payment"]):
            return "I'll assist you with your transfer request. Could you please specify the accounts you'd like to transfer between and the amount?"
        
        elif any(word in user_lower for word in ["loan", "mortgage", "credit"]):
            return "I can provide information about our loan products and current rates. What type of loan or credit are you interested in?"
        
        elif any(word in user_lower for word in ["hello", "hi", "help"]):
            return "Hello! I'm your banking assistant. I can help you with account inquiries, transfers, loan information, and other banking services. How may I assist you today?"
        
        elif any(word in user_lower for word in ["thank", "thanks"]):
            return "You're very welcome! Is there anything else I can help you with today?"
        
        else:
            return f"I understand you're asking about '{user_input}'. I'm here to help with your banking needs. Could you please provide more specific details about what you'd like assistance with?"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_time = self.total_time / self.total_calls if self.total_calls > 0 else 0
        
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "average_time_per_call": avg_time,
            "model_name": self.model_name,
            "ollama_available": OLLAMA_AVAILABLE,
            "client_connected": self._client is not None
        }
    
    async def test_connection(self) -> bool:
        """Test connection to Ollama."""
        try:
            client = self._get_client()
            if not client:
                return False
            
            # Try a simple generation
            response = await asyncio.to_thread(
                client.generate,
                model=self.model_name,
                prompt="Hello",
                options={'num_predict': 5}
            )
            
            return bool(response.get('response'))
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
