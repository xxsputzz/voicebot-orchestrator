#!/usr/bin/env python3
"""
WebSocket Streaming LLM Service

Real-time language model service that connects to the WebSocket orchestrator
and provides token-by-token streaming responses.
"""

import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, AsyncGenerator
import websockets
from websockets.exceptions import ConnectionClosed, InvalidMessage

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import service registry components
from ws_service_registry import (
    ServiceRegistration, 
    ServiceCapabilities, 
    ServiceType,
    WebSocketServiceClient
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamingLLMService(WebSocketServiceClient):
    """WebSocket-based streaming LLM service with token streaming"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", port: int = 8022):
        # Initialize service registration
        capabilities = ServiceCapabilities(
            streaming=True,
            languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
            max_concurrent=10
        )
        
        registration = ServiceRegistration(
            service_id=f"streaming_llm_{port}",
            service_type=ServiceType.LLM.value,
            service_name="Streaming LLM Service",
            version="1.0.0",
            endpoint=f"ws://localhost:{port}",
            websocket_port=port,
            http_port=port + 100,  # HTTP port for health checks
            capabilities=capabilities,
            metadata={
                "model": model_name,
                "streaming": True,
                "max_tokens": 2048,
                "temperature": 0.7,
                "supports_functions": True
            }
        )
        
        super().__init__(registration)
        self.model_name = model_name
        self.port = port
        self.llm_engine = None
        self.active_conversations = {}
        
    async def initialize(self):
        """Initialize the LLM engine"""
        try:
            # Try different LLM implementations based on model name
            if "gpt" in self.model_name.lower() or "openai" in self.model_name.lower():
                try:
                    from enhanced_llm import EnhancedOpenAILLM
                    self.llm_engine = EnhancedOpenAILLM(
                        model_name=self.model_name,
                        max_tokens=2048,
                        temperature=0.7,
                        enable_streaming=True
                    )
                    logger.info(f"✅ Initialized OpenAI LLM with model: {self.model_name}")
                except ImportError:
                    # Fallback to mock implementation
                    from simple_gpt_llm import SimpleGPTLLM
                    self.llm_engine = SimpleGPTLLM()
                    logger.warning("⚠️  Using simple GPT implementation")
                    
            elif "mistral" in self.model_name.lower():
                try:
                    from enhanced_llm import EnhancedMistralLLM
                    self.llm_engine = EnhancedMistralLLM(
                        model_path=self.model_name,
                        max_tokens=2048,
                        temperature=0.7,
                        enable_cache=True,
                        enable_streaming=True
                    )
                    logger.info(f"✅ Initialized Mistral LLM with model: {self.model_name}")
                except ImportError:
                    logger.error("Mistral LLM not available, using fallback")
                    raise
            
            elif "ollama" in self.model_name.lower():
                try:
                    from ollama_real_llm import OllamaLLM
                    self.llm_engine = OllamaLLM(
                        model_name=self.model_name,
                        enable_streaming=True
                    )
                    logger.info(f"✅ Initialized Ollama LLM with model: {self.model_name}")
                except ImportError:
                    logger.error("Ollama LLM not available")
                    raise
            
            else:
                # Default fallback
                from simple_gpt_llm import SimpleGPTLLM
                self.llm_engine = SimpleGPTLLM()
                logger.warning(f"⚠️  Unknown model {self.model_name}, using simple GPT fallback")
            
            await self.llm_engine.initialize()
            logger.info("LLM engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM engine: {e}")
            raise
    
    def _purge_emojis(self, text: str) -> str:
        """Remove emojis from text to prevent TTS encoding issues"""
        if not text:
            return text
        
        # Comprehensive emoji removal patterns
        import re
        emoji_patterns = [
            r'[\U0001F600-\U0001F64F]',  # Emoticons
            r'[\U0001F300-\U0001F5FF]',  # Misc Symbols
            r'[\U0001F680-\U0001F6FF]',  # Transport & Maps
            r'[\U0001F700-\U0001F77F]',  # Alchemical
            r'[\U0001F780-\U0001F7FF]',  # Geometric Shapes Extended
            r'[\U0001F800-\U0001F8FF]',  # Supplemental Arrows-C
            r'[\U0001F900-\U0001F9FF]',  # Supplemental Symbols and Pictographs
            r'[\U0001FA00-\U0001FA6F]',  # Chess Symbols
            r'[\U0001FA70-\U0001FAFF]',  # Symbols and Pictographs Extended-A
            r'[\U00002702-\U000027B0]',  # Dingbats
            r'[\U000024C2-\U0001F251]'   # Enclosed characters
        ]
        
        for pattern in emoji_patterns:
            text = re.sub(pattern, '', text)
        
        return text.strip()
    
    async def stream_llm_response(self, prompt: str, conversation_id: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream LLM response token by token.
        
        Args:
            prompt: Input prompt
            conversation_id: Conversation identifier
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Yields:
            Token streaming results
        """
        try:
            # Initialize conversation if needed
            if conversation_id not in self.active_conversations:
                self.active_conversations[conversation_id] = {
                    'history': [],
                    'start_time': time.time(),
                    'token_count': 0
                }
            
            conversation = self.active_conversations[conversation_id]
            conversation['history'].append({"role": "user", "content": prompt})
            
            # Prepare streaming parameters
            stream_params = {
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1024),
                'stream': True
            }
            
            # Start streaming response
            response_text = ""
            token_count = 0
            
            async for token_data in self.llm_engine.stream_completion(
                messages=conversation['history'][-10:],  # Keep last 10 messages for context
                **stream_params
            ):
                token_count += 1
                conversation['token_count'] += 1
                
                if isinstance(token_data, dict):
                    token = token_data.get('content', '')
                    finish_reason = token_data.get('finish_reason')
                else:
                    token = str(token_data)
                    finish_reason = None
                
                if token:
                    response_text += token
                    
                    # Purge emojis for TTS compatibility
                    clean_token = self._purge_emojis(token)
                    
                    yield {
                        "type": "token",
                        "conversation_id": conversation_id,
                        "token": clean_token,
                        "token_id": token_count,
                        "partial_text": self._purge_emojis(response_text),
                        "timestamp": time.time(),
                        "finished": False
                    }
                
                # Check for completion
                if finish_reason == 'stop' or finish_reason == 'length':
                    break
            
            # Add assistant response to conversation history
            clean_response = self._purge_emojis(response_text)
            conversation['history'].append({"role": "assistant", "content": clean_response})
            
            # Send completion signal
            yield {
                "type": "completion",
                "conversation_id": conversation_id,
                "full_text": clean_response,
                "total_tokens": token_count,
                "timestamp": time.time(),
                "finished": True,
                "finish_reason": "complete"
            }
            
        except Exception as e:
            logger.error(f"Error in LLM streaming for conversation {conversation_id}: {e}")
            yield {
                "type": "error",
                "conversation_id": conversation_id,
                "message": f"LLM streaming error: {str(e)}",
                "timestamp": time.time()
            }
    
    async def handle_service_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle incoming messages from the orchestrator.
        
        Args:
            message: Incoming message
            
        Returns:
            Response message or None for streaming responses
        """
        try:
            message_type = message.get('type')
            conversation_id = message.get('conversation_id', str(uuid.uuid4()))
            
            if message_type == 'chat_completion':
                # Handle streaming chat completion
                prompt = message.get('prompt', '')
                if not prompt:
                    return {
                        "type": "error",
                        "message": "No prompt provided"
                    }
                
                logger.info(f"Starting LLM completion for conversation: {conversation_id}")
                
                # Extract parameters
                temperature = message.get('temperature', 0.7)
                max_tokens = message.get('max_tokens', 1024)
                
                # Stream response tokens back to orchestrator
                async for result in self.stream_llm_response(
                    prompt, conversation_id, 
                    temperature=temperature, 
                    max_tokens=max_tokens
                ):
                    await self.send_to_orchestrator(result)
                
                return None  # Streaming response, no immediate reply
            
            elif message_type == 'get_conversation':
                # Return conversation history
                conversation = self.active_conversations.get(conversation_id, {})
                return {
                    "type": "conversation_history",
                    "conversation_id": conversation_id,
                    "history": conversation.get('history', []),
                    "token_count": conversation.get('token_count', 0)
                }
            
            elif message_type == 'clear_conversation':
                # Clear conversation history
                if conversation_id in self.active_conversations:
                    del self.active_conversations[conversation_id]
                
                return {
                    "type": "conversation_cleared",
                    "conversation_id": conversation_id
                }
            
            elif message_type == 'health_check':
                return {
                    "type": "health_response",
                    "status": "healthy",
                    "active_conversations": len(self.active_conversations),
                    "service": "llm",
                    "model": self.model_name
                }
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
                return {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return {
                "type": "error",
                "message": f"Message processing error: {str(e)}"
            }

async def main():
    """Main entry point for the streaming LLM service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WebSocket Streaming LLM Service")
    parser.add_argument("--port", type=int, default=8022, help="Service port")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model name")
    parser.add_argument("--orchestrator-host", default="localhost", help="Orchestrator host")
    parser.add_argument("--orchestrator-port", type=int, default=9001, help="Orchestrator service port")
    
    args = parser.parse_args()
    
    # Create and initialize the service
    service = StreamingLLMService(model_name=args.model, port=args.port)
    
    try:
        # Initialize LLM engine
        await service.initialize()
        
        # Connect to orchestrator and run service
        orchestrator_uri = f"ws://{args.orchestrator_host}:{args.orchestrator_port}/service"
        await service.connect_and_serve(orchestrator_uri)
        
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        raise
    finally:
        # Cleanup
        if service.llm_engine and hasattr(service.llm_engine, 'cleanup'):
            await service.llm_engine.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
