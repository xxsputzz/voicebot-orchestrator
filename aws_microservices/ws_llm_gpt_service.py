#!/usr/bin/env python3
"""
WebSocket GPT LLM Service - Converted from HTTP to WebSocket Streaming
Language model using GPT with real-time token streaming capabilities
"""

import asyncio
import json
import logging
import sys
import os
import time
import re
from typing import Dict, Any, List, Optional
import websockets

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import WebSocket base client and service registry
try:
    from ws_service_registry import ServiceRegistration, ServiceCapabilities
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("Warning: WebSocket infrastructure not available")

# Import your existing LLM implementation
try:
    from voicebot_orchestrator.enhanced_llm import EnhancedMistralLLM
    # Import prompt loader and conversation manager
    from prompt_loader import prompt_loader
    from conversation_manager import ConversationManager
    LLM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️  LLM import error: {e}")
    LLM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# EMOJI PURGING: Comprehensive emoji detection and removal for TTS compatibility
def _purge_emojis_from_llm_response(text: str) -> str:
    """
    Remove all emojis from LLM responses to prevent TTS encoding issues.
    Simply removes emojis without text replacement to preserve sentence meaning.
    """
    if not text:
        return text
    
    # Nuclear emoji removal - comprehensive Unicode ranges
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
        r'[\U0000FE0F]',             # Variation selector
        r'[\U0000200D]',             # Zero width joiner
    ]
    for pattern in emoji_patterns:
        text = re.sub(pattern, '', text)
    
    # Clean up extra spaces but preserve trailing spaces for token streaming
    text = re.sub(r'\s+', ' ', text)
    # Don't strip() here as it removes trailing spaces needed for token joining
    
    return text

class WebSocketLLMService:
    """WebSocket-enabled LLM service that converts existing HTTP service to streaming WebSocket"""
    
    def __init__(self, model_path: str = "microsoft/DialoGPT-small", max_tokens: int = 512):
        self.service_id = "llm_gpt_ws"
        self.service_name = "WebSocket GPT LLM"
        self.websocket = None
        self.orchestrator_url = "ws://localhost:9001"  # Service port
        self.running = False
        
        # Initialize LLM
        self.llm_service = None
        self.model_path = model_path
        self.max_tokens = max_tokens
        
        # Conversation management
        if LLM_AVAILABLE:
            self.conversation_manager = ConversationManager()
        else:
            self.conversation_manager = None
        
        # Session management for streaming
        self.sessions = {}  # session_id -> session_data
        
        # Service registration info
        self.registration = ServiceRegistration(
            service_id=self.service_id,
            service_name=self.service_name,
            service_type="llm",
            version="1.0.0",
            endpoint="localhost",
            websocket_port=8002,
            http_port=8002,
            capabilities=ServiceCapabilities(
                realtime=True,
                streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"],
                max_concurrent=10,
                latency_ms=150
            ),
            metadata={
                "model": model_path,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "implementation": "enhanced_mistral",
                "emoji_filtering": True,
                "conversation_aware": True,
                "input_types": ["text/plain"],
                "output_types": ["text/plain"],
                "status": "starting"
            }
        )
    
    async def initialize_llm(self):
        """Initialize the LLM engine"""
        if not LLM_AVAILABLE:
            logging.warning("⚠️  LLM dependencies not available, using mock responses")
            self.registration.metadata["implementation_type"] = "mock"
            self.registration.status = "ready"
            return True
        
        try:
            logging.info(f"[LLM-WS] Initializing Enhanced Mistral LLM (model: {self.model_path})")
            
            # Initialize with memory-optimized GPT model
            self.llm_service = EnhancedMistralLLM(
                model_path=self.model_path,
                max_tokens=self.max_tokens,
                temperature=0.7,
                enable_cache=True,
                enable_adapters=True,
                cache_dir="./cache",
                adapter_dir="./adapters"
            )
            
            self.registration.status = "ready"
            self.registration.metadata["implementation_type"] = "real"
            logging.info("✅ LLM initialized successfully!")
            return True
            
        except Exception as e:
            logging.error(f"❌ LLM initialization failed: {e}")
            self.registration.status = "error"
            self.registration.metadata["error"] = str(e)
            self.registration.metadata["implementation_type"] = "mock"
            return False
    
    async def connect_to_orchestrator(self):
        """Connect to WebSocket orchestrator and register service"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries and not self.running:
            try:
                logging.info(f"[LLM-WS] Connecting to orchestrator: {self.orchestrator_url}")
                
                self.websocket = await websockets.connect(
                    self.orchestrator_url,
                    ping_interval=20,
                    ping_timeout=10
                )
                
                # Register with orchestrator
                registration_msg = {
                    "type": "service_registration",
                    "data": self.registration.to_dict()
                }
                await self.websocket.send(json.dumps(registration_msg))
                
                # Start message handling
                self.running = True
                logging.info("✅ Connected to orchestrator and registered LLM service")
                
                return True
                
            except Exception as e:
                retry_count += 1
                logging.error(f"❌ Connection attempt {retry_count} failed: {e}")
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                
        return False
    
    async def handle_message(self, message_data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        try:
            message_type = message_data.get("type")
            data = message_data.get("data", {})
            session_id = message_data.get("session_id")
            
            if message_type == "text_input":
                await self.handle_text_input(data, session_id)
            elif message_type == "generate_request":
                await self.handle_generate_request(data, session_id)
            elif message_type == "stream_request":
                await self.handle_stream_request(data, session_id)
            elif message_type == "session_start":
                await self.handle_session_start(data, session_id)
            elif message_type == "session_end":
                await self.handle_session_end(data, session_id)
            else:
                logging.warning(f"[LLM-WS] Unknown message type: {message_type}")
                
        except Exception as e:
            logging.error(f"[LLM-WS] Error handling message: {e}")
            await self.send_error(session_id, f"Message handling error: {e}")
    
    async def handle_text_input(self, data: Dict[str, Any], session_id: str):
        """Handle text input for LLM processing"""
        try:
            text = data.get("text", "").strip()
            stream_tokens = data.get("stream_tokens", True)
            use_cache = data.get("use_cache", True)
            domain_context = data.get("domain_context")
            conversation_history = data.get("conversation_history", [])
            
            if not text:
                await self.send_error(session_id, "No text provided for processing")
                return
            
            # Initialize session if needed
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    "conversation_history": [],
                    "start_time": time.time(),
                    "metadata": data.get("metadata", {})
                }
            
            session = self.sessions[session_id]
            
            # Add to conversation history
            session["conversation_history"].append({
                "role": "user",
                "content": text,
                "timestamp": time.time()
            })
            
            if stream_tokens:
                await self.stream_llm_response(text, session_id, session, domain_context, use_cache)
            else:
                await self.generate_llm_response(text, session_id, session, domain_context, use_cache)
                
        except Exception as e:
            logging.error(f"[LLM-WS] Error handling text input: {e}")
            await self.send_error(session_id, f"Text processing error: {e}")
    
    async def handle_generate_request(self, data: Dict[str, Any], session_id: str):
        """Handle generate request (backward compatibility)"""
        await self.handle_text_input(data, session_id)
    
    async def handle_stream_request(self, data: Dict[str, Any], session_id: str):
        """Handle streaming request"""
        data["stream_tokens"] = True
        await self.handle_text_input(data, session_id)
    
    async def stream_llm_response(self, text: str, session_id: str, session: Dict, domain_context: str = None, use_cache: bool = True):
        """Generate streaming LLM response token by token"""
        start_time = time.time()
        
        try:
            # Send streaming start notification
            await self.websocket.send(json.dumps({
                "type": "llm_stream_start",
                "session_id": session_id,
                "data": {
                    "service": self.service_id,
                    "input_text": text[:100] + "..." if len(text) > 100 else text
                }
            }))
            
            if self.llm_service and LLM_AVAILABLE:
                # Use real LLM with streaming
                try:
                    # Prepare conversation context
                    conversation_context = session.get("conversation_history", [])
                    
                    # Generate response with streaming
                    full_response = ""
                    async for token in self._stream_llm_tokens(text, conversation_context, domain_context, use_cache):
                        # Clean token for TTS compatibility
                        clean_token = _purge_emojis_from_llm_response(token)
                        full_response += clean_token
                        
                        # Send token
                        await self.websocket.send(json.dumps({
                            "type": "llm_token",
                            "session_id": session_id,
                            "data": {
                                "token": clean_token,
                                "full_text": full_response,
                                "is_final": False
                            }
                        }))
                        
                        # Small delay for realistic streaming
                        await asyncio.sleep(0.02)
                    
                    # Clean final response
                    full_response = _purge_emojis_from_llm_response(full_response.strip())
                    
                except Exception as e:
                    logging.error(f"[LLM-WS] Real LLM streaming error: {e}")
                    # Fallback to mock
                    full_response = await self._generate_mock_streaming_response(text, session_id)
            else:
                # Use mock streaming response
                full_response = await self._generate_mock_streaming_response(text, session_id)
            
            processing_time = time.time() - start_time
            
            # Add to conversation history
            session["conversation_history"].append({
                "role": "assistant",
                "content": full_response,
                "timestamp": time.time(),
                "processing_time": processing_time
            })
            
            # Send final response
            await self.websocket.send(json.dumps({
                "type": "llm_stream_complete",
                "session_id": session_id,
                "data": {
                    "full_text": full_response,
                    "processing_time_seconds": processing_time,
                    "token_count": len(full_response.split()),
                    "is_final": True
                }
            }))
            
            logging.info(f"[LLM-WS] Streaming response completed in {processing_time:.2f}s: '{full_response[:100]}...'")
            
        except Exception as e:
            logging.error(f"[LLM-WS] Error in streaming response: {e}")
            await self.send_error(session_id, f"Streaming error: {e}")
    
    async def _stream_llm_tokens(self, text: str, conversation_context: List, domain_context: str = None, use_cache: bool = True):
        """Stream tokens from real LLM (placeholder for actual streaming implementation)"""
        try:
            # Generate full response first (since our LLM might not support true streaming)
            response = await self.llm_service.generate_response(
                user_input=text,  # Correct parameter name
                conversation_history=conversation_context,
                use_cache=use_cache,
                domain_context=domain_context
            )
            
            # Simulate streaming by splitting into tokens/words
            words = response.split()
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                yield token
                
        except Exception as e:
            logging.error(f"[LLM-WS] Error in token streaming: {e}")
            # Fallback to simple mock
            words = f"I understand you said: {text}. Here's my response.".split()
            for word in words:
                yield word + " "
    
    async def _generate_mock_streaming_response(self, text: str, session_id: str) -> str:
        """Generate mock streaming response for testing"""
        mock_response = f"I understand you said: '{text[:50]}...'. This is a mock streaming response from the WebSocket LLM service. The real implementation would provide more sophisticated language understanding and generation."
        
        # Stream mock tokens
        words = mock_response.split()
        full_response = ""
        
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            full_response += token
            
            await self.websocket.send(json.dumps({
                "type": "llm_token",
                "session_id": session_id,
                "data": {
                    "token": token,
                    "full_text": full_response,
                    "is_final": i == len(words) - 1
                }
            }))
            
            # Simulate processing time
            await asyncio.sleep(0.03)
        
        return full_response
    
    async def generate_llm_response(self, text: str, session_id: str, session: Dict, domain_context: str = None, use_cache: bool = True):
        """Generate non-streaming LLM response"""
        start_time = time.time()
        
        try:
            if self.llm_service and LLM_AVAILABLE:
                # Use real LLM
                conversation_context = session.get("conversation_history", [])
                response = await self.llm_service.generate_response(
                    user_input=text,  # Correct parameter name
                    conversation_history=conversation_context,
                    use_cache=use_cache,
                    domain_context=domain_context
                )
            else:
                # Mock response
                response = f"I understand you said: '{text[:100]}...'. This is a mock response from the WebSocket LLM service."
            
            # Clean response for TTS compatibility
            clean_response = _purge_emojis_from_llm_response(response.strip())
            processing_time = time.time() - start_time
            
            # Add to conversation history
            session["conversation_history"].append({
                "role": "assistant",
                "content": clean_response,
                "timestamp": time.time(),
                "processing_time": processing_time
            })
            
            # Send response
            response_msg = {
                "type": "llm_response",
                "session_id": session_id,
                "data": {
                    "text": clean_response,
                    "processing_time_seconds": processing_time,
                    "token_count": len(clean_response.split()),
                    "use_cache": use_cache,
                    "domain_context": domain_context
                }
            }
            
            await self.websocket.send(json.dumps(response_msg))
            logging.info(f"[LLM-WS] Response generated in {processing_time:.2f}s: '{clean_response[:100]}...'")
            
        except Exception as e:
            logging.error(f"[LLM-WS] Error generating response: {e}")
            await self.send_error(session_id, f"Response generation error: {e}")
    
    async def handle_session_start(self, data: Dict[str, Any], session_id: str):
        """Handle session start"""
        self.sessions[session_id] = {
            "conversation_history": [],
            "start_time": time.time(),
            "metadata": data.get("metadata", {}),
            "domain_context": data.get("domain_context"),
            "preferences": data.get("preferences", {})
        }
        
        response = {
            "type": "session_started",
            "session_id": session_id,
            "data": {
                "service": self.service_id,
                "ready": True,
                "capabilities": {
                    "streaming": True,
                    "conversation_aware": True,
                    "emoji_filtering": True
                }
            }
        }
        await self.websocket.send(json.dumps(response))
        logging.info(f"[LLM-WS] Session started: {session_id}")
    
    async def handle_session_end(self, data: Dict[str, Any], session_id: str):
        """Handle session end"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Save conversation if needed
            conversation_length = len(session["conversation_history"])
            duration = time.time() - session["start_time"]
            
            # Clean up session
            del self.sessions[session_id]
            
            response = {
                "type": "session_ended",
                "session_id": session_id,
                "data": {
                    "service": self.service_id,
                    "duration": duration,
                    "conversation_length": conversation_length
                }
            }
            await self.websocket.send(json.dumps(response))
            logging.info(f"[LLM-WS] Session ended: {session_id} (duration: {duration:.2f}s, messages: {conversation_length})")
    
    async def send_error(self, session_id: str, error_message: str):
        """Send error message"""
        if self.websocket:
            error_response = {
                "type": "error",
                "session_id": session_id,
                "data": {
                    "service": self.service_id,
                    "error": error_message,
                    "timestamp": time.time()
                }
            }
            await self.websocket.send(json.dumps(error_response))
    
    async def send_health_update(self):
        """Send periodic health updates"""
        while self.running:
            try:
                if self.websocket:
                    health_data = {
                        "type": "service_health",
                        "data": {
                            "service_id": self.service_id,
                            "status": "healthy" if self.llm_service or not LLM_AVAILABLE else "degraded",
                            "active_sessions": len(self.sessions),
                            "model": self.model_path,
                            "max_tokens": self.max_tokens,
                            "implementation": self.registration.metadata.get("implementation_type", "unknown"),
                            "timestamp": time.time()
                        }
                    }
                    await self.websocket.send(json.dumps(health_data))
                
                await asyncio.sleep(30)  # Send health update every 30 seconds
                
            except Exception as e:
                logging.error(f"[LLM-WS] Error sending health update: {e}")
                break
    
    async def message_loop(self):
        """Main message handling loop with keep-alive"""
        try:
            while self.running:
                try:
                    # Wait for messages with timeout to allow for keep-alive checks
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                    try:
                        data = json.loads(message)
                        await self.handle_message(data)
                    except json.JSONDecodeError:
                        logging.error(f"[WS] Invalid JSON received: {message}")
                    except Exception as e:
                        logging.error(f"[WS] Error processing message: {e}")
                        
                except asyncio.TimeoutError:
                    # No message received, continue loop (keep-alive)
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logging.info("[WS] WebSocket connection closed")
                    break
                    
        except Exception as e:
            logging.error(f"[WS] Message loop error: {e}")
        finally:
            self.running = False
    
    async def run(self):
        """Run the WebSocket LLM service"""
        # Initialize LLM engine
        await self.initialize_llm()  # Don't fail if LLM init fails, use mock instead
        
        # Connect to orchestrator
        if not await self.connect_to_orchestrator():
            logging.error("[LLM-WS] Failed to connect to orchestrator")
            return False
        
        try:
            # Start health monitoring task
            health_task = asyncio.create_task(self.send_health_update())
            
            # Start message handling
            await self.message_loop()
            
        except KeyboardInterrupt:
            logging.info("[LLM-WS] Received interrupt, shutting down...")
        except Exception as e:
            logging.error(f"[LLM-WS] Service error: {e}")
        finally:
            self.running = False
            if self.websocket:
                await self.websocket.close()
            
            # Clean up sessions
            self.sessions.clear()
            
            logging.info("[LLM-WS] Service shut down complete")
        
        return True

def safe_print(text):
    """Safe print function that handles Unicode characters for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)

async def main():
    """Main entry point"""
    safe_print("🤖 Starting WebSocket GPT LLM Service...")
    safe_print("📡 Connecting to orchestrator at ws://localhost:9001")
    safe_print("🔄 Converting HTTP LLM service to WebSocket streaming")
    safe_print("🚫 Emoji filtering enabled for TTS compatibility")
    safe_print("-" * 60)
    
    service = WebSocketLLMService(
        model_path="microsoft/DialoGPT-small",  # Use stable model
        max_tokens=512
    )
    
    success = await service.run()
    
    if success:
        safe_print("✅ WebSocket LLM service completed successfully")
    else:
        safe_print("❌ WebSocket LLM service encountered errors")
        sys.exit(1)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the service
    asyncio.run(main())
