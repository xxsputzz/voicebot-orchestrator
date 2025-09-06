#!/usr/bin/env python3
"""
WebSocket Orchestrator Service

Central hub for real-time streaming communication between services and clients.
Handles service registration, message routing, and session management.
"""

import asyncio
import json
import logging
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import websockets
try:
    from websockets.server import WebSocketServerProtocol
except ImportError:
    # Fallback for newer websockets versions
    from typing import Any
    WebSocketServerProtocol = Any
import aiohttp
from aiohttp import web

# Import our service registry
try:
    from ws_service_registry import ServiceRegistry, ServiceRegistration, validate_message, MESSAGE_TYPES
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    print("Warning: Service registry not available, using basic mode")

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

# Message Types
class MessageType(Enum):
    # Client Messages
    AUDIO_CHUNK = "audio_chunk"
    TEXT_INPUT = "text_input"
    
    # Service Messages  
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    RESPONSE_TOKEN = "response_token"
    RESPONSE_FINAL = "response_final"
    AUDIO_OUTPUT = "audio_output"
    
    # Control Messages
    SERVICE_REGISTER = "service_register"
    SERVICE_UNREGISTER = "service_unregister"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

@dataclass
class StreamingMessage:
    """Standard message format for WebSocket communication"""
    type: str
    session_id: str
    timestamp: str
    data: Any
    metadata: Dict[str, Any] = None
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StreamingMessage':
        """Create message from JSON string"""
        data = json.loads(json_str)
        return cls(**data)

@dataclass
class ServiceInfo:
    """Information about registered services"""
    service_id: str
    service_type: str  # stt, llm, tts
    websocket: Any
    endpoint: str
    capabilities: Dict[str, Any]
    last_heartbeat: float
    load_factor: float = 0.0
    active_sessions: Set[str] = None
    
    def __post_init__(self):
        if self.active_sessions is None:
            self.active_sessions = set()

@dataclass  
class ClientSession:
    """Client session information"""
    session_id: str
    client_websocket: Any
    client_type: str  # headset, twilio, web
    assigned_services: Dict[str, str]  # service_type -> service_id
    created_at: float
    last_activity: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class WebSocketOrchestrator:
    """Central WebSocket orchestrator for streaming services"""
    
    def __init__(self, host: str = "localhost", port: int = 9000, service_port: int = 9001):
        self.host = host
        self.port = port
        self.service_port = service_port
        
        # Service registry
        if REGISTRY_AVAILABLE:
            self.service_registry = ServiceRegistry()
            # Also keep legacy structure for compatibility
            self.services: Dict[str, ServiceInfo] = {}
            self.service_types: Dict[str, Set[str]] = {
                "stt": set(),
                "llm": set(), 
                "tts": set()
            }
        else:
            self.services: Dict[str, ServiceInfo] = {}
            self.service_types: Dict[str, Set[str]] = {
                "stt": set(),
                "llm": set(), 
                "tts": set()
            }
        
        # Client sessions
        self.sessions: Dict[str, ClientSession] = {}
        
        # Service connections (WebSocket connections)
        self.service_connections: Dict[str, Any] = {}
        
        # Configuration
        self.heartbeat_interval = 30
        self.session_timeout = 300
        self.max_connections = 1000
        self.start_time = time.time()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    async def start_server(self):
        """Start the WebSocket orchestrator server"""
        self.logger.info(f"Starting WebSocket Orchestrator on {self.host}:{self.port}")
        
        # Start client WebSocket server
        client_server = websockets.serve(
            self.handle_client_connection,
            self.host,
            self.port,
            max_size=10**7,  # 10MB max message size
            ping_interval=20,
            ping_timeout=10
        )
        
        # Start service WebSocket server  
        service_server = websockets.serve(
            self.handle_service_connection,
            self.host,
            self.service_port,
            max_size=10**7,
            ping_interval=20,
            ping_timeout=10
        )
        
        # Start HTTP server for health checks and management
        http_app = web.Application()
        http_app.router.add_get('/health', self.health_check)
        http_app.router.add_get('/services', self.list_services)
        http_app.router.add_get('/sessions', self.list_sessions)
        
        # Service registration endpoints
        if REGISTRY_AVAILABLE:
            http_app.router.add_post('/register_service', self.register_service_http)
            http_app.router.add_delete('/unregister_service/{service_id}', self.unregister_service_http)
            http_app.router.add_get('/registry_status', self.registry_status)
        
        http_runner = web.AppRunner(http_app)
        await http_runner.setup()
        http_site = web.TCPSite(http_runner, self.host, 8080)  # Use port 8080 instead of 8000
        
        self.logger.info("Starting HTTP server on port 8080...")
        
        # Start all servers
        await asyncio.gather(
            client_server,
            service_server, 
            http_site.start(),
            self.heartbeat_monitor(),
            self.cleanup_monitor()
        )
        
        self.logger.info("All servers started successfully")
    
    async def handle_client_connection(self, websocket: Any):
        """Handle client WebSocket connections (headset, Twilio, web)"""
        session_id = str(uuid.uuid4())
        client_type = "headset"  # Default client type since we can't extract from path
        
        self.logger.info(f"New client connection: {session_id} ({client_type})")
        
        # Create session
        session = ClientSession(
            session_id=session_id,
            client_websocket=websocket,
            client_type=client_type,
            assigned_services={},
            created_at=time.time(),
            last_activity=time.time()
        )
        
        self.sessions[session_id] = session
        
        try:
            # Send session start confirmation
            welcome_msg = StreamingMessage(
                type=MessageType.SESSION_START.value,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                data={"status": "connected", "client_type": client_type},
                metadata={"orchestrator_version": "1.0.0"}
            )
            
            await websocket.send(welcome_msg.to_json())
            
            # Handle client messages
            async for message in websocket:
                await self.process_client_message(session_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client disconnected: {session_id}")
        except Exception as e:
            self.logger.error(f"Error handling client {session_id}: {e}")
        finally:
            await self.cleanup_session(session_id)
    
    async def handle_service_connection(self, websocket: Any):
        """Handle service WebSocket connections (STT, LLM, TTS)"""
        service_id = None
        
        try:
            # Handle service registration and ongoing messages in single loop
            service_registered = False
            async for message in websocket:
                try:
                    # Try to parse as StreamingMessage first
                    msg = StreamingMessage.from_json(message)
                except:
                    # If that fails, try to parse as simple JSON
                    try:
                        raw_data = json.loads(message)
                        # Convert simple JSON to StreamingMessage format
                        msg = StreamingMessage(
                            type=raw_data.get("type", "unknown"),
                            session_id=raw_data.get("session_id", "temp_session"),
                            timestamp=raw_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                            data=raw_data.get("data", {}),
                            metadata=raw_data.get("metadata", {})
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to parse message: {e}")
                        continue
                
                if not service_registered and (msg.type == MessageType.SERVICE_REGISTER.value or msg.type == "service_registration"):
                    service_id = await self.register_service(websocket, msg)
                    if service_id:
                        service_registered = True
                        self.logger.info(f"Service registered: {service_id}")
                    else:
                        await websocket.close(code=4000, reason="Service registration failed")
                        return
                        
                elif service_registered:
                    # Handle ongoing service messages
                    await self.process_service_message(service_id, message)
                else:
                    # Service not registered yet, close connection
                    await websocket.close(code=4000, reason="Service registration required")
                    return
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Service disconnected: {service_id if 'service_id' in locals() else 'unknown'}")
        except Exception as e:
            self.logger.error(f"Error handling service {service_id if 'service_id' in locals() else 'unknown'}: {e}")
        finally:
            if 'service_id' in locals() and service_id:
                await self.unregister_service(service_id)
    
    async def register_service(self, websocket: Any, msg: StreamingMessage) -> str:
        """Register a new service"""
        service_data = msg.data
        service_id = service_data.get("service_id", str(uuid.uuid4()))
        service_type = service_data["service_type"]
        
        service_info = ServiceInfo(
            service_id=service_id,
            service_type=service_type,
            websocket=websocket,
            endpoint=service_data.get("endpoint", ""),
            capabilities=service_data.get("capabilities", {}),
            last_heartbeat=time.time()
        )
        
        if REGISTRY_AVAILABLE and hasattr(self, 'service_registry'):
            # Store in service registry
            try:
                from ws_service_registry import ServiceRegistration, ServiceCapabilities
                
                capabilities = ServiceCapabilities(**service_data.get("capabilities", {}))
                registration = ServiceRegistration(
                    service_id=service_id,
                    service_type=service_type,
                    service_name=service_data.get("service_name", f"{service_type} service"),
                    version=service_data.get("version", "1.0.0"),
                    endpoint=service_data.get("endpoint", ""),
                    websocket_port=service_data.get("websocket_port", 0),
                    http_port=service_data.get("http_port", 0),
                    capabilities=capabilities,
                    metadata=service_data.get("metadata", {})
                )
                
                self.service_registry.register_service(registration)
                # Also store websocket connection
                self.service_connections[service_id] = websocket
                self.logger.info(f"Service registered successfully: {service_id}, total services: {len(self.service_registry.services)}")
                
            except Exception as e:
                self.logger.error(f"Failed to register with service registry: {e}")
                # Fallback to legacy mode
                self.services[service_id] = service_info
                if hasattr(self, 'service_types'):
                    self.service_types[service_type].add(service_id)
        else:
            # Legacy mode
            self.services[service_id] = service_info
            if hasattr(self, 'service_types'):
                self.service_types[service_type].add(service_id)
        
        # Send registration confirmation
        response = StreamingMessage(
            type="registration_confirmed",
            session_id=service_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data={"service_id": service_id, "status": "registered"}
        )
        
        await websocket.send(response.to_json())
        return service_id
    
    async def unregister_service(self, service_id: str):
        """Unregister a service"""
        if REGISTRY_AVAILABLE and hasattr(self, 'service_registry'):
            # Remove from service registry
            self.service_registry.unregister_service(service_id)
            # Remove websocket connection
            if service_id in self.service_connections:
                del self.service_connections[service_id]
        else:
            # Legacy mode
            if service_id in self.services:
                service_info = self.services[service_id]
                if hasattr(self, 'service_types'):
                    self.service_types[service_info.service_type].discard(service_id)
                del self.services[service_id]
            
        self.logger.info(f"Service unregistered: {service_id}")
    
    async def process_client_message(self, session_id: str, message: str):
        """Process messages from clients"""
        try:
            # First try to parse as raw JSON to handle health_check and other simple messages
            raw_data = json.loads(message)
            
            # Handle health check directly
            if raw_data.get("type") == "health_check":
                await self.send_health_status_to_client(session_id)
                return
            
            # For other messages, parse as StreamingMessage
            msg = StreamingMessage.from_json(message)
            session = self.sessions[session_id]
            session.last_activity = time.time()
            
            # Route message based on type
            if msg.type == MessageType.AUDIO_CHUNK.value:
                self.logger.info(f"🎯 DEBUG: Received audio_chunk message for session {session_id}")
                await self.route_to_stt(session_id, msg)
            elif msg.type == MessageType.TEXT_INPUT.value:
                await self.route_to_llm(session_id, msg)
            elif msg.type == "tts_request":
                await self.route_to_tts(session_id, msg)
            elif msg.type == MessageType.HEARTBEAT.value:
                await self.handle_client_heartbeat(session_id)
            else:
                self.logger.warning(f"Unknown message type from client {session_id}: {msg.type}")
                
        except Exception as e:
            self.logger.error(f"Error processing client message from {session_id}: {e}")
    
    async def process_service_message(self, service_id: str, message: str):
        """Process messages from services"""
        try:
            # First try to parse as regular JSON for service health messages
            raw_data = json.loads(message)
            
            # Handle service health messages
            if raw_data.get("type") == "service_health":
                if REGISTRY_AVAILABLE and hasattr(self, 'service_registry'):
                    # Update service health in registry using the existing method
                    self.service_registry.service_health[service_id] = time.time()
                else:
                    # Update legacy service info
                    if service_id in self.services:
                        self.services[service_id].last_heartbeat = time.time()
                self.logger.debug(f"Received health update from service {service_id}")
                return
            
            # Try to parse as StreamingMessage for other message types
            try:
                msg = StreamingMessage.from_json(message)
            except Exception:
                # If it can't be parsed as StreamingMessage, try creating one from raw data
                msg = StreamingMessage(
                    type=raw_data.get("type", "unknown"),
                    session_id=raw_data.get("session_id", service_id),
                    timestamp=raw_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    data=raw_data.get("data", {}),  # Fixed: data is required and comes after timestamp
                    metadata=raw_data.get("metadata")
                )
            
            # Update service heartbeat
            if service_id in self.services:
                service_info = self.services[service_id]
                service_info.last_heartbeat = time.time()
            
            # Route message based on type and target session
            if msg.type == MessageType.TRANSCRIPT_PARTIAL.value:
                await self.route_transcript_to_client(msg)
                await self.route_to_llm_for_context(msg)
            elif msg.type == MessageType.TRANSCRIPT_FINAL.value:
                await self.route_transcript_to_client(msg)
                await self.route_to_llm(msg.session_id, msg)
            elif msg.type == MessageType.RESPONSE_TOKEN.value:
                await self.route_to_client(msg)
                await self.route_to_tts(msg)
            elif msg.type == MessageType.AUDIO_OUTPUT.value:
                await self.route_to_client(msg)
            elif msg.type in ["audio_output", "tts_response"]:
                # Handle TTS responses - route back to client
                await self.route_to_client(msg)
            elif msg.type == MessageType.HEARTBEAT.value:
                # Heartbeat handled by updating last_heartbeat above
                pass
            elif msg.type in ["llm_stream_start", "llm_token", "llm_stream_complete", "text_response"]:
                # Handle LLM streaming responses - route back to client
                await self.route_to_client(msg)
            elif msg.type in ["llm_response", "llm_final_response"]:
                # Handle LLM response messages - route back to client
                await self.route_to_client(msg)
            elif msg.type == "error":
                # Handle error messages from services - route back to client
                await self.route_to_client(msg)
            else:
                self.logger.warning(f"Unknown message type from service {service_id}: {msg.type}")
                
        except Exception as e:
            self.logger.error(f"Error processing service message from {service_id}: {e}")
    
    async def route_to_stt(self, session_id: str, msg: StreamingMessage):
        """Route audio to STT service"""
        self.logger.info(f"🎯 DEBUG: Routing to STT for session {session_id}")
        stt_service = self.get_best_service("stt", session_id)
        
        if stt_service:
            self.logger.info(f"🎯 DEBUG: Found STT service: {stt_service.service_id}")
            self.logger.info(f"🎯 DEBUG: STT websocket exists: {stt_service.websocket is not None}")
            if stt_service.websocket:
                self.logger.info(f"🎯 DEBUG: Sending message to STT service...")
                await stt_service.websocket.send(msg.to_json())
                self.logger.info(f"🎯 DEBUG: Message sent to STT service successfully")
            else:
                self.logger.error(f"🎯 DEBUG: STT service websocket is None!")
                await self.send_error_to_client(session_id, "STT service websocket not available")
        else:
            self.logger.error(f"🎯 DEBUG: No STT service found!")
            await self.send_error_to_client(session_id, "No STT service available")
    
    async def route_to_llm(self, session_id: str, msg: StreamingMessage):
        """Route text to LLM service"""
        llm_service = self.get_best_service("llm", session_id)
        if llm_service:
            await llm_service.websocket.send(msg.to_json())
        else:
            await self.send_error_to_client(session_id, "No LLM service available")
    
    async def route_to_tts(self, session_id: str, msg: StreamingMessage):
        """Route text to TTS service"""
        tts_service = self.get_best_service("tts", session_id)
        if tts_service:
            await tts_service.websocket.send(msg.to_json())
            self.logger.info(f"Routed TTS request from {session_id} to TTS service")
        else:
            await self.send_error_to_client(session_id, "No TTS service available")
    
    async def route_to_llm_for_context(self, msg: StreamingMessage):
        """Route partial transcript to LLM for context (non-blocking)"""
        llm_service = self.get_best_service("llm", msg.session_id)
        if llm_service:
            context_msg = StreamingMessage(
                type="context_update",
                session_id=msg.session_id,
                timestamp=msg.timestamp,
                data=msg.data,
                metadata={"partial": True}
            )
            await llm_service.websocket.send(context_msg.to_json())
    
    async def route_tokens_to_tts(self, msg: StreamingMessage):
        """Route response tokens to TTS service"""
        tts_service = self.get_best_service("tts", msg.session_id)
        if tts_service:
            await tts_service.websocket.send(msg.to_json())
    
    async def route_to_client(self, msg: StreamingMessage):
        """Route message to client"""
        # Try direct session ID first
        if msg.session_id in self.sessions:
            session = self.sessions[msg.session_id]
            await session.client_websocket.send(msg.to_json())
            self.logger.info(f"🎯 DEBUG: Message routed to client session {msg.session_id}")
        else:
            # If direct session not found, try to find by matching pattern
            # This handles cases where client uses custom session IDs
            self.logger.warning(f"🎯 DEBUG: Session {msg.session_id} not found directly. Available sessions: {list(self.sessions.keys())}")
            
            # For now, route to all active sessions (simple fallback)
            # This ensures the message gets to the client even with session ID mismatch
            if self.sessions:
                # Route to the most recent session
                recent_session = max(self.sessions.values(), key=lambda s: s.last_activity)
                await recent_session.client_websocket.send(msg.to_json())
                self.logger.info(f"🎯 DEBUG: Message routed to most recent session {recent_session.session_id}")
            else:
                self.logger.warning(f"🎯 DEBUG: No sessions available to route message to")
    
    async def route_transcript_to_client(self, msg: StreamingMessage):
        """Route transcript to client for display"""
        await self.route_to_client(msg)
    
    def get_best_service(self, service_type: str, session_id: str) -> Optional[ServiceInfo]:
        """Get the best available service of given type"""
        if REGISTRY_AVAILABLE and hasattr(self, 'service_registry'):
            # Use service registry
            services = self.service_registry.get_services_by_type(service_type)
            if not services:
                return None
            
            # Convert to ServiceInfo for compatibility (simplified version)
            best_service = self.service_registry.find_best_service(service_type)
            if best_service:
                # Get the actual websocket connection for this service
                websocket_connection = self.service_connections.get(best_service.service_id)
                
                # Create a compatible ServiceInfo object with ALL required parameters
                import time
                return ServiceInfo(
                    service_id=best_service.service_id,
                    service_type=best_service.service_type,
                    endpoint=best_service.endpoint,
                    capabilities=best_service.capabilities.to_dict() if hasattr(best_service.capabilities, 'to_dict') else {},
                    websocket=websocket_connection,  # Now using actual websocket connection!
                    last_heartbeat=time.time(),  # Add required parameter
                    load_factor=0.0,  # Add required parameter
                    active_sessions=set()  # Add required parameter
                )
            return None
        else:
            # Use legacy mode
            if not hasattr(self, 'service_types'):
                return None
                
            available_services = [
                self.services[sid] for sid in self.service_types.get(service_type, set())
                if sid in self.services
            ]
            
            if not available_services:
                return None
            
            # Simple load balancing - choose service with lowest load
            return min(available_services, key=lambda s: s.load_factor)
    
    def extract_client_type(self, path: str) -> str:
        """Extract client type from WebSocket path"""
        if "/headset" in path:
            return "headset"
        elif "/twilio" in path:
            return "twilio"
        elif "/web" in path:
            return "web"
        else:
            return "unknown"
    
    async def send_error_to_client(self, session_id: str, error_msg: str):
        """Send error message to client"""
        if session_id in self.sessions:
            error = StreamingMessage(
                type=MessageType.ERROR.value,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                data={"error": error_msg}
            )
            session = self.sessions[session_id]
            await session.client_websocket.send(error.to_json())
    
    async def handle_client_heartbeat(self, session_id: str):
        """Handle client heartbeat"""
        if session_id in self.sessions:
            response = StreamingMessage(
                type=MessageType.HEARTBEAT.value,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                data={"status": "alive"}
            )
            session = self.sessions[session_id]
            await session.client_websocket.send(response.to_json())
    
    async def send_health_status_to_client(self, session_id: str):
        """Send health status to client"""
        if session_id in self.sessions:
            # Get service status
            available_services = []
            services_status = {}
            for service_type in ["stt", "llm", "tts"]:
                service = self.get_best_service(service_type, session_id)
                if service:
                    available_services.append(service_type)
                    services_status[service_type] = "available"
                else:
                    services_status[service_type] = "unavailable"
            
            response = StreamingMessage(
                type="health_status",
                session_id=session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                data={
                    "orchestrator_status": "healthy",
                    "services": available_services,  # List format for compatibility
                    "service_details": services_status,  # Detailed status
                    "registered_services": list(self.services.keys()) if hasattr(self, 'services') else []
                }
            )
            session = self.sessions[session_id]
            await session.client_websocket.send(response.to_json())
    
    async def cleanup_session(self, session_id: str):
        """Clean up session resources"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Remove session from assigned services
            for service_id in session.assigned_services.values():
                if service_id in self.services:
                    self.services[service_id].active_sessions.discard(session_id)
            
            del self.sessions[session_id]
            self.logger.info(f"Session cleaned up: {session_id}")
    
    async def heartbeat_monitor(self):
        """Monitor service heartbeats"""
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            current_time = time.time()
            
            if REGISTRY_AVAILABLE:
                # Use service registry for heartbeat monitoring
                unhealthy_services = self.service_registry.get_unhealthy_services(self.heartbeat_interval * 2)
                
                for service_id in unhealthy_services:
                    self.logger.warning(f"Service {service_id} appears stale, removing")
                    self.service_registry.unregister_service(service_id)
                    
                    # Also remove from service connections
                    if service_id in self.service_connections:
                        try:
                            await self.service_connections[service_id].close()
                        except:
                            pass
                        del self.service_connections[service_id]
            else:
                # Legacy heartbeat monitoring
                stale_services = [
                    service_id for service_id, service_info in self.services.items()
                    if current_time - service_info.last_heartbeat > self.heartbeat_interval * 2
                ]
                
                for service_id in stale_services:
                    self.logger.warning(f"Service {service_id} appears stale, removing")
                    await self.unregister_service(service_id)
    
    async def cleanup_monitor(self):
        """Monitor and cleanup stale sessions"""
        while True:
            await asyncio.sleep(60)  # Check every minute
            current_time = time.time()
            
            # Check for stale sessions
            stale_sessions = [
                session_id for session_id, session in self.sessions.items()
                if current_time - session.last_activity > self.session_timeout
            ]
            
            for session_id in stale_sessions:
                self.logger.warning(f"Session {session_id} timed out, cleaning up")
                await self.cleanup_session(session_id)
    
    # HTTP endpoints for monitoring
    async def health_check(self, request):
        """Health check endpoint"""
        if REGISTRY_AVAILABLE:
            service_count = len(self.service_registry.services)
            uptime = time.time() - getattr(self, 'start_time', time.time())
        else:
            service_count = len(self.services)
            uptime = time.time() - getattr(self, 'start_time', time.time())
            
        return web.json_response({
            "status": "healthy",
            "services": service_count,
            "active_sessions": len(self.sessions),
            "uptime": uptime,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def list_services(self, request):
        """List registered services"""
        if REGISTRY_AVAILABLE:
            # Use service registry
            services_data = []
            self.logger.info(f"Listing services, registry has {len(self.service_registry.services)} services")
            for service_id, service in self.service_registry.services.items():
                self.logger.info(f"Service in registry: {service_id}")
                last_health = self.service_registry.service_health.get(service_id, 0)
                healthy = time.time() - last_health < 60 if last_health > 0 else False
                
                service_data = service.to_dict()
                service_data['service_id'] = service_id
                service_data['healthy'] = healthy
                service_data['last_heartbeat'] = last_health
                services_data.append(service_data)
            
            return web.json_response(services_data)
        else:
            # Use legacy service info
            services_data = {}
            for service_id, service_info in self.services.items():
                services_data[service_id] = {
                    "service_type": service_info.service_type,
                    "endpoint": service_info.endpoint,
                    "capabilities": service_info.capabilities,
                    "active_sessions": len(service_info.active_sessions),
                    "load_factor": service_info.load_factor,
                    "last_heartbeat": service_info.last_heartbeat
                }
            
            return web.json_response(services_data)
    
    async def list_sessions(self, request):
        """List active sessions"""
        sessions_data = {}
        for session_id, session in self.sessions.items():
            sessions_data[session_id] = {
                "client_type": session.client_type,
                "assigned_services": session.assigned_services,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "metadata": session.metadata
            }
        
        return web.json_response(sessions_data)
    
    # Service Registry HTTP Endpoints (when registry is available)
    async def register_service_http(self, request):
        """HTTP endpoint for service registration"""
        if not REGISTRY_AVAILABLE:
            return web.json_response({"error": "Service registry not available"}, status=500)
        
        try:
            data = await request.json()
            registration = ServiceRegistration.from_dict(data)
            
            success = self.service_registry.register_service(registration)
            if success:
                return web.json_response({
                    "status": "registered", 
                    "service_id": registration.service_id,
                    "message": f"Service {registration.service_id} registered successfully"
                })
            else:
                return web.json_response({"error": "Failed to register service"}, status=400)
                
        except Exception as e:
            self.logger.error(f"Service registration error: {e}")
            return web.json_response({"error": str(e)}, status=400)
    
    async def unregister_service_http(self, request):
        """HTTP endpoint for service unregistration"""
        if not REGISTRY_AVAILABLE:
            return web.json_response({"error": "Service registry not available"}, status=500)
        
        service_id = request.match_info['service_id']
        
        try:
            success = self.service_registry.unregister_service(service_id)
            if success:
                return web.json_response({
                    "status": "unregistered",
                    "service_id": service_id,
                    "message": f"Service {service_id} unregistered successfully"
                })
            else:
                return web.json_response({"error": "Service not found"}, status=404)
                
        except Exception as e:
            self.logger.error(f"Service unregistration error: {e}")
            return web.json_response({"error": str(e)}, status=400)
    
    async def registry_status(self, request):
        """Get service registry status"""
        if not REGISTRY_AVAILABLE:
            return web.json_response({"error": "Service registry not available"}, status=500)
        
        try:
            status = self.service_registry.get_registry_status()
            return web.json_response(status)
        except Exception as e:
            self.logger.error(f"Registry status error: {e}")
            return web.json_response({"error": str(e)}, status=500)

async def main():
    """Main entry point"""
    # Use uvloop for better performance if available
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass  # Use default event loop
    
    orchestrator = WebSocketOrchestrator(
        host="0.0.0.0",  # Listen on all interfaces
        port=9000,       # Client connections
        service_port=9001 # Service connections
    )
    
    try:
        await orchestrator.start_server()
    except KeyboardInterrupt:
        logging.info("Shutting down WebSocket Orchestrator")

if __name__ == "__main__":
    asyncio.run(main())
