#!/usr/bin/env python3
"""
WebSocket Service Registration System

Handles service discovery, registration, and health monitoring for the streaming architecture.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import websockets

class ServiceType(Enum):
    STT = "stt"
    LLM = "llm" 
    TTS = "tts"
    ORCHESTRATOR = "orchestrator"

class ServiceCapability(Enum):
    # STT Capabilities
    REALTIME = "realtime"
    VAD = "voice_activity_detection"
    MULTILINGUAL = "multilingual"
    SPEAKER_IDENTIFICATION = "speaker_id"
    
    # LLM Capabilities  
    STREAMING = "streaming"
    CONTEXT_AWARE = "context_aware"
    FUNCTION_CALLING = "function_calling"
    MULTI_TURN = "multi_turn"
    
    # TTS Capabilities
    STREAMING_SYNTHESIS = "streaming_synthesis"
    VOICE_CLONING = "voice_cloning"
    EMOTION_CONTROL = "emotion_control"
    SSML = "ssml"

@dataclass
class ServiceCapabilities:
    """Service capability definition"""
    realtime: bool = False
    streaming: bool = False
    languages: List[str] = None
    voice_models: List[str] = None
    max_concurrent: int = 10
    latency_ms: int = 100
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en"]
        if self.voice_models is None:
            self.voice_models = []

@dataclass
class ServiceRegistration:
    """Service registration message"""
    service_id: str
    service_type: str
    service_name: str
    version: str
    endpoint: str
    websocket_port: int
    http_port: int
    capabilities: ServiceCapabilities
    metadata: Dict[str, any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ServiceRegistration':
        """Create from dictionary"""
        capabilities_data = data.get('capabilities', {})
        if isinstance(capabilities_data, dict):
            capabilities = ServiceCapabilities(**capabilities_data)
        else:
            capabilities = capabilities_data
            
        return cls(
            service_id=data['service_id'],
            service_type=data['service_type'],
            service_name=data['service_name'], 
            version=data['version'],
            endpoint=data['endpoint'],
            websocket_port=data['websocket_port'],
            http_port=data['http_port'],
            capabilities=capabilities,
            metadata=data.get('metadata', {})
        )

class ServiceRegistry:
    """Service registry for managing available services"""
    
    def __init__(self):
        self.services: Dict[str, ServiceRegistration] = {}
        self.services_by_type: Dict[str, Set[str]] = {}
        self.service_health: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
        
    def register_service(self, registration: ServiceRegistration) -> bool:
        """Register a new service"""
        service_id = registration.service_id
        service_type = registration.service_type
        
        # Add to main registry
        self.services[service_id] = registration
        
        # Add to type index
        if service_type not in self.services_by_type:
            self.services_by_type[service_type] = set()
        self.services_by_type[service_type].add(service_id)
        
        # Initialize health
        self.service_health[service_id] = time.time()
        
        self.logger.info(f"Registered service: {service_id} ({service_type})")
        return True
    
    def unregister_service(self, service_id: str) -> bool:
        """Unregister a service"""
        if service_id not in self.services:
            return False
        
        registration = self.services[service_id]
        service_type = registration.service_type
        
        # Remove from main registry
        del self.services[service_id]
        
        # Remove from type index
        if service_type in self.services_by_type:
            self.services_by_type[service_type].discard(service_id)
            if not self.services_by_type[service_type]:
                del self.services_by_type[service_type]
        
        # Remove health tracking
        if service_id in self.service_health:
            del self.service_health[service_id]
            
        self.logger.info(f"Unregistered service: {service_id}")
        return True
    
    def get_services_by_type(self, service_type: str) -> List[ServiceRegistration]:
        """Get all services of a specific type"""
        service_ids = self.services_by_type.get(service_type, set())
        return [self.services[sid] for sid in service_ids if sid in self.services]
    
    def get_service(self, service_id: str) -> Optional[ServiceRegistration]:
        """Get service by ID"""
        return self.services.get(service_id)
    
    def find_best_service(self, service_type: str, requirements: Dict = None) -> Optional[ServiceRegistration]:
        """Find the best service of a type based on requirements"""
        services = self.get_services_by_type(service_type)
        
        if not services:
            return None
        
        if not requirements:
            # Return first available service
            return services[0]
        
        # Score services based on requirements
        scored_services = []
        for service in services:
            score = self._score_service(service, requirements)
            scored_services.append((score, service))
        
        # Return highest scored service
        scored_services.sort(reverse=True)
        return scored_services[0][1] if scored_services else None
    
    def _score_service(self, service: ServiceRegistration, requirements: Dict) -> float:
        """Score a service based on requirements"""
        score = 100.0  # Base score
        caps = service.capabilities
        
        # Check realtime requirement
        if requirements.get('realtime', False) and not caps.realtime:
            score -= 50
        
        # Check streaming requirement
        if requirements.get('streaming', False) and not caps.streaming:
            score -= 30
            
        # Check language requirements
        required_langs = requirements.get('languages', [])
        if required_langs:
            supported = set(caps.languages)
            required = set(required_langs)
            if not required.issubset(supported):
                score -= 20
        
        # Prefer lower latency
        latency_penalty = caps.latency_ms / 10.0
        score -= latency_penalty
        
        # Consider load (if available)
        # This would be updated by the orchestrator based on current usage
        
        return max(0, score)
    
    def update_service_health(self, service_id: str):
        """Update service health timestamp"""
        if service_id in self.services:
            self.service_health[service_id] = time.time()
    
    def get_unhealthy_services(self, timeout_seconds: int = 60) -> List[str]:
        """Get list of unhealthy service IDs"""
        current_time = time.time()
        unhealthy = []
        
        for service_id, last_health in self.service_health.items():
            if current_time - last_health > timeout_seconds:
                unhealthy.append(service_id)
        
        return unhealthy
    
    def get_registry_status(self) -> Dict:
        """Get registry status summary"""
        return {
            'total_services': len(self.services),
            'services_by_type': {
                stype: len(sids) for stype, sids in self.services_by_type.items()
            },
            'healthy_services': len([
                sid for sid in self.service_health.keys()
                if time.time() - self.service_health[sid] < 60
            ])
        }

class WebSocketServiceClient:
    """Base class for WebSocket service clients"""
    
    def __init__(self, service_registration: ServiceRegistration, orchestrator_host: str = "localhost", orchestrator_port: int = 9001):
        self.registration = service_registration
        self.orchestrator_host = orchestrator_host
        self.orchestrator_port = orchestrator_port
        self.websocket = None
        self.connected = False
        self.logger = logging.getLogger(f"{__name__}.{service_registration.service_id}")
        
    async def connect_to_orchestrator(self):
        """Connect to orchestrator and register service"""
        uri = f"ws://{self.orchestrator_host}:{self.orchestrator_port}/service"
        
        try:
            self.websocket = await websockets.connect(uri)
            self.connected = True
            
            # Send registration message
            await self.register()
            
            self.logger.info(f"Connected to orchestrator: {uri}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to orchestrator: {e}")
            raise
    
    async def register(self):
        """Register this service with the orchestrator"""
        registration_msg = {
            "type": "service_register",
            "session_id": self.registration.service_id,
            "timestamp": time.time(),
            "data": self.registration.to_dict()
        }
        
        await self.websocket.send(json.dumps(registration_msg))
        self.logger.info(f"Sent registration for {self.registration.service_id}")
    
    async def send_heartbeat(self):
        """Send heartbeat to orchestrator"""
        if self.connected and self.websocket:
            heartbeat_msg = {
                "type": "heartbeat",
                "session_id": self.registration.service_id,
                "timestamp": time.time(),
                "data": {"status": "alive"}
            }
            
            try:
                await self.websocket.send(json.dumps(heartbeat_msg))
            except Exception as e:
                self.logger.warning(f"Failed to send heartbeat: {e}")
                self.connected = False
    
    async def listen_for_messages(self):
        """Listen for messages from orchestrator"""
        try:
            async for message in self.websocket:
                await self.handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Connection to orchestrator closed")
            self.connected = False
        except Exception as e:
            self.logger.error(f"Error listening for messages: {e}")
            self.connected = False
    
    async def handle_message(self, message: str):
        """Handle message from orchestrator - override in subclasses"""
        try:
            msg_data = json.loads(message)
            msg_type = msg_data.get('type')
            
            if msg_type == 'registration_confirmed':
                self.logger.info("Service registration confirmed")
            else:
                # Process service-specific messages
                await self.process_message(msg_data)
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def process_message(self, msg_data: Dict):
        """Process service-specific messages - override in subclasses"""
        pass
    
    async def send_message(self, msg_type: str, session_id: str, data: any, metadata: Dict = None):
        """Send message to orchestrator"""
        if not self.connected or not self.websocket:
            self.logger.warning("Not connected to orchestrator")
            return
        
        message = {
            "type": msg_type,
            "session_id": session_id,
            "timestamp": time.time(),
            "data": data,
            "metadata": metadata or {}
        }
        
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            self.connected = False
    
    async def start_heartbeat_task(self):
        """Start heartbeat task"""
        while self.connected:
            await self.send_heartbeat()
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
    
    async def disconnect(self):
        """Disconnect from orchestrator"""
        if self.websocket:
            await self.websocket.close()
        self.connected = False
        self.logger.info("Disconnected from orchestrator")

# Message protocol constants
MESSAGE_TYPES = {
    # Client to Service
    'AUDIO_CHUNK': 'audio_chunk',
    'TEXT_INPUT': 'text_input',
    
    # Service to Service/Client
    'TRANSCRIPT_PARTIAL': 'transcript_partial',
    'TRANSCRIPT_FINAL': 'transcript_final', 
    'RESPONSE_TOKEN': 'response_token',
    'RESPONSE_FINAL': 'response_final',
    'AUDIO_OUTPUT': 'audio_output',
    
    # Control Messages
    'SERVICE_REGISTER': 'service_register',
    'SERVICE_UNREGISTER': 'service_unregister',
    'SESSION_START': 'session_start',
    'SESSION_END': 'session_end',
    'HEARTBEAT': 'heartbeat',
    'ERROR': 'error',
    'CONTEXT_UPDATE': 'context_update'
}

# Standard message structure validation
def validate_message(message: Dict) -> bool:
    """Validate message structure"""
    required_fields = ['type', 'session_id', 'timestamp', 'data']
    return all(field in message for field in required_fields)

def create_message(msg_type: str, session_id: str, data: any, metadata: Dict = None) -> Dict:
    """Create standard message structure"""
    return {
        'type': msg_type,
        'session_id': session_id,
        'timestamp': time.time(),
        'data': data,
        'metadata': metadata or {}
    }
