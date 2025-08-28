"""
Session management for voicebot orchestrator.
"""
import asyncio
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class SessionState(Enum):
    """Session state enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"


@dataclass
class SessionData:
    """Session data container."""
    session_id: str
    state: SessionState = SessionState.ACTIVE
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    conversation_history: list = field(default_factory=list)


class SessionManager:
    """Manages voicebot session state and lifecycle."""
    
    def __init__(self, timeout: int = 3600, max_sessions: int = 10) -> None:
        """
        Initialize session manager.
        
        Args:
            timeout: Session timeout in seconds
            max_sessions: Maximum concurrent sessions
        """
        self.timeout = timeout
        self.max_sessions = max_sessions
        self._sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()
    
    async def create_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> SessionData:
        """
        Create a new session.
        
        Args:
            session_id: Unique session identifier
            metadata: Optional session metadata
            
        Returns:
            Created session data
            
        Raises:
            ValueError: If session already exists or max sessions exceeded
        """
        async with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session {session_id} already exists")
            
            if len(self._sessions) >= self.max_sessions:
                # Clean up expired sessions first
                await self._cleanup_expired_sessions()
                
                if len(self._sessions) >= self.max_sessions:
                    raise ValueError("Maximum number of sessions exceeded")
            
            session_data = SessionData(
                session_id=session_id,
                metadata=metadata or {}
            )
            self._sessions[session_id] = session_data
            return session_data
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data if exists and not expired, None otherwise
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            
            if self._is_expired(session):
                session.state = SessionState.EXPIRED
                return None
            
            return session
    
    async def update_activity(self, session_id: str) -> bool:
        """
        Update session last activity timestamp.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists and was updated, False otherwise
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session or self._is_expired(session):
                return False
            
            session.last_activity = time.time()
            return True
    
    async def add_to_history(self, session_id: str, user_input: str, bot_response: str) -> bool:
        """
        Add conversation exchange to session history.
        
        Args:
            session_id: Session identifier
            user_input: User's input text
            bot_response: Bot's response text
            
        Returns:
            True if added successfully, False if session not found
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session.conversation_history.append({
            "timestamp": time.time(),
            "user_input": user_input,
            "bot_response": bot_response
        })
        
        # Keep only last 50 exchanges to manage memory
        if len(session.conversation_history) > 50:
            session.conversation_history = session.conversation_history[-50:]
        
        await self.update_activity(session_id)
        return True
    
    async def end_session(self, session_id: str) -> bool:
        """
        End a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session existed and was ended, False otherwise
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            session.state = SessionState.INACTIVE
            del self._sessions[session_id]
            return True
    
    async def list_active_sessions(self) -> list[str]:
        """
        Get list of active session IDs.
        
        Returns:
            List of active session IDs
        """
        await self._cleanup_expired_sessions()
        return [
            session_id for session_id, session in self._sessions.items()
            if session.state == SessionState.ACTIVE
        ]
    
    def _is_expired(self, session: SessionData) -> bool:
        """Check if session is expired."""
        return time.time() - session.last_activity > self.timeout
    
    async def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions."""
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, session in self._sessions.items()
            if current_time - session.last_activity > self.timeout
        ]
        
        for session_id in expired_sessions:
            del self._sessions[session_id]
