"""
Tests for session management functionality.
"""
import asyncio
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebot_orchestrator.session_manager import SessionManager, SessionState


class TestSessionManager:
    """Test cases for SessionManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.session_manager = SessionManager(timeout=5, max_sessions=3)
    
    async def test_create_session(self):
        """Test session creation."""
        session_id = "test-session-1"
        session = await self.session_manager.create_session(session_id)
        
        assert session.session_id == session_id
        assert session.state == SessionState.ACTIVE
        assert session.created_at > 0
        assert session.last_activity > 0
        assert len(session.conversation_history) == 0
    
    async def test_create_duplicate_session(self):
        """Test creating duplicate session raises error."""
        session_id = "test-session-2"
        await self.session_manager.create_session(session_id)
        
        try:
            await self.session_manager.create_session(session_id)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "already exists" in str(e)
    
    async def test_get_session(self):
        """Test getting existing session."""
        session_id = "test-session-3"
        created_session = await self.session_manager.create_session(session_id)
        
        retrieved_session = await self.session_manager.get_session(session_id)
        
        assert retrieved_session is not None
        assert retrieved_session.session_id == created_session.session_id
        assert retrieved_session.state == SessionState.ACTIVE
    
    async def test_get_nonexistent_session(self):
        """Test getting non-existent session returns None."""
        session = await self.session_manager.get_session("nonexistent")
        assert session is None
    
    async def test_update_activity(self):
        """Test updating session activity."""
        session_id = "test-session-4"
        session = await self.session_manager.create_session(session_id)
        initial_activity = session.last_activity
        
        # Wait a bit and update activity
        await asyncio.sleep(0.1)
        success = await self.session_manager.update_activity(session_id)
        
        assert success is True
        
        updated_session = await self.session_manager.get_session(session_id)
        assert updated_session.last_activity > initial_activity
    
    async def test_add_to_history(self):
        """Test adding conversation to history."""
        session_id = "test-session-5"
        await self.session_manager.create_session(session_id)
        
        user_input = "Hello"
        bot_response = "Hi there!"
        
        success = await self.session_manager.add_to_history(session_id, user_input, bot_response)
        assert success is True
        
        session = await self.session_manager.get_session(session_id)
        assert len(session.conversation_history) == 1
        
        history_entry = session.conversation_history[0]
        assert history_entry["user_input"] == user_input
        assert history_entry["bot_response"] == bot_response
        assert "timestamp" in history_entry
    
    async def test_end_session(self):
        """Test ending a session."""
        session_id = "test-session-6"
        await self.session_manager.create_session(session_id)
        
        success = await self.session_manager.end_session(session_id)
        assert success is True
        
        # Session should no longer exist
        session = await self.session_manager.get_session(session_id)
        assert session is None
    
    async def test_list_active_sessions(self):
        """Test listing active sessions."""
        session_ids = ["test-session-7", "test-session-8", "test-session-9"]
        
        for session_id in session_ids:
            await self.session_manager.create_session(session_id)
        
        active_sessions = await self.session_manager.list_active_sessions()
        
        assert len(active_sessions) == len(session_ids)
        for session_id in session_ids:
            assert session_id in active_sessions
    
    async def test_max_sessions_limit(self):
        """Test maximum sessions limit."""
        # Create maximum number of sessions
        for i in range(3):
            await self.session_manager.create_session(f"test-session-max-{i}")
        
        # Try to create one more - should fail
        try:
            await self.session_manager.create_session("test-session-max-overflow")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Maximum number of sessions exceeded" in str(e)
    
    async def test_session_timeout(self):
        """Test session timeout functionality."""
        # Create session manager with very short timeout
        short_timeout_manager = SessionManager(timeout=0.1, max_sessions=5)
        
        session_id = "test-session-timeout"
        await short_timeout_manager.create_session(session_id)
        
        # Wait for timeout
        await asyncio.sleep(0.2)
        
        # Session should be expired
        session = await short_timeout_manager.get_session(session_id)
        assert session is None


# Test runner for pytest compatibility
async def run_tests():
    """Run all tests."""
    test_class = TestSessionManager()
    
    test_methods = [
        test_class.test_create_session,
        test_class.test_create_duplicate_session,
        test_class.test_get_session,
        test_class.test_get_nonexistent_session,
        test_class.test_update_activity,
        test_class.test_add_to_history,
        test_class.test_end_session,
        test_class.test_list_active_sessions,
        test_class.test_max_sessions_limit,
        test_class.test_session_timeout,
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        test_class.setup_method()
        try:
            await test_method()
            print(f"✓ {test_method.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_method.__name__}: {e}")
            failed += 1
    
    print(f"\nSession Manager Tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
