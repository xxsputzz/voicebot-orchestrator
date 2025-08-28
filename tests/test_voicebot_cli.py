"""
Tests for voicebot CLI functionality - Sprint 2
"""
import asyncio
import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebot_orchestrator.voicebot_cli import (
    VoicebotCLI, StructuredLogger, SessionStore, CacheManager, 
    AdapterController, PipelineEvent, EventType, LogLevel
)


class TestStructuredLogger:
    """Test cases for StructuredLogger."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_log = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
        self.logger = StructuredLogger(self.temp_log.name)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_log.name):
            os.unlink(self.temp_log.name)
    
    def test_log_event(self):
        """Test logging pipeline events."""
        event = PipelineEvent(
            timestamp=1234567890.0,
            session_id="test-session",
            event_type=EventType.STT_START,
            component="whisper",
            data={"input_size": 1024},
            duration_ms=50.0
        )
        
        self.logger.log_event(event)
        
        # Verify event was stored
        assert len(self.logger.events) == 1
        assert self.logger.events[0].session_id == "test-session"
        assert self.logger.events[0].event_type == EventType.STT_START
        
        # Verify log file was written
        assert os.path.exists(self.temp_log.name)
        with open(self.temp_log.name, "r") as f:
            log_content = f.read()
            assert "test-session" in log_content
            assert "stt_start" in log_content
    
    def test_get_session_events(self):
        """Test filtering events by session ID."""
        event1 = PipelineEvent(
            timestamp=1234567890.0,
            session_id="session-1",
            event_type=EventType.STT_START,
            component="whisper",
            data={}
        )
        
        event2 = PipelineEvent(
            timestamp=1234567891.0,
            session_id="session-2",
            event_type=EventType.LLM_START,
            component="mistral",
            data={}
        )
        
        event3 = PipelineEvent(
            timestamp=1234567892.0,
            session_id="session-1",
            event_type=EventType.TTS_START,
            component="kokoro",
            data={}
        )
        
        for event in [event1, event2, event3]:
            self.logger.log_event(event)
        
        session1_events = self.logger.get_session_events("session-1")
        assert len(session1_events) == 2
        assert all(e.session_id == "session-1" for e in session1_events)
        
        session2_events = self.logger.get_session_events("session-2")
        assert len(session2_events) == 1
        assert session2_events[0].session_id == "session-2"
    
    def test_get_events_by_type(self):
        """Test filtering events by type."""
        events = [
            PipelineEvent(1234567890.0, "s1", EventType.STT_START, "whisper", {}),
            PipelineEvent(1234567891.0, "s2", EventType.STT_START, "whisper", {}),
            PipelineEvent(1234567892.0, "s1", EventType.LLM_START, "mistral", {}),
        ]
        
        for event in events:
            self.logger.log_event(event)
        
        stt_events = self.logger.get_events_by_type(EventType.STT_START)
        assert len(stt_events) == 2
        assert all(e.event_type == EventType.STT_START for e in stt_events)


class TestSessionStore:
    """Test cases for SessionStore."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = SessionStore(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_session(self):
        """Test session persistence."""
        session_id = "test-session-123"
        session_state = {
            "created_at": 1234567890.0,
            "last_activity": 1234567900.0,
            "conversation_history": [
                {"user_input": "Hello", "bot_response": "Hi there!"}
            ]
        }
        
        # Save session
        self.store.save_session(session_id, session_state)
        
        # Verify in memory
        assert session_id in self.store.memory_store
        assert self.store.memory_store[session_id] == session_state
        
        # Verify on disk
        session_file = Path(self.temp_dir) / f"{session_id}.json"
        assert session_file.exists()
        
        # Load session
        loaded_state = self.store.load_session(session_id)
        assert loaded_state == session_state
    
    def test_list_sessions(self):
        """Test listing available sessions."""
        session_ids = ["session-1", "session-2", "session-3"]
        
        for session_id in session_ids:
            self.store.save_session(session_id, {"test": "data"})
        
        listed_sessions = self.store.list_sessions()
        assert set(listed_sessions) == set(session_ids)
    
    def test_delete_session(self):
        """Test session deletion."""
        session_id = "delete-test"
        self.store.save_session(session_id, {"test": "data"})
        
        # Verify session exists
        assert self.store.load_session(session_id) is not None
        
        # Delete session
        success = self.store.delete_session(session_id)
        assert success is True
        
        # Verify session is gone
        assert self.store.load_session(session_id) is None
        assert session_id not in self.store.memory_store
        
        session_file = Path(self.temp_dir) / f"{session_id}.json"
        assert not session_file.exists()
    
    def test_get_session_stats(self):
        """Test session statistics calculation."""
        session_id = "stats-test"
        session_state = {
            "created_at": 1234567890.0,
            "last_activity": 1234567950.0,
            "conversation_history": [
                {"user_input": "Hello world", "bot_response": "Hi there friend"},
                {"user_input": "How are you", "bot_response": "I am doing well today"},
            ]
        }
        
        self.store.save_session(session_id, session_state)
        stats = self.store.get_session_stats(session_id)
        
        assert stats is not None
        assert stats["session_id"] == session_id
        assert stats["message_count"] == 2
        assert stats["total_user_words"] == 5  # "Hello world" + "How are you"
        assert stats["total_bot_words"] == 8   # "Hi there friend" + "I am doing well today"
        assert stats["session_duration"] == 60.0  # 1234567950 - 1234567890


class TestCacheManager:
    """Test cases for CacheManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_cache = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.cache_manager = CacheManager(self.temp_cache.name)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_cache.name):
            os.unlink(self.temp_cache.name)
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        # Test set and get
        self.cache_manager.set_cache_entry("test_key", "test_value")
        value = self.cache_manager.get_cache_entry("test_key")
        assert value == "test_value"
        
        # Test non-existent key
        assert self.cache_manager.get_cache_entry("non_existent") is None
        
        # Test cache persistence
        cache_manager2 = CacheManager(self.temp_cache.name)
        value2 = cache_manager2.get_cache_entry("test_key")
        assert value2 == "test_value"
    
    def test_clear_cache(self):
        """Test cache clearing."""
        self.cache_manager.set_cache_entry("key1", "value1")
        self.cache_manager.set_cache_entry("key2", "value2")
        
        assert len(self.cache_manager.cache) == 2
        
        self.cache_manager.clear_cache()
        assert len(self.cache_manager.cache) == 0
        assert self.cache_manager.get_cache_entry("key1") is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        self.cache_manager.set_cache_entry("test", "data")
        stats = self.cache_manager.get_cache_stats()
        
        assert stats["total_entries"] == 1
        assert stats["cache_size_bytes"] > 0
        assert "cache_file" in stats


class TestAdapterController:
    """Test cases for AdapterController."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.controller = AdapterController(self.temp_dir)
        
        # Create test adapter files
        for i in range(3):
            adapter_file = Path(self.temp_dir) / f"adapter_{i}.json"
            adapter_file.write_text('{"name": "test_adapter"}')
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_list_adapters(self):
        """Test listing available adapters."""
        adapters = self.controller.list_adapters()
        expected = {"adapter_0", "adapter_1", "adapter_2"}
        assert set(adapters) == expected
    
    def test_load_unload_adapter(self):
        """Test loading and unloading adapters."""
        # Test loading existing adapter
        success = self.controller.load_adapter("adapter_0")
        assert success is True
        assert "adapter_0" in self.controller.active_adapters
        
        # Test loading non-existent adapter
        success = self.controller.load_adapter("non_existent")
        assert success is False
        
        # Test unloading adapter
        success = self.controller.unload_adapter("adapter_0")
        assert success is True
        assert "adapter_0" not in self.controller.active_adapters
        
        # Test unloading non-loaded adapter
        success = self.controller.unload_adapter("adapter_1")
        assert success is False
    
    def test_adapter_status(self):
        """Test adapter status reporting."""
        self.controller.load_adapter("adapter_0")
        self.controller.load_adapter("adapter_1")
        
        status = self.controller.get_adapter_status()
        
        assert len(status["active_adapters"]) == 2
        assert "adapter_0" in status["active_adapters"]
        assert "adapter_1" in status["active_adapters"]
        assert len(status["available_adapters"]) == 3


class TestVoicebotCLI:
    """Test cases for VoicebotCLI main functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cli = VoicebotCLI()
        
        # Mock session data
        test_session_state = {
            "created_at": 1234567890.0,
            "last_activity": 1234567950.0,
            "conversation_history": [
                {"user_input": "Hello", "bot_response": "Hi there!", "timestamp": 1234567891.0},
                {"user_input": "How are you?", "bot_response": "I'm doing well!", "timestamp": 1234567920.0}
            ]
        }
        self.cli.session_store.save_session("test-session", test_session_state)
    
    def test_cache_manager_operations(self):
        """Test cache manager CLI operations."""
        # Test stats
        try:
            self.cli.cache_manager_cmd("stats")
        except Exception as e:
            assert False, f"Cache stats failed: {e}"
        
        # Test set and get
        self.cli.cache_manager_cmd("set", "test_key", "test_value")
        
        # Verify by getting
        value = self.cli.cache_manager.get_cache_entry("test_key")
        assert value == "test_value"
        
        # Test clear
        self.cli.cache_manager_cmd("clear")
        value = self.cli.cache_manager.get_cache_entry("test_key")
        assert value is None
    
    def test_adapter_control_operations(self):
        """Test adapter control CLI operations."""
        # Test status
        try:
            self.cli.adapter_control("status")
        except Exception as e:
            assert False, f"Adapter status failed: {e}"
        
        # Test list
        try:
            self.cli.adapter_control("list")
        except Exception as e:
            assert False, f"Adapter list failed: {e}"
    
    def test_session_diagnostics(self):
        """Test session diagnostics functionality."""
        # Test single session diagnostics
        try:
            self.cli.session_diagnostics("test-session")
        except Exception as e:
            assert False, f"Single session diagnostics failed: {e}"
        
        # Test all sessions diagnostics
        try:
            self.cli.session_diagnostics()
        except Exception as e:
            assert False, f"All sessions diagnostics failed: {e}"
    
    def test_invalid_operations(self):
        """Test error handling for invalid operations."""
        # Test invalid cache action
        try:
            self.cli.cache_manager_cmd("invalid_action")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Test invalid adapter action
        try:
            self.cli.adapter_control("invalid_action")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Test non-existent session monitoring
        try:
            self.cli.monitor_session("non-existent-session")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected


# Test runner for pytest compatibility
def run_tests():
    """Run all CLI tests."""
    test_classes = [
        TestStructuredLogger,
        TestSessionStore,
        TestCacheManager,
        TestAdapterController,
        TestVoicebotCLI,
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"\n🔄 Running {class_name} tests...")
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        class_passed = 0
        class_failed = 0
        
        for method_name in test_methods:
            try:
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                test_method = getattr(test_instance, method_name)
                test_method()
                
                print(f"✓ {method_name}")
                class_passed += 1
                
            except Exception as e:
                print(f"✗ {method_name}: {e}")
                class_failed += 1
            
            finally:
                if hasattr(test_instance, 'teardown_method'):
                    try:
                        test_instance.teardown_method()
                    except Exception:
                        pass  # Ignore cleanup errors
        
        print(f"{class_name}: {class_passed} passed, {class_failed} failed")
        total_passed += class_passed
        total_failed += class_failed
    
    print(f"\n📊 Total CLI Tests: {total_passed} passed, {total_failed} failed")
    return total_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
