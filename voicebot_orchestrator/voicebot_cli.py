#!/usr/bin/env python3
"""
Advanced CLI interface for voicebot orchestration - Sprint 2
"""
from typing import Optional, Dict, List, Any, Union
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebot_orchestrator.session_manager import SessionManager, SessionState
from voicebot_orchestrator.config import settings


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class EventType(str, Enum):
    """Pipeline event types."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    STT_START = "stt_start"
    STT_COMPLETE = "stt_complete"
    LLM_START = "llm_start"
    LLM_COMPLETE = "llm_complete"
    TTS_START = "tts_start"
    TTS_COMPLETE = "tts_complete"
    ERROR = "error"


@dataclass
class PipelineEvent:
    """Pipeline event data structure."""
    timestamp: float
    session_id: str
    event_type: EventType
    component: str
    data: Dict[str, Any]
    duration_ms: Optional[float] = None
    error: Optional[str] = None


class StructuredLogger:
    """Structured logging for pipeline events."""
    
    def __init__(self, log_file: str = "orchestrator.log"):
        self.log_file = log_file
        self.events: List[PipelineEvent] = []
        self._ensure_log_directory()
    
    def _ensure_log_directory(self) -> None:
        """Ensure log directory exists."""
        log_path = Path(self.log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)
    
    def log_event(self, event: PipelineEvent) -> None:
        """Log a pipeline event."""
        self.events.append(event)
        
        log_entry = {
            "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(),
            "session_id": event.session_id,
            "event_type": event.event_type.value,
            "component": event.component,
            "data": event.data,
            "duration_ms": event.duration_ms,
            "error": event.error
        }
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def get_session_events(self, session_id: str) -> List[PipelineEvent]:
        """Get all events for a specific session."""
        return [event for event in self.events if event.session_id == session_id]
    
    def get_events_by_type(self, event_type: EventType) -> List[PipelineEvent]:
        """Get all events of a specific type."""
        return [event for event in self.events if event.event_type == event_type]


class SessionStore:
    """Enhanced session state persistence."""
    
    def __init__(self, storage_dir: str = "./sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.memory_store: Dict[str, dict] = {}
        self.session_manager = SessionManager()
    
    def save_session(self, session_id: str, state: dict) -> None:
        """Persist session state to disk as JSON."""
        self.memory_store[session_id] = state
        
        session_file = self.storage_dir / f"{session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_session(self, session_id: str) -> Optional[dict]:
        """Load session state from disk."""
        # Check memory first
        if session_id in self.memory_store:
            return self.memory_store[session_id]
        
        # Load from disk
        session_file = self.storage_dir / f"{session_id}.json"
        if session_file.exists():
            with open(session_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                self.memory_store[session_id] = state
                return state
        
        return None
    
    def list_sessions(self) -> List[str]:
        """List all available sessions."""
        session_files = list(self.storage_dir.glob("*.json"))
        return [f.stem for f in session_files]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session from memory and disk."""
        # Remove from memory
        if session_id in self.memory_store:
            del self.memory_store[session_id]
        
        # Remove from disk
        session_file = self.storage_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            return True
        
        return False
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session statistics."""
        state = self.load_session(session_id)
        if not state:
            return None
        
        conversation_history = state.get("conversation_history", [])
        return {
            "session_id": session_id,
            "created_at": state.get("created_at"),
            "last_activity": state.get("last_activity"),
            "message_count": len(conversation_history),
            "total_user_words": sum(len(msg.get("user_input", "").split()) for msg in conversation_history),
            "total_bot_words": sum(len(msg.get("bot_response", "").split()) for msg in conversation_history),
            "session_duration": state.get("last_activity", 0) - state.get("created_at", 0)
        }


class CacheManager:
    """Semantic cache management."""
    
    def __init__(self, cache_file: str = "./cache/semantic_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Any] = {}
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        self.cache = json.loads(content)
                    else:
                        self.cache = {}
            except (json.JSONDecodeError, IOError):
                self.cache = {}
        else:
            self.cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)
    
    def get_cache_entry(self, key: str) -> Optional[Any]:
        """Get cache entry by key."""
        return self.cache.get(key)
    
    def set_cache_entry(self, key: str, value: Any) -> None:
        """Set cache entry."""
        self.cache[key] = value
        self._save_cache()
    
    def clear_cache(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self._save_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self.cache),
            "cache_size_bytes": len(json.dumps(self.cache)),
            "cache_file": str(self.cache_file)
        }


class AdapterController:
    """LoRA adapter management."""
    
    def __init__(self, adapter_dir: str = "./adapters"):
        self.adapter_dir = Path(adapter_dir)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.active_adapters: Dict[str, str] = {}
    
    def list_adapters(self) -> List[str]:
        """List available adapters."""
        adapter_files = list(self.adapter_dir.glob("*.json"))
        return [f.stem for f in adapter_files]
    
    def load_adapter(self, adapter_name: str) -> bool:
        """Load an adapter configuration."""
        adapter_file = self.adapter_dir / f"{adapter_name}.json"
        if adapter_file.exists():
            self.active_adapters[adapter_name] = str(adapter_file)
            return True
        return False
    
    def unload_adapter(self, adapter_name: str) -> bool:
        """Unload an adapter."""
        if adapter_name in self.active_adapters:
            del self.active_adapters[adapter_name]
            return True
        return False
    
    def get_adapter_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            "active_adapters": list(self.active_adapters.keys()),
            "available_adapters": self.list_adapters(),
            "adapter_directory": str(self.adapter_dir)
        }


class VoicebotCLI:
    """Main CLI application."""
    
    def __init__(self):
        self.logger = StructuredLogger()
        self.session_store = SessionStore()
        self.cache_manager = CacheManager()
        self.adapter_controller = AdapterController()
    
    def monitor_session(self, session_id: str, tail: bool = False, follow: bool = False) -> None:
        """
        Live-tail state transitions for the given session.
        
        Args:
            session_id: Session ID to monitor
            tail: Show last N events
            follow: Continuously monitor for new events
        """
        session_state = self.session_store.load_session(session_id)
        if not session_state:
            raise ValueError(f"Session {session_id!r} not found")
        
        print(f"📊 Monitoring session: {session_id}")
        print("-" * 50)
        
        # Show session stats
        stats = self.session_store.get_session_stats(session_id)
        if stats:
            print(f"Created: {datetime.fromtimestamp(stats['created_at']).isoformat()}")
            print(f"Messages: {stats['message_count']}")
            print(f"Duration: {stats['session_duration']:.2f}s")
            print()
        
        # Show recent events
        events = self.logger.get_session_events(session_id)
        if tail and events:
            events = events[-10:]  # Last 10 events
        
        for event in events:
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            duration = f" ({event.duration_ms:.1f}ms)" if event.duration_ms else ""
            error = f" ERROR: {event.error}" if event.error else ""
            print(f"[{timestamp}] {event.component} - {event.event_type.value}{duration}{error}")
        
        if follow:
            print("\n🔄 Following new events (Ctrl+C to stop)...")
            try:
                while True:
                    time.sleep(1)
                    # In real implementation, would check for new events
                    pass
            except KeyboardInterrupt:
                print("\n👋 Stopped monitoring")
    
    def orchestrator_log(self, level: LogLevel = LogLevel.INFO, lines: int = 50, follow: bool = False) -> None:
        """
        View orchestrator logs with filtering.
        
        Args:
            level: Minimum log level to show
            lines: Number of lines to show
            follow: Continuously monitor for new log entries
        """
        print(f"📋 Orchestrator logs (level: {level.value}, last {lines} lines)")
        print("-" * 60)
        
        log_file = Path(self.logger.log_file)
        if not log_file.exists():
            print("No log file found")
            return
        
        # Read and filter logs
        with open(log_file, "r", encoding="utf-8") as f:
            log_lines = f.readlines()
        
        # Show last N lines
        if lines and len(log_lines) > lines:
            log_lines = log_lines[-lines:]
        
        for line in log_lines:
            try:
                log_entry = json.loads(line.strip())
                timestamp = log_entry.get("timestamp", "")
                event_type = log_entry.get("event_type", "")
                component = log_entry.get("component", "")
                session_id = log_entry.get("session_id", "")[:8]
                
                print(f"[{timestamp}] {session_id} {component} - {event_type}")
                
                if log_entry.get("error"):
                    print(f"  ERROR: {log_entry['error']}")
                
            except json.JSONDecodeError:
                print(line.strip())
        
        if follow:
            print("\n🔄 Following new logs (Ctrl+C to stop)...")
            try:
                while True:
                    time.sleep(1)
                    # In real implementation, would tail log file
                    pass
            except KeyboardInterrupt:
                print("\n👋 Stopped following logs")
    
    def replay_session(self, session_id: str, step_by_step: bool = False, output_format: str = "text") -> None:
        """
        Replay a session's conversation history.
        
        Args:
            session_id: Session ID to replay
            step_by_step: Pause between each exchange
            output_format: Output format (text, json)
        """
        session_state = self.session_store.load_session(session_id)
        if not session_state:
            raise ValueError(f"Session {session_id!r} not found")
        
        conversation_history = session_state.get("conversation_history", [])
        
        print(f"🎬 Replaying session: {session_id}")
        print(f"📝 {len(conversation_history)} exchanges found")
        print("-" * 50)
        
        if output_format == "json":
            print(json.dumps(conversation_history, indent=2))
            return
        
        for i, exchange in enumerate(conversation_history, 1):
            timestamp = exchange.get("timestamp", time.time())
            user_input = exchange.get("user_input", "")
            bot_response = exchange.get("bot_response", "")
            
            print(f"\n[Exchange {i}] {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")
            print(f"👤 User: {user_input}")
            print(f"🤖 Bot:  {bot_response}")
            
            if step_by_step and i < len(conversation_history):
                input("Press Enter to continue...")
        
        print(f"\n✅ Replay complete - {len(conversation_history)} exchanges")
    
    def cache_manager_cmd(self, action: str, key: Optional[str] = None, value: Optional[str] = None) -> None:
        """
        Manage semantic cache.
        
        Args:
            action: Action to perform (get, set, clear, stats)
            key: Cache key for get/set operations
            value: Cache value for set operation
        """
        if action == "stats":
            stats = self.cache_manager.get_cache_stats()
            print("📊 Cache Statistics:")
            print(f"  Total entries: {stats['total_entries']}")
            print(f"  Cache size: {stats['cache_size_bytes']} bytes")
            print(f"  Cache file: {stats['cache_file']}")
        
        elif action == "clear":
            self.cache_manager.clear_cache()
            print("🗑️  Cache cleared")
        
        elif action == "get":
            if not key:
                raise ValueError("Key required for get operation")
            
            value = self.cache_manager.get_cache_entry(key)
            if value is not None:
                print(f"📦 Cache entry '{key}': {value}")
            else:
                print(f"❌ Cache entry '{key}' not found")
        
        elif action == "set":
            if not key or value is None:
                raise ValueError("Key and value required for set operation")
            
            self.cache_manager.set_cache_entry(key, value)
            print(f"✅ Cache entry '{key}' set")
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def adapter_control(self, action: str, adapter_name: Optional[str] = None) -> None:
        """
        Control LoRA adapters.
        
        Args:
            action: Action to perform (list, load, unload, status)
            adapter_name: Adapter name for load/unload operations
        """
        if action == "list":
            adapters = self.adapter_controller.list_adapters()
            print("📋 Available adapters:")
            for adapter in adapters:
                print(f"  • {adapter}")
            
            if not adapters:
                print("  No adapters found")
        
        elif action == "status":
            status = self.adapter_controller.get_adapter_status()
            print("📊 Adapter Status:")
            print(f"  Active adapters: {', '.join(status['active_adapters']) or 'None'}")
            print(f"  Available adapters: {len(status['available_adapters'])}")
            print(f"  Adapter directory: {status['adapter_directory']}")
        
        elif action == "load":
            if not adapter_name:
                raise ValueError("Adapter name required for load operation")
            
            success = self.adapter_controller.load_adapter(adapter_name)
            if success:
                print(f"✅ Adapter '{adapter_name}' loaded")
            else:
                print(f"❌ Failed to load adapter '{adapter_name}'")
        
        elif action == "unload":
            if not adapter_name:
                raise ValueError("Adapter name required for unload operation")
            
            success = self.adapter_controller.unload_adapter(adapter_name)
            if success:
                print(f"✅ Adapter '{adapter_name}' unloaded")
            else:
                print(f"❌ Adapter '{adapter_name}' not currently loaded")
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def session_diagnostics(self, session_id: Optional[str] = None) -> None:
        """
        Show comprehensive session diagnostics.
        
        Args:
            session_id: Specific session ID, or None for all sessions
        """
        if session_id:
            # Single session diagnostics
            stats = self.session_store.get_session_stats(session_id)
            if not stats:
                print(f"❌ Session '{session_id}' not found")
                return
            
            print(f"🔍 Session Diagnostics: {session_id}")
            print("-" * 40)
            print(f"Created: {datetime.fromtimestamp(stats['created_at']).isoformat()}")
            print(f"Last Activity: {datetime.fromtimestamp(stats['last_activity']).isoformat()}")
            print(f"Duration: {stats['session_duration']:.2f}s")
            print(f"Messages: {stats['message_count']}")
            print(f"User Words: {stats['total_user_words']}")
            print(f"Bot Words: {stats['total_bot_words']}")
            
            # Show events for this session
            events = self.logger.get_session_events(session_id)
            if events:
                print(f"\n📋 Pipeline Events ({len(events)}):")
                event_counts = {}
                for event in events:
                    event_counts[event.event_type.value] = event_counts.get(event.event_type.value, 0) + 1
                
                for event_type, count in event_counts.items():
                    print(f"  {event_type}: {count}")
        
        else:
            # All sessions overview
            sessions = self.session_store.list_sessions()
            print(f"🔍 All Sessions Diagnostics ({len(sessions)} sessions)")
            print("-" * 50)
            
            total_messages = 0
            total_duration = 0
            
            for sid in sessions:
                stats = self.session_store.get_session_stats(sid)
                if stats:
                    total_messages += stats['message_count']
                    total_duration += stats['session_duration']
                    
                    print(f"{sid}: {stats['message_count']} msgs, {stats['session_duration']:.1f}s")
            
            print(f"\n📊 Totals:")
            print(f"  Total messages: {total_messages}")
            print(f"  Total duration: {total_duration:.2f}s")
            print(f"  Average session duration: {total_duration/len(sessions):.2f}s" if sessions else "  No sessions")


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voicebot Orchestrator CLI - Sprint 2")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Monitor session command
    monitor_parser = subparsers.add_parser("monitor-session", help="Monitor session state")
    monitor_parser.add_argument("session_id", help="Session ID to monitor")
    monitor_parser.add_argument("--tail", action="store_true", help="Show last events only")
    monitor_parser.add_argument("--follow", action="store_true", help="Follow new events")
    
    # Orchestrator log command
    log_parser = subparsers.add_parser("orchestrator-log", help="View orchestrator logs")
    log_parser.add_argument("--level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                           default="INFO", help="Minimum log level")
    log_parser.add_argument("--lines", type=int, default=50, help="Number of lines to show")
    log_parser.add_argument("--follow", action="store_true", help="Follow new logs")
    
    # Replay session command
    replay_parser = subparsers.add_parser("replay-session", help="Replay session conversation")
    replay_parser.add_argument("session_id", help="Session ID to replay")
    replay_parser.add_argument("--step-by-step", action="store_true", help="Pause between exchanges")
    replay_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    
    # Cache manager command
    cache_parser = subparsers.add_parser("cache-manager", help="Manage semantic cache")
    cache_parser.add_argument("action", choices=["get", "set", "clear", "stats"], help="Cache action")
    cache_parser.add_argument("--key", help="Cache key")
    cache_parser.add_argument("--value", help="Cache value")
    
    # Adapter control command
    adapter_parser = subparsers.add_parser("adapter-control", help="Control LoRA adapters")
    adapter_parser.add_argument("action", choices=["list", "load", "unload", "status"], help="Adapter action")
    adapter_parser.add_argument("--name", help="Adapter name")
    
    # Session diagnostics command
    diag_parser = subparsers.add_parser("diagnostics", help="Show session diagnostics")
    diag_parser.add_argument("--session-id", help="Specific session ID")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = VoicebotCLI()
    
    try:
        if args.command == "monitor-session":
            cli.monitor_session(args.session_id, args.tail, args.follow)
        
        elif args.command == "orchestrator-log":
            cli.orchestrator_log(LogLevel(args.level), args.lines, args.follow)
        
        elif args.command == "replay-session":
            cli.replay_session(args.session_id, args.step_by_step, args.format)
        
        elif args.command == "cache-manager":
            cli.cache_manager_cmd(args.action, args.key, args.value)
        
        elif args.command == "adapter-control":
            cli.adapter_control(args.action, args.name)
        
        elif args.command == "diagnostics":
            cli.session_diagnostics(args.session_id)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
