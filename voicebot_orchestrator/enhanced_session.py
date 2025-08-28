"""
Enhanced session management integration with CLI - Sprint 2
"""
import asyncio
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path

from .session_manager import SessionManager, SessionData, SessionState
from .voicebot_cli import StructuredLogger, PipelineEvent, EventType


class EnhancedSessionManager(SessionManager):
    """Session manager with integrated logging and diagnostics."""
    
    def __init__(self, timeout: int = 3600, max_sessions: int = 10, logger: Optional[StructuredLogger] = None):
        """
        Initialize enhanced session manager.
        
        Args:
            timeout: Session timeout in seconds
            max_sessions: Maximum concurrent sessions
            logger: Structured logger instance
        """
        super().__init__(timeout, max_sessions)
        self.logger = logger or StructuredLogger()
        self.session_metrics: Dict[str, Dict[str, Any]] = {}
    
    async def create_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> SessionData:
        """Create session with logging."""
        session_data = await super().create_session(session_id, metadata)
        
        # Log session creation
        event = PipelineEvent(
            timestamp=time.time(),
            session_id=session_id,
            event_type=EventType.SESSION_START,
            component="session_manager",
            data={"metadata": metadata or {}}
        )
        self.logger.log_event(event)
        
        # Initialize metrics
        self.session_metrics[session_id] = {
            "created_at": session_data.created_at,
            "total_exchanges": 0,
            "total_processing_time": 0.0,
            "component_times": {
                "stt": 0.0,
                "llm": 0.0,
                "tts": 0.0
            },
            "error_count": 0
        }
        
        return session_data
    
    async def end_session(self, session_id: str) -> bool:
        """End session with logging."""
        success = await super().end_session(session_id)
        
        if success:
            # Log session end
            metrics = self.session_metrics.get(session_id, {})
            event = PipelineEvent(
                timestamp=time.time(),
                session_id=session_id,
                event_type=EventType.SESSION_END,
                component="session_manager",
                data=metrics
            )
            self.logger.log_event(event)
            
            # Clean up metrics
            if session_id in self.session_metrics:
                del self.session_metrics[session_id]
        
        return success
    
    async def log_component_execution(self, session_id: str, component: str, start_time: float, end_time: float, data: Dict[str, Any], error: Optional[str] = None) -> None:
        """
        Log component execution metrics.
        
        Args:
            session_id: Session identifier
            component: Component name (stt, llm, tts)
            start_time: Start timestamp
            end_time: End timestamp
            data: Component-specific data
            error: Error message if any
        """
        duration_ms = (end_time - start_time) * 1000
        
        # Log start event
        start_event = PipelineEvent(
            timestamp=start_time,
            session_id=session_id,
            event_type=getattr(EventType, f"{component.upper()}_START"),
            component=component,
            data=data
        )
        self.logger.log_event(start_event)
        
        # Log completion event
        end_event = PipelineEvent(
            timestamp=end_time,
            session_id=session_id,
            event_type=getattr(EventType, f"{component.upper()}_COMPLETE"),
            component=component,
            data=data,
            duration_ms=duration_ms,
            error=error
        )
        self.logger.log_event(end_event)
        
        # Update metrics
        if session_id in self.session_metrics:
            metrics = self.session_metrics[session_id]
            metrics["total_processing_time"] += duration_ms
            metrics["component_times"][component] += duration_ms
            
            if error:
                metrics["error_count"] += 1
    
    def get_session_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive session metrics."""
        return self.session_metrics.get(session_id)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all active sessions."""
        return self.session_metrics.copy()


class SessionReplayManager:
    """Manages session replay functionality."""
    
    def __init__(self, session_store_dir: str = "./sessions"):
        self.session_store_dir = Path(session_store_dir)
        self.session_store_dir.mkdir(parents=True, exist_ok=True)
    
    def save_session_snapshot(self, session_id: str, session_data: SessionData) -> None:
        """Save session snapshot for replay."""
        snapshot = {
            "session_id": session_data.session_id,
            "state": session_data.state.value,
            "created_at": session_data.created_at,
            "last_activity": session_data.last_activity,
            "metadata": session_data.metadata,
            "conversation_history": session_data.conversation_history
        }
        
        snapshot_file = self.session_store_dir / f"{session_id}_snapshot.json"
        with open(snapshot_file, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, default=str)
    
    def load_session_snapshot(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session snapshot for replay."""
        snapshot_file = self.session_store_dir / f"{session_id}_snapshot.json"
        
        if snapshot_file.exists():
            with open(snapshot_file, "r", encoding="utf-8") as f:
                return json.load(f)
        
        return None
    
    def list_available_replays(self) -> list[str]:
        """List sessions available for replay."""
        snapshot_files = list(self.session_store_dir.glob("*_snapshot.json"))
        return [f.stem.replace("_snapshot", "") for f in snapshot_files]
    
    def generate_replay_script(self, session_id: str) -> Optional[str]:
        """Generate a replay script for a session."""
        snapshot = self.load_session_snapshot(session_id)
        if not snapshot:
            return None
        
        script_lines = [
            f"# Replay script for session: {session_id}",
            f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "import asyncio",
            "from voicebot_orchestrator.main import app",
            "",
            "async def replay_session():",
            f'    session_id = "{session_id}"',
            "    # Session metadata:",
            f"    # Created: {snapshot.get('created_at')}",
            f"    # Messages: {len(snapshot.get('conversation_history', []))}",
            "",
        ]
        
        for i, exchange in enumerate(snapshot.get('conversation_history', []), 1):
            user_input = exchange.get('user_input', '')
            bot_response = exchange.get('bot_response', '')
            
            script_lines.extend([
                f"    # Exchange {i}",
                f'    user_input = "{user_input}"',
                f'    expected_response = "{bot_response}"',
                "    # TODO: Add test assertions",
                "",
            ])
        
        script_lines.extend([
            "if __name__ == '__main__':",
            "    asyncio.run(replay_session())"
        ])
        
        return "\n".join(script_lines)


class ChainlitIntegration:
    """Integration with Chainlit for browser-based testing."""
    
    def __init__(self, session_manager: EnhancedSessionManager):
        self.session_manager = session_manager
        self.test_scenarios: Dict[str, Dict[str, Any]] = {}
    
    def register_test_scenario(self, scenario_name: str, scenario_config: Dict[str, Any]) -> None:
        """
        Register a test scenario for browser testing.
        
        Args:
            scenario_name: Name of the test scenario
            scenario_config: Scenario configuration including test steps
        """
        self.test_scenarios[scenario_name] = scenario_config
    
    def get_test_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered test scenarios."""
        return self.test_scenarios.copy()
    
    def create_chainlit_config(self) -> Dict[str, Any]:
        """Create Chainlit configuration for browser testing."""
        return {
            "app_name": "Voicebot Orchestrator Test Suite",
            "scenarios": self.test_scenarios,
            "session_config": {
                "timeout": self.session_manager.timeout,
                "max_sessions": self.session_manager.max_sessions
            },
            "test_endpoints": [
                "/health",
                "/sessions",
                "/stt/test",
                "/tts/test"
            ]
        }
    
    def generate_chainlit_app(self) -> str:
        """Generate Chainlit app code for browser testing."""
        return '''"""
Chainlit app for Voicebot Orchestrator browser testing.
"""
import chainlit as cl
import asyncio
import json
from voicebot_orchestrator.main import app

@cl.on_chat_start
async def start():
    """Initialize chat session."""
    await cl.Message(content="🤖 Voicebot Orchestrator Test Interface").send()
    
    # Show available test scenarios
    scenarios = get_test_scenarios()
    scenario_list = "\\n".join([f"• {name}" for name in scenarios.keys()])
    
    await cl.Message(
        content=f"Available test scenarios:\\n{scenario_list}\\n\\nType a scenario name to start testing."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle chat messages."""
    user_input = message.content.strip()
    
    if user_input in get_test_scenarios():
        # Run test scenario
        await run_test_scenario(user_input)
    else:
        # Regular voice processing
        await process_voice_input(user_input)

async def run_test_scenario(scenario_name: str):
    """Run a specific test scenario."""
    await cl.Message(content=f"🧪 Running test scenario: {scenario_name}").send()
    
    # TODO: Implement scenario execution
    await cl.Message(content="✅ Test scenario completed").send()

async def process_voice_input(user_input: str):
    """Process voice input through the pipeline."""
    await cl.Message(content=f"🎙️ Processing: {user_input}").send()
    
    # TODO: Integrate with voice pipeline
    await cl.Message(content="🔊 Voice response generated").send()

def get_test_scenarios():
    """Get available test scenarios."""
    return {
        "banking_balance": {"description": "Test balance inquiry flow"},
        "banking_transfer": {"description": "Test transfer request flow"},
        "error_handling": {"description": "Test error scenarios"}
    }
'''
