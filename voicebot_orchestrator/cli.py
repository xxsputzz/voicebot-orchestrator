"""
CLI module for voicebot orchestrator management.
"""
import asyncio
import sys
import os
from pathlib import Path
from typing import Optional
import json

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebot_orchestrator.config import settings
from voicebot_orchestrator.stt import WhisperSTT
from voicebot_orchestrator.tts import KokoroTTS
from voicebot_orchestrator.session_manager import SessionManager


class VoicebotCLI:
    """Command-line interface for voicebot orchestrator."""
    
    def __init__(self):
        """Initialize CLI with services."""
        self.stt_service = WhisperSTT(
            model_name=settings.whisper_model,
            device=settings.whisper_device
        )
        
        self.tts_service = KokoroTTS(
            voice=settings.kokoro_voice,
            language=settings.kokoro_language,
            speed=settings.kokoro_speed
        )
        
        self.session_manager = SessionManager(
            timeout=settings.session_timeout,
            max_sessions=settings.max_concurrent_sessions
        )
    
    async def start_call(self, session_id: str) -> None:
        """
        Start a new call session.
        
        Args:
            session_id: Unique session identifier
        """
        try:
            session = await self.session_manager.create_session(session_id)
            print(f"✓ Call session started: {session_id}")
            print(f"  Created at: {session.created_at}")
            print(f"  State: {session.state.value}")
            print(f"  WebSocket URL: ws://{settings.host}:{settings.port}/ws/{session_id}")
            
        except ValueError as e:
            print(f"✗ Error starting call session: {e}")
            sys.exit(1)
    
    async def stt_test(self, input_path: str) -> None:
        """
        Test speech-to-text functionality.
        
        Args:
            input_path: Path to audio file for testing
        """
        try:
            if not os.path.exists(input_path):
                print(f"✗ Audio file not found: {input_path}")
                sys.exit(1)
            
            if not self.stt_service.validate_audio_format(input_path):
                print(f"✗ Unsupported audio format: {input_path}")
                print(f"  Supported formats: {', '.join(self.stt_service.get_supported_formats())}")
                sys.exit(1)
            
            print(f"🎵 Testing STT with: {input_path}")
            print("  Processing audio...")
            
            result = await self.stt_service.transcribe_file(input_path)
            
            print("✓ STT Test Results:")
            print(f"  Transcription: '{result}'")
            print(f"  Model: {self.stt_service.model_name}")
            print(f"  Device: {self.stt_service.device}")
            
        except Exception as e:
            print(f"✗ STT test failed: {e}")
            sys.exit(1)
    
    async def tts_test(self, text: str, output_path: Optional[str] = None) -> None:
        """
        Test text-to-speech functionality.
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save audio output
        """
        try:
            if not await self.tts_service.validate_text(text):
                print(f"✗ Invalid text input: {text}")
                sys.exit(1)
            
            print(f"🔊 Testing TTS with: '{text}'")
            print("  Synthesizing speech...")
            
            audio_data = await self.tts_service.synthesize_speech(text)
            
            print("✓ TTS Test Results:")
            print(f"  Audio size: {len(audio_data)} bytes")
            print(f"  Voice: {self.tts_service.voice}")
            print(f"  Language: {self.tts_service.language}")
            print(f"  Speed: {self.tts_service.speed}")
            
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(audio_data)
                print(f"  Audio saved to: {output_path}")
            
        except Exception as e:
            print(f"✗ TTS test failed: {e}")
            sys.exit(1)
    
    async def list_sessions(self) -> None:
        """List all active sessions."""
        try:
            active_sessions = await self.session_manager.list_active_sessions()
            
            print("📋 Active Sessions:")
            if not active_sessions:
                print("  No active sessions")
            else:
                for session_id in active_sessions:
                    session = await self.session_manager.get_session(session_id)
                    if session:
                        print(f"  • {session_id}")
                        print(f"    Created: {session.created_at}")
                        print(f"    Last activity: {session.last_activity}")
                        print(f"    History entries: {len(session.conversation_history)}")
            
            print(f"Total: {len(active_sessions)} sessions")
            
        except Exception as e:
            print(f"✗ Error listing sessions: {e}")
            sys.exit(1)
    
    async def end_session(self, session_id: str) -> None:
        """
        End a specific session.
        
        Args:
            session_id: Session identifier to end
        """
        try:
            success = await self.session_manager.end_session(session_id)
            
            if success:
                print(f"✓ Session ended: {session_id}")
            else:
                print(f"✗ Session not found: {session_id}")
                sys.exit(1)
            
        except Exception as e:
            print(f"✗ Error ending session: {e}")
            sys.exit(1)
    
    def show_config(self) -> None:
        """Display current configuration."""
        print("⚙️  Current Configuration:")
        print(f"  Host: {settings.host}")
        print(f"  Port: {settings.port}")
        print(f"  Log Level: {settings.log_level}")
        print()
        print("  STT Configuration:")
        print(f"    Model: {settings.whisper_model}")
        print(f"    Device: {settings.whisper_device}")
        print()
        print("  LLM Configuration:")
        print(f"    Model Path: {settings.mistral_model_path}")
        print(f"    Max Tokens: {settings.mistral_max_tokens}")
        print(f"    Temperature: {settings.mistral_temperature}")
        print()
        print("  TTS Configuration:")
        print(f"    Voice: {settings.kokoro_voice}")
        print(f"    Language: {settings.kokoro_language}")
        print(f"    Speed: {settings.kokoro_speed}")
        print()
        print("  Session Configuration:")
        print(f"    Timeout: {settings.session_timeout}s")
        print(f"    Max Concurrent: {settings.max_concurrent_sessions}")


async def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Voicebot Orchestrator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m voicebot_orchestrator.cli start-call --session-id test-123
  python -m voicebot_orchestrator.cli stt-test --input audio.wav
  python -m voicebot_orchestrator.cli tts-test --text "Hello world"
  python -m voicebot_orchestrator.cli list-sessions
  python -m voicebot_orchestrator.cli config
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # start-call command
    start_call_parser = subparsers.add_parser('start-call', help='Start a new call session')
    start_call_parser.add_argument('--session-id', required=True, help='Unique session identifier')
    
    # stt-test command
    stt_test_parser = subparsers.add_parser('stt-test', help='Test speech-to-text functionality')
    stt_test_parser.add_argument('--input', required=True, help='Path to audio file')
    
    # tts-test command
    tts_test_parser = subparsers.add_parser('tts-test', help='Test text-to-speech functionality')
    tts_test_parser.add_argument('--text', required=True, help='Text to synthesize')
    tts_test_parser.add_argument('--output', help='Path to save audio output')
    
    # list-sessions command
    subparsers.add_parser('list-sessions', help='List all active sessions')
    
    # end-session command
    end_session_parser = subparsers.add_parser('end-session', help='End a specific session')
    end_session_parser.add_argument('--session-id', required=True, help='Session identifier to end')
    
    # config command
    subparsers.add_parser('config', help='Display current configuration')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = VoicebotCLI()
    
    try:
        if args.command == 'start-call':
            await cli.start_call(args.session_id)
        
        elif args.command == 'stt-test':
            await cli.stt_test(args.input)
        
        elif args.command == 'tts-test':
            await cli.tts_test(args.text, args.output)
        
        elif args.command == 'list-sessions':
            await cli.list_sessions()
        
        elif args.command == 'end-session':
            await cli.end_session(args.session_id)
        
        elif args.command == 'config':
            cli.show_config()
        
    except KeyboardInterrupt:
        print("\n✗ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
