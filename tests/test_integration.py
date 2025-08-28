"""
Integration tests for the voicebot orchestrator.
"""
import asyncio
import sys
import tempfile
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebot_orchestrator.session_manager import SessionManager
from voicebot_orchestrator.stt import WhisperSTT
from voicebot_orchestrator.llm import MistralLLM
from voicebot_orchestrator.tts import KokoroTTS


class TestIntegration:
    """Integration test cases for the complete pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.session_manager = SessionManager(timeout=60, max_sessions=5)
        self.stt = WhisperSTT(model_name="base", device="cpu")
        self.llm = MistralLLM(
            model_path="./mock_models/mistral",
            max_tokens=256,
            temperature=0.7
        )
        self.tts = KokoroTTS(voice="default", language="en", speed=1.0)
    
    async def test_complete_pipeline(self):
        """Test complete STT -> LLM -> TTS pipeline."""
        # Create a session
        session_id = "integration-test-1"
        session = await self.session_manager.create_session(session_id)
        assert session is not None
        
        # Mock audio input (simulate user saying "What's my balance?")
        audio_input = b"x" * 2000  # Medium size to trigger balance inquiry
        
        # Step 1: Speech-to-Text
        user_text = await self.stt.transcribe_audio(audio_input)
        assert user_text == "How can I help you today?"
        
        # Step 2: Language Model processing
        response_text = await self.llm.generate_response(user_text)
        assert isinstance(response_text, str)
        assert len(response_text) > 0
        
        # Step 3: Text-to-Speech
        audio_output = await self.tts.synthesize_speech(response_text)
        assert isinstance(audio_output, bytes)
        assert len(audio_output) > 0
        
        # Add to conversation history
        success = await self.session_manager.add_to_history(
            session_id, user_text, response_text
        )
        assert success is True
        
        # Verify session state
        updated_session = await self.session_manager.get_session(session_id)
        assert len(updated_session.conversation_history) == 1
        
        # Clean up
        await self.session_manager.end_session(session_id)
    
    async def test_conversation_flow(self):
        """Test multi-turn conversation flow."""
        session_id = "integration-test-2"
        session = await self.session_manager.create_session(session_id)
        
        # First exchange: Greeting
        audio1 = b"x" * 500  # Small size for "Hello"
        text1 = await self.stt.transcribe_audio(audio1)
        response1 = await self.llm.generate_response(text1)
        audio_out1 = await self.tts.synthesize_speech(response1)
        
        await self.session_manager.add_to_history(session_id, text1, response1)
        
        # Second exchange: Balance inquiry with history
        audio2 = b"x" * 2000  # Medium size for balance inquiry
        text2 = await self.stt.transcribe_audio(audio2)
        
        # Get conversation history
        session = await self.session_manager.get_session(session_id)
        history = session.conversation_history
        
        response2 = await self.llm.generate_response(text2, history)
        audio_out2 = await self.tts.synthesize_speech(response2)
        
        await self.session_manager.add_to_history(session_id, text2, response2)
        
        # Third exchange: Thank you
        audio3 = b"thank you for your help"
        text3 = await self.stt.transcribe_audio(audio3)
        
        session = await self.session_manager.get_session(session_id)
        history = session.conversation_history
        
        response3 = await self.llm.generate_response(text3, history)
        audio_out3 = await self.tts.synthesize_speech(response3)
        
        await self.session_manager.add_to_history(session_id, text3, response3)
        
        # Verify conversation history
        final_session = await self.session_manager.get_session(session_id)
        assert len(final_session.conversation_history) == 3
        
        # Verify all audio outputs are valid
        assert all(isinstance(audio, bytes) and len(audio) > 0 
                  for audio in [audio_out1, audio_out2, audio_out3])
        
        # Clean up
        await self.session_manager.end_session(session_id)
    
    async def test_error_handling_pipeline(self):
        """Test error handling in the pipeline."""
        session_id = "integration-test-3"
        session = await self.session_manager.create_session(session_id)
        
        # Test with empty audio (should raise error)
        try:
            await self.stt.transcribe_audio(b"")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Test with invalid text for LLM
        try:
            await self.llm.generate_response("")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Test with invalid text for TTS
        try:
            await self.tts.synthesize_speech("")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Clean up
        await self.session_manager.end_session(session_id)
    
    async def test_concurrent_sessions(self):
        """Test handling multiple concurrent sessions."""
        session_ids = ["concurrent-1", "concurrent-2", "concurrent-3"]
        
        # Create multiple sessions
        sessions = {}
        for session_id in session_ids:
            session = await self.session_manager.create_session(session_id)
            sessions[session_id] = session
        
        # Process audio in each session concurrently
        async def process_session(session_id, audio_data):
            text = await self.stt.transcribe_audio(audio_data)
            response = await self.llm.generate_response(text)
            audio_out = await self.tts.synthesize_speech(response)
            await self.session_manager.add_to_history(session_id, text, response)
            return len(audio_out)
        
        # Run concurrent processing
        tasks = []
        for i, session_id in enumerate(session_ids):
            audio_data = b"x" * (1000 * (i + 1))  # Different sizes
            task = process_session(session_id, audio_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Verify all sessions processed successfully
        assert len(results) == 3
        assert all(result > 0 for result in results)
        
        # Verify sessions still exist and have history
        for session_id in session_ids:
            session = await self.session_manager.get_session(session_id)
            assert session is not None
            assert len(session.conversation_history) == 1
        
        # Clean up
        for session_id in session_ids:
            await self.session_manager.end_session(session_id)
    
    async def test_session_persistence(self):
        """Test session data persistence through conversation."""
        session_id = "persistence-test"
        
        # Create session with metadata
        metadata = {"user_id": "test-user", "channel": "web"}
        session = await self.session_manager.create_session(session_id, metadata)
        
        # Verify metadata persists
        assert session.metadata == metadata
        
        # Process multiple exchanges
        exchanges = [
            b"x" * 500,   # "Hello"
            b"x" * 2000,  # "How can I help you today?"
            b"x" * 5000,  # "I would like to check my account balance please."
        ]
        
        for i, audio_data in enumerate(exchanges):
            text = await self.stt.transcribe_audio(audio_data)
            
            # Get current session for history
            current_session = await self.session_manager.get_session(session_id)
            history = current_session.conversation_history
            
            response = await self.llm.generate_response(text, history)
            await self.session_manager.add_to_history(session_id, text, response)
            
            # Verify session state after each exchange
            updated_session = await self.session_manager.get_session(session_id)
            assert len(updated_session.conversation_history) == i + 1
            assert updated_session.metadata == metadata
        
        # Verify final state
        final_session = await self.session_manager.get_session(session_id)
        assert len(final_session.conversation_history) == 3
        assert final_session.metadata == metadata
        
        # Clean up
        await self.session_manager.end_session(session_id)
    
    async def test_input_validation_pipeline(self):
        """Test input validation throughout the pipeline."""
        session_id = "validation-test"
        session = await self.session_manager.create_session(session_id)
        
        # Valid input should pass through entire pipeline
        valid_audio = b"x" * 1000
        text = await self.stt.transcribe_audio(valid_audio)
        
        # Validate LLM input
        is_valid_llm = await self.llm.validate_input(text)
        assert is_valid_llm is True
        
        response = await self.llm.generate_response(text)
        
        # Validate TTS input
        is_valid_tts = await self.tts.validate_text(response)
        assert is_valid_tts is True
        
        audio_out = await self.tts.synthesize_speech(response)
        assert len(audio_out) > 0
        
        # Test with potentially problematic input
        problematic_text = "What is my password for login?"
        is_valid_problematic = await self.llm.validate_input(problematic_text)
        assert is_valid_problematic is False  # Should be rejected
        
        # Clean up
        await self.session_manager.end_session(session_id)


# Test runner for pytest compatibility
async def run_tests():
    """Run all integration tests."""
    test_class = TestIntegration()
    
    test_methods = [
        test_class.test_complete_pipeline,
        test_class.test_conversation_flow,
        test_class.test_error_handling_pipeline,
        test_class.test_concurrent_sessions,
        test_class.test_session_persistence,
        test_class.test_input_validation_pipeline,
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
    
    print(f"\nIntegration Tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
