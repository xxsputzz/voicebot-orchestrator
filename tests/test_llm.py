"""
Tests for Large Language Model (LLM) functionality.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebot_orchestrator.llm import MistralLLM


class TestMistralLLM:
    """Test cases for MistralLLM."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.llm = MistralLLM(
            model_path="./mock_models/mistral",
            max_tokens=256,
            temperature=0.7
        )
    
    async def test_generate_response_simple(self):
        """Test generating response to simple input."""
        user_input = "Hello"
        response = await self.llm.generate_response(user_input)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Hello!" in response or "Hi" in response or "help" in response
    
    async def test_generate_response_banking_balance(self):
        """Test generating response to balance inquiry."""
        user_input = "What is my account balance?"
        response = await self.llm.generate_response(user_input)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "balance" in response.lower()
    
    async def test_generate_response_banking_transactions(self):
        """Test generating response to transaction inquiry."""
        user_input = "Show me my recent transactions"
        response = await self.llm.generate_response(user_input)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "transaction" in response.lower()
    
    async def test_generate_response_with_history(self):
        """Test generating response with conversation history."""
        user_input = "Thank you"
        conversation_history = [
            {"user_input": "Hello", "bot_response": "Hi there!"},
            {"user_input": "What's my balance?", "bot_response": "Your balance is $100."}
        ]
        
        response = await self.llm.generate_response(user_input, conversation_history)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "welcome" in response.lower() or "thank" in response.lower()
    
    async def test_generate_response_empty_input(self):
        """Test generating response to empty input raises error."""
        try:
            await self.llm.generate_response("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "cannot be empty" in str(e)
    
    async def test_generate_response_whitespace_input(self):
        """Test generating response to whitespace-only input raises error."""
        try:
            await self.llm.generate_response("   \n\t   ")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "cannot be empty" in str(e)
    
    async def test_validate_input_valid(self):
        """Test input validation with valid input."""
        valid_inputs = [
            "Hello",
            "What is my balance?",
            "Show me transactions",
            "Help me with my account"
        ]
        
        for text in valid_inputs:
            is_valid = await self.llm.validate_input(text)
            assert is_valid is True
    
    async def test_validate_input_invalid(self):
        """Test input validation with invalid input."""
        invalid_inputs = [
            "",  # Empty
            "   ",  # Whitespace only
            "x" * 1001,  # Too long
            "What is my password?",  # Contains prohibited word
            "My SSN is 123-45-6789",  # Contains prohibited word
        ]
        
        for text in invalid_inputs:
            is_valid = await self.llm.validate_input(text)
            assert is_valid is False
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.llm.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_path" in info
        assert "max_tokens" in info
        assert "temperature" in info
        assert "loaded" in info
        
        assert info["model_path"] == "./mock_models/mistral"
        assert info["max_tokens"] == 256
        assert info["temperature"] == 0.7
        assert info["loaded"] is False  # Not loaded initially
    
    def test_build_context(self):
        """Test context building functionality."""
        user_input = "Hello"
        history = [
            {"user_input": "Hi", "bot_response": "Hello there!"}
        ]
        
        context = self.llm._build_context(user_input, history)
        
        assert isinstance(context, str)
        assert "System:" in context
        assert "banking assistant" in context.lower()
        assert "User: Hi" in context
        assert "Assistant: Hello there!" in context
        assert "User: Hello" in context
        assert context.endswith("Assistant:")
    
    def test_build_context_no_history(self):
        """Test context building without history."""
        user_input = "Hello"
        context = self.llm._build_context(user_input, None)
        
        assert isinstance(context, str)
        assert "System:" in context
        assert "User: Hello" in context
        assert context.endswith("Assistant:")
    
    def test_build_context_long_history(self):
        """Test context building with long history (should truncate)."""
        user_input = "Hello"
        long_history = []
        
        # Create 10 history entries
        for i in range(10):
            long_history.append({
                "user_input": f"Question {i}",
                "bot_response": f"Answer {i}"
            })
        
        context = self.llm._build_context(user_input, long_history)
        
        # Should only include last 5 exchanges
        question_count = context.count("Question")
        assert question_count == 5
        
        # Should include the most recent questions (5-9)
        assert "Question 9" in context
        assert "Question 8" in context
        assert "Question 5" in context
        assert "Question 4" not in context  # Should be truncated
    
    def test_model_loading(self):
        """Test model loading functionality."""
        # Model should not be loaded initially
        assert self.llm._model is None
        
        # Trigger model loading
        self.llm._load_model()
        
        # Model should now be loaded (mock)
        assert self.llm._model is not None
        assert isinstance(self.llm._model, str)


# Test runner for pytest compatibility
async def run_tests():
    """Run all tests."""
    test_class = TestMistralLLM()
    
    async_test_methods = [
        test_class.test_generate_response_simple,
        test_class.test_generate_response_banking_balance,
        test_class.test_generate_response_banking_transactions,
        test_class.test_generate_response_with_history,
        test_class.test_generate_response_empty_input,
        test_class.test_generate_response_whitespace_input,
        test_class.test_validate_input_valid,
        test_class.test_validate_input_invalid,
    ]
    
    sync_test_methods = [
        test_class.test_get_model_info,
        test_class.test_build_context,
        test_class.test_build_context_no_history,
        test_class.test_build_context_long_history,
        test_class.test_model_loading,
    ]
    
    passed = 0
    failed = 0
    
    # Run async tests
    for test_method in async_test_methods:
        test_class.setup_method()
        try:
            await test_method()
            print(f"✓ {test_method.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_method.__name__}: {e}")
            failed += 1
    
    # Run sync tests
    for test_method in sync_test_methods:
        test_class.setup_method()
        try:
            test_method()
            print(f"✓ {test_method.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_method.__name__}: {e}")
            failed += 1
    
    print(f"\nLLM Tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
