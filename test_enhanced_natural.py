"""
Test Enhanced LLM with Natural GPT Integration

Tests that the main system now uses natural conversational responses.
"""

import asyncio
import os
import sys
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(__file__))

# Import the enhanced LLM
from voicebot_orchestrator.enhanced_llm import EnhancedMistralLLM

# Set up logging
logging.basicConfig(level=logging.WARNING)

async def test_enhanced_natural_llm():
    """Test enhanced LLM with natural conversational integration."""
    print("=== Testing Enhanced LLM with Natural GPT Integration ===\n")
    
    # Create enhanced LLM with natural GPT approach
    llm = EnhancedMistralLLM(
        model_path="mistral:latest",  # Will try GPT first internally
        enable_cache=True,
        enable_adapters=True
    )
    
    # Test scenarios to verify natural responses
    test_cases = [
        "My name is Mike, I have $15,000 in credit card debt",
        "Hi, can you help me qualify for a loan?",  
        "What are your interest rates?",
        "I'm struggling with multiple credit cards, can you help?",
        "How can I lower my monthly payments?"
    ]
    
    successful_tests = 0
    total_tests = len(test_cases)
    
    print("Testing natural conversational responses...\n")
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"Test {i}: {test_input}")
        
        try:
            response = await llm.generate_response(
                user_input=test_input,
                domain_context="banking",
                use_cache=False  # Disable cache for testing
            )
            
            print(f"Response: {response}\n")
            print("-" * 60 + "\n")
            
            # Check for natural conversational elements
            is_natural = (
                len(response.strip()) > 20 and
                ("hey" in response.lower() or "hi" in response.lower() or "hello" in response.lower()) and
                ("mike" in response.lower() if "mike" in test_input.lower() else True)
            )
            
            if is_natural:
                successful_tests += 1
                print(f"✅ Natural response detected\n")
            else:
                print(f"⚠️  Response may not be fully natural\n")
            
        except Exception as e:
            print(f"ERROR: {e}\n")
            print("-" * 60 + "\n")
    
    # Print results
    print(f"\n=== Enhanced LLM Natural Integration Test Results ===")
    print(f"Natural responses: {successful_tests}/{total_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    # Get performance metrics
    try:
        metrics = llm.get_performance_metrics()
        print(f"\nPerformance metrics:")
        print(f"- Total queries: {metrics['total_queries']}")
        print(f"- Cache hits: {metrics['cache_hits']}")
        print(f"- Ollama calls: {metrics['ollama_calls']}")
        print(f"- Ollama success rate: {metrics.get('ollama_success_rate', 0)*100:.1f}%")
        
        if metrics.get('performance_improvement'):
            perf = metrics['performance_improvement']
            print(f"- AI accuracy improvement: {perf.get('ai_accuracy_improvement', 0):.1f}%")
    except Exception as e:
        print(f"Could not get metrics: {e}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_natural_llm())
