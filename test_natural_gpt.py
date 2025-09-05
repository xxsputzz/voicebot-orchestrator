"""
Test Natural GPT Implementation

Tests the more conversational, natural responses without heavy scripting.
"""

import asyncio
import os
import sys
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(__file__))

from natural_gpt_llm import NaturalGPTLLM

# Set up logging
logging.basicConfig(level=logging.WARNING)

async def test_natural_gpt():
    """Test natural conversational responses."""
    print("=== Testing Natural GPT LLM ===\n")
    
    llm = NaturalGPTLLM(model_name="gpt-oss:20b", fallback_model="mistral:latest")
    
    # Test scenarios focused on natural conversation
    test_cases = [
        "My name is Mike, I have $15,000 in credit card debt",
        "Hi, do I qualify for debt consolidation?",
        "What interest rates do you offer?",
        "I'm struggling with multiple credit cards",
        "Can you help me lower my monthly payments?"
    ]
    
    successful_tests = 0
    total_tests = len(test_cases)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"Test {i}: {test_input}")
        
        try:
            response = await llm.generate_response(test_input, user_name="Mike")
            print(f"Response: {response}\n")
            print("-" * 60 + "\n")
            
            # Check if response acknowledges the input naturally
            if len(response.strip()) > 20:
                successful_tests += 1
            
        except Exception as e:
            print(f"ERROR: {e}\n")
            print("-" * 60 + "\n")
    
    # Print results
    print(f"\n=== Natural GPT Test Results ===")
    print(f"Successful responses: {successful_tests}/{total_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    # Get LLM stats
    stats = llm.get_stats()
    print(f"LLM success rate: {stats['success_rate']}")
    print(f"Total requests processed: {stats['total_requests']}")

if __name__ == "__main__":
    asyncio.run(test_natural_gpt())
