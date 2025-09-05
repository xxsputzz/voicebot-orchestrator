#!/usr/bin/env python3
"""
Quick test to verify LLM responds to specific user input
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

import asyncio

from voicebot_orchestrator.enhanced_llm import EnhancedMistralLLM

async def test_llm_response():
    print("🧪 Testing LLM Response to Specific Input")
    print("=" * 50)
    
    # Initialize the LLM with Ollama model
    llm = EnhancedMistralLLM(
        model_path="mistral:latest",
        enable_cache=False,
        enable_adapters=True
    )
    
    # Test input
    test_input = "My name is Mike, follow me Mike, or virtual news Mike."
    print(f"📝 Input: '{test_input}'")
    
    # Generate response using banking domain context
    try:
        response = await llm.generate_response(
            user_input=test_input,
            domain_context="Alex banking specialist from Finally Payoff Debt"
        )
        
        print(f"🤖 Response: '{response}'")
        
        # Check if the response acknowledges the specific input
        if test_input.lower() in response.lower() or "mike" in response.lower():
            print("✅ SUCCESS: LLM acknowledged the specific input!")
        else:
            print("❌ ISSUE: LLM gave generic response without acknowledging input")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm_response())
