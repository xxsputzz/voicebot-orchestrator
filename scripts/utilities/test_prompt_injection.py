#!/usr/bin/env python3
"""
Test script to demonstrate prompt injection functionality in LLM services
"""
import requests
import json
import time
import sys
from pathlib import Path

def test_prompt_injection():
    """Test the prompt injection functionality"""
    print("🧪 Testing LLM Prompt Injection System")
    print("=" * 50)
    
    # LLM service endpoints
    services = {
        "Mistral LLM": "http://localhost:8021",
        "GPT LLM": "http://localhost:8022"
    }
    
    for service_name, base_url in services.items():
        print(f"\n🔍 Testing {service_name}")
        print("-" * 30)
        
        # Test 1: Check if service is running
        try:
            health_response = requests.get(f"{base_url}/health", timeout=5)
            if health_response.status_code != 200:
                print(f"❌ {service_name} not healthy (Status: {health_response.status_code})")
                continue
        except requests.exceptions.RequestException as e:
            print(f"❌ {service_name} not accessible: {e}")
            continue
            
        print(f"✅ {service_name} is healthy")
        
        # Test 2: Check prompts info
        try:
            prompts_response = requests.get(f"{base_url}/prompts", timeout=10)
            if prompts_response.status_code == 200:
                prompts_info = prompts_response.json()
                print(f"📁 Prompts directory: {prompts_info['prompts_directory']}")
                print(f"📝 Available prompts: {prompts_info['available_prompts']}")
                print(f"📊 System prompt length: {prompts_info['system_prompt_length']} characters")
                print(f"👀 Preview: {prompts_info['system_prompt_preview']}")
            else:
                print(f"❌ Failed to get prompts info (Status: {prompts_response.status_code})")
                continue
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get prompts info: {e}")
            continue
        
        # Test 3: Test generation with prompt injection
        test_messages = [
            "Hello, I need help with debt consolidation.",
            "What loan options do you have for me?",
            "Can you tell me about your rates?",
            "I want to speak to a representative."
        ]
        
        print(f"\n🧠 Testing {service_name} Response Generation:")
        for i, message in enumerate(test_messages, 1):
            print(f"\n  Test {i}: '{message}'")
            try:
                generation_response = requests.post(
                    f"{base_url}/generate",
                    json={
                        "text": message,
                        "use_cache": False,  # Disable cache for testing
                        "max_tokens": 150,
                        "temperature": 0.7
                    },
                    timeout=30
                )
                
                if generation_response.status_code == 200:
                    result = generation_response.json()
                    response_text = result['response']
                    processing_time = result['processing_time_seconds']
                    print(f"    ✅ Generated in {processing_time:.2f}s")
                    print(f"    💬 Response: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
                    
                    # Check if response contains elements from the prompt (Alex, Finally Payoff Debt, etc.)
                    prompt_indicators = ["Alex", "Finally Payoff Debt", "loan", "income", "credit", "qualify"]
                    found_indicators = [indicator for indicator in prompt_indicators if indicator.lower() in response_text.lower()]
                    if found_indicators:
                        print(f"    🎯 Prompt injection detected! Found: {', '.join(found_indicators)}")
                    else:
                        print(f"    ⚠️  No clear prompt indicators found")
                else:
                    print(f"    ❌ Generation failed (Status: {generation_response.status_code})")
                    print(f"        Error: {generation_response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"    ❌ Generation request failed: {e}")
            
            time.sleep(1)  # Small delay between requests
    
    print("\n" + "=" * 50)
    print("🏁 Test Complete!")
    print("\n💡 How to verify prompt injection:")
    print("   1. Check if LLM responses mention 'Alex' or 'Finally Payoff Debt'")
    print("   2. Look for banking/loan-specific language and tone")
    print("   3. Verify responses follow the conversation flow from prompt-main.txt")
    print("\n📁 Prompt files location: docs/prompts/")
    print("   - Add more .txt files to expand the prompt library")
    print("   - Use /prompts/reload endpoint to refresh prompts without restart")

if __name__ == "__main__":
    test_prompt_injection()
