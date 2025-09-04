#!/usr/bin/env python3
"""
Enhanced LLM Call Type Testing
Tests both prompt injection and call type functionality with real LLM services
"""
import requests
import json
import time
import sys
from pathlib import Path

def test_call_type_with_llm_services():
    """Test call type functionality with actual LLM services"""
    print("🧪 Enhanced LLM Call Type Testing")
    print("=" * 50)
    
    # LLM service endpoints
    services = {
        "Mistral LLM": "http://localhost:8021",
        "GPT LLM": "http://localhost:8022"
    }
    
    # Test scenarios for both inbound and outbound calls
    test_scenarios = [
        {
            "call_type": "inbound",
            "description": "Customer called us",
            "test_messages": [
                "Hi, I saw your ad about debt consolidation",
                "I need help with my credit card debt",
                "What loan options do you have?"
            ]
        },
        {
            "call_type": "outbound", 
            "description": "We called customer",
            "test_messages": [
                "",  # Initial greeting
                "What is this call about?",
                "I'm not interested",
                "Tell me more about the rates"
            ]
        },
        {
            "call_type": None,
            "description": "General/default",
            "test_messages": [
                "Hello, I need financial help"
            ]
        }
    ]
    
    for service_name, base_url in services.items():
        print(f"\n🧠 Testing {service_name}")
        print("-" * 40)
        
        # Check service health
        try:
            health_response = requests.get(f"{base_url}/health", timeout=5)
            if health_response.status_code != 200:
                print(f"❌ {service_name} not healthy")
                continue
        except requests.exceptions.RequestException:
            print(f"❌ {service_name} not accessible")
            continue
        
        print(f"✅ {service_name} is running")
        
        # Test prompts endpoint (if available)
        try:
            prompts_response = requests.get(f"{base_url}/prompts", timeout=5)
            if prompts_response.status_code == 200:
                info = prompts_response.json()
                print(f"📝 Prompts loaded: {info['total_prompts']}")
            else:
                print("⚠️  Prompts endpoint not available (service needs restart)")
        except:
            print("⚠️  Prompts endpoint not available (service needs restart)")
        
        # Test each call type scenario
        for scenario in test_scenarios:
            call_type = scenario["call_type"]
            print(f"\n📞 {scenario['description']} (call_type: {call_type})")
            
            for i, message in enumerate(scenario["test_messages"]):
                print(f"\n   Test {i+1}: '{message if message else '[Initial greeting]'}'")
                
                try:
                    # Prepare request with call_type
                    request_data = {
                        "text": message,
                        "use_cache": False,
                        "max_tokens": 150,
                        "temperature": 0.7
                    }
                    
                    # Add call_type if service supports it
                    if call_type:
                        request_data["call_type"] = call_type
                    
                    # Make request
                    response = requests.post(
                        f"{base_url}/generate", 
                        json=request_data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        response_text = result['response']
                        processing_time = result['processing_time_seconds']
                        
                        print(f"      ✅ Response in {processing_time:.2f}s:")
                        print(f"      💬 \"{response_text[:150]}{'...' if len(response_text) > 150 else ''}\"")
                        
                        # Analyze response for call type appropriateness
                        response_lower = response_text.lower()
                        
                        if call_type == "inbound":
                            if "thank you for calling" in response_lower or "glad you called" in response_lower:
                                print("      🎯 ✅ Appropriate INBOUND greeting detected!")
                            else:
                                print("      🎯 ⚠️  May not be using inbound-specific prompt")
                        
                        elif call_type == "outbound":
                            if "calling from" in response_lower or "thanks for your interest" in response_lower:
                                print("      🎯 ✅ Appropriate OUTBOUND greeting detected!")  
                            else:
                                print("      🎯 ⚠️  May not be using outbound-specific prompt")
                        
                        # Check for Alex/Finally Payoff Debt
                        if "alex" in response_lower and "finally payoff debt" in response_lower:
                            print("      👤 ✅ Alex persona active")
                        else:
                            print("      👤 ⚠️  Alex persona not detected")
                            
                    else:
                        print(f"      ❌ Request failed: {response.status_code}")
                        if response.text:
                            print(f"         Error: {response.text[:100]}")
                
                except requests.exceptions.RequestException as e:
                    print(f"      ❌ Request error: {e}")
                
                time.sleep(1)  # Small delay between requests
    
    print("\n" + "=" * 50)
    print("🏁 Test Complete!")
    
    print("\n💡 What to Look For:")
    print("   ✅ INBOUND calls: 'Thank you for calling Finally Payoff Debt'")
    print("   ✅ OUTBOUND calls: 'Hi [Name], this is Alex calling from...'") 
    print("   ✅ Both should mention Alex and Finally Payoff Debt")
    print("   ✅ Responses should follow banking conversation flow")
    
    print("\n🔧 If call types aren't working:")
    print("   1. Restart LLM services to load updated prompt injection code")
    print("   2. Verify call_type parameter is being sent in requests")
    print("   3. Check service logs for prompt injection messages")

if __name__ == "__main__":
    test_call_type_with_llm_services()
