#!/usr/bin/env python3
"""
LLM Conversation Management System Test
Tests the complete prompt injection and conversation management implementation.
"""
import requests
import time
import json
from pathlib import Path

class LLMConversationSystemTester:
    """Test the complete LLM conversation management system"""
    
    def __init__(self):
        """Initialize the tester"""
        self.mistral_url = "http://localhost:8021"
        self.gpt_url = "http://localhost:8022"
        self.test_phone = "test-phone-conversation-123"
        
        print("🧠 LLM Conversation Management System Tester")
        print("=" * 60)
    
    def test_service_health(self, service_name: str, url: str) -> bool:
        """Test if LLM service is healthy"""
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"  ✅ {service_name} service is healthy")
                return True
            else:
                print(f"  ❌ {service_name} service unhealthy (Status: {response.status_code})")
                return False
        except Exception as e:
            print(f"  ❌ {service_name} service not reachable: {e}")
            return False
    
    def test_prompt_injection(self, service_name: str, url: str):
        """Test prompt injection system"""
        print(f"\n🎯 Testing Prompt Injection - {service_name}")
        print("-" * 50)
        
        # Test each call type
        call_types = ["inbound", "outbound", None]
        
        for call_type in call_types:
            call_type_desc = call_type if call_type else "general"
            
            payload = {
                "text": "What is your name and how can you help me?",
                "call_type": call_type,
                "use_cache": False  # Force fresh response to see prompt effects
            }
            
            try:
                response = requests.post(f"{url}/generate", json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "")
                    
                    # Check if Alex persona is present
                    alex_indicators = ["Alex", "Finally Payoff", "banking", "debt"]
                    has_alex_persona = any(indicator.lower() in response_text.lower() for indicator in alex_indicators)
                    
                    print(f"    ✅ {call_type_desc} call type: {'Alex persona detected' if has_alex_persona else 'Generic response'}")
                    print(f"    💬 Response preview: {response_text[:120]}...")
                    print(f"    ⏱️  Processing time: {result.get('processing_time_seconds', 'N/A')}s")
                else:
                    print(f"    ❌ {call_type_desc} call type failed (Status: {response.status_code})")
                    
            except Exception as e:
                print(f"    ❌ {call_type_desc} call type error: {e}")
                
            time.sleep(1)  # Brief pause between requests
    
    def test_conversation_management(self, service_name: str, url: str):
        """Test conversation management system"""
        print(f"\n💬 Testing Conversation Management - {service_name}")
        print("-" * 50)
        
        conversation_id = None
        
        # Test conversation messages
        conversation_messages = [
            "Hello, I need help with my debt.",
            "I have $15,000 in credit card debt at 24% APR.",
            "How much could I save with a personal loan?",
            "What are the qualification requirements?",
            "I'm interested in applying today."
        ]
        
        for i, message in enumerate(conversation_messages, 1):
            print(f"  📨 Message {i}: {message}")
            
            payload = {
                "text": message,
                "customer_phone": self.test_phone,
                "conversation_id": conversation_id,
                "call_type": "inbound",
                "use_cache": False
            }
            
            try:
                response = requests.post(f"{url}/generate", json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "")
                    conversation_id = result.get("conversation_id")
                    
                    # Check for conversation continuity
                    intro_indicators = ["Hi, I'm Alex", "Hello, I'm Alex", "My name is Alex"]
                    has_intro = any(intro.lower() in response_text.lower() for intro in intro_indicators)
                    
                    if i == 1:
                        # First message should have introduction
                        intro_status = "✅ Has introduction (expected)" if has_intro else "⚠️  No introduction (unexpected)"
                    else:
                        # Subsequent messages should NOT have introduction
                        intro_status = "⚠️  Has introduction (unexpected)" if has_intro else "✅ No introduction (expected)"
                    
                    print(f"    💭 Response: {response_text[:100]}...")
                    print(f"    🆔 Conversation ID: {conversation_id}")
                    print(f"    👋 {intro_status}")
                    print()
                else:
                    print(f"    ❌ Message {i} failed (Status: {response.status_code})")
                    break
                    
            except Exception as e:
                print(f"    ❌ Message {i} error: {e}")
                break
                
            time.sleep(1)  # Brief pause between messages
        
        return conversation_id
    
    def test_conversation_context(self, service_name: str, url: str, conversation_id: str):
        """Test conversation context retrieval"""
        if not conversation_id:
            print(f"  ⚠️  No conversation ID available for context test")
            return
        
        print(f"\n🧠 Testing Conversation Context - {service_name}")
        print("-" * 50)
        
        # Ask about previous conversation
        context_test_message = "What did I tell you about my debt situation?"
        
        payload = {
            "text": context_test_message,
            "customer_phone": self.test_phone,
            "conversation_id": conversation_id,
            "call_type": "inbound",
            "use_cache": False
        }
        
        try:
            response = requests.post(f"{url}/generate", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                # Check if response references previous conversation
                context_indicators = ["$15,000", "15000", "credit card", "debt", "24%", "APR"]
                has_context = any(indicator in response_text for indicator in context_indicators)
                
                print(f"    💭 Context test response: {response_text[:120]}...")
                print(f"    🧠 Context awareness: {'✅ Has context' if has_context else '❌ No context'}")
            else:
                print(f"    ❌ Context test failed (Status: {response.status_code})")
                
        except Exception as e:
            print(f"    ❌ Context test error: {e}")
    
    def run_complete_test(self):
        """Run complete conversation management system test"""
        print("\n🚀 Starting Complete LLM Conversation Management Test")
        print("=" * 70)
        
        # Test both services
        services = [
            ("Mistral LLM", self.mistral_url),
            ("GPT LLM", self.gpt_url)
        ]
        
        for service_name, url in services:
            print(f"\n🎯 Testing {service_name}")
            print("=" * 40)
            
            # Health check first
            if not self.test_service_health(service_name, url):
                print(f"  ⚠️  Skipping {service_name} tests (service not available)")
                continue
            
            # Test prompt injection
            self.test_prompt_injection(service_name, url)
            
            # Test conversation management
            conversation_id = self.test_conversation_management(service_name, url)
            
            # Test conversation context
            self.test_conversation_context(service_name, url, conversation_id)
            
            print(f"\n✅ {service_name} testing completed")
            print("-" * 40)
        
        print("\n🎉 Complete LLM Conversation Management Test Finished!")
        print("=" * 70)

def main():
    """Main test execution"""
    tester = LLMConversationSystemTester()
    tester.run_complete_test()

if __name__ == "__main__":
    main()
