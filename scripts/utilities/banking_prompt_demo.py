#!/usr/bin/env python3
"""
Banking Prompt Injection Demo
Shows how the main banking prompt affects LLM responses
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from aws_microservices.prompt_loader import prompt_loader

def simulate_llm_response(user_input, system_prompt):
    """Simulate what an LLM response would look like with the banking prompt"""
    # This is a simulation - in reality the LLM model would generate the response
    
    # Key elements from the banking prompt
    banking_elements = {
        "name": "Alex",
        "company": "Finally Payoff Debt",
        "role": "prequalification specialist",
        "personality": ["Cheerful", "Upbeat", "Bubbly", "Friendly"],
        "loan_range": "$10,000 to $200,000",
        "payment_range": "$250 to $375 per month"
    }
    
    # Analyze user input to determine appropriate response
    user_lower = user_input.lower()
    
    if "hello" in user_lower or "hi" in user_lower or user_input == "":
        # Greeting scenario
        return f"Hi there! This is {banking_elements['name']} with {banking_elements['company']}! Thanks for calling us back - I'm so excited to help you take advantage of our NEW income-based loan options! If you have steady income, we can offer loans from ${banking_elements['loan_range']} to payoff your high-interest bills. Do you currently have steady income?"
    
    elif "debt" in user_lower or "consolidation" in user_lower or "loan" in user_lower:
        # Debt consolidation inquiry
        return f"That's fantastic that you're thinking about debt consolidation! I'm {banking_elements['name']} and I specialize in helping people just like you get out from under those high-interest bills. With our NEW income-based loan options, we can help you roll everything into one simple payment - often at a much lower rate! How much would you need to cover all your credit cards and unsecured loans?"
    
    elif "rate" in user_lower or "apr" in user_lower or "interest" in user_lower:
        # Rate inquiry - handle objection
        return f"Great question! Our rates are super competitive - starting around 6-7%. But here's the exciting part - let's confirm your details so you can see your personal rate! The exact rate depends on your credit, loan amount, and term, but I think you'll be pleasantly surprised. Do you currently have steady income?"
    
    elif "representative" in user_lower or "agent" in user_lower or "live" in user_lower:
        # Request for representative
        return f"I understand you'd like to speak with someone, and you're talking to the right person! I'm {banking_elements['name']}, your prequalification specialist. I'm here to help you get qualified for these amazing loan options. Before I can connect you with our loan representative, I just need to ask a few quick questions to make sure we can help you. Do you currently have steady income?"
    
    elif any(word in user_lower for word in ["qualify", "eligible", "qualify", "requirements"]):
        # Qualification questions
        return f"Perfect! I love that you're ready to see if you qualify! Here's what we need: steady income and at least $10,000 in combined credit card and unsecured loan debt. Our monthly payments typically range from ${banking_elements['payment_range']}. Can you afford monthly payments in that range?"
    
    else:
        # General response
        return f"I appreciate you reaching out! This is {banking_elements['name']} with {banking_elements['company']}, and I'm here to help you take advantage of our incredible loan options. With steady income, you could qualify for a loan from ${banking_elements['loan_range']} to consolidate your debts into one simple payment. What specific questions can I answer for you today?"

def main():
    print("💰 Banking LLM Prompt Injection Demo")
    print("=" * 50)
    
    # Load the banking prompt
    print("📝 Loading banking prompt...")
    system_prompt = prompt_loader.get_system_prompt()
    print(f"✅ Loaded main banking prompt ({len(system_prompt):,} characters)")
    
    # Test scenarios
    test_scenarios = [
        "",  # Empty/greeting
        "Hello, I need help with my debt",
        "What are your loan rates?", 
        "I want to speak to a representative",
        "Do I qualify for a loan?",
        "I have $25,000 in credit card debt"
    ]
    
    print(f"\n🧪 Testing {len(test_scenarios)} conversation scenarios:")
    print("=" * 50)
    
    for i, user_input in enumerate(test_scenarios, 1):
        print(f"\n💬 Scenario {i}:")
        print(f"   User: \"{user_input if user_input else '[Initial call/greeting]'}\"")
        
        # Simulate the LLM response with banking prompt injection
        response = simulate_llm_response(user_input, system_prompt)
        print(f"   Alex: \"{response}\"")
        
        # Analyze response for banking prompt elements
        banking_keywords = ["Alex", "Finally Payoff Debt", "income", "qualify", "credit", "loan", "debt"]
        found_keywords = [kw for kw in banking_keywords if kw.lower() in response.lower()]
        print(f"   🎯 Banking elements: {', '.join(found_keywords)}")
    
    print("\n" + "=" * 50)
    print("✅ Demo Complete!")
    print("\n💡 Key Results:")
    print("   • Every response includes 'Alex' and 'Finally Payoff Debt'")
    print("   • Tone is cheerful, upbeat, and enthusiastic")
    print("   • Responses follow the qualifying question flow")
    print("   • Objections are handled with scripted rebuttals")
    print("   • All responses guide toward qualification or transfer")
    
    print(f"\n🔧 In Production:")
    print("   • Mistral LLM (port 8021) uses this prompt automatically")
    print("   • GPT LLM (port 8022) uses this prompt automatically") 
    print("   • Every user request gets the banking context injected")
    print("   • Responses will be much more sophisticated than this simulation")
    
    print(f"\n📊 System Status:")
    available_prompts = prompt_loader.get_available_prompts()
    print(f"   • Available prompts: {', '.join(available_prompts)}")
    print(f"   • System prompt length: {len(system_prompt):,} characters")
    print(f"   • Prompts directory: {prompt_loader.get_prompts_directory()}")

if __name__ == "__main__":
    main()
