#!/usr/bin/env python3
"""
Call Type Demo - Test inbound vs outbound call prompts
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from aws_microservices.prompt_loader import prompt_loader

def test_call_types():
    print("📞 Call Type Prompt System Demo")
    print("=" * 50)
    
    # Test different call types
    call_types = [
        None,          # General/default
        "inbound",     # Customer called us
        "outbound",    # We called customer
        "unknown"      # Invalid type (should fall back)
    ]
    
    for call_type in call_types:
        print(f"\n🎯 Testing call_type: {call_type or 'None (general)'}")
        print("-" * 40)
        
        # Get system prompt for this call type
        system_prompt = prompt_loader.get_system_prompt(call_type=call_type)
        
        # Show what prompts were loaded
        print(f"📊 System prompt length: {len(system_prompt):,} characters")
        
        # Analyze which prompts were included
        if "OUTBOUND CALL SCRIPT" in system_prompt:
            print("📞 Includes: OUTBOUND call prompt")
        if "INBOUND CALL SCRIPT" in system_prompt:
            print("📞 Includes: INBOUND call prompt")
        if "Greeting & Hook" in system_prompt and "OUTBOUND CALL SCRIPT" not in system_prompt and "INBOUND CALL SCRIPT" not in system_prompt:
            print("📞 Includes: MAIN prompt only")
        
        # Show greeting differences
        print("🎭 Opening greeting would be:")
        if call_type == "outbound":
            print("   'Hi [Customer Name], this is Alex calling from Finally Payoff Debt!'")
        elif call_type == "inbound":
            print("   'Thank you for calling Finally Payoff Debt, this is Alex!'")
        else:
            print("   [Uses main prompt - general greeting]")
    
    print("\n" + "=" * 50)
    print("✅ Call Type Demo Complete!")
    
    print("\n💡 Usage in Practice:")
    print("   Inbound Call API Request:")
    print("   {")
    print('     "text": "I need help with my debt",')
    print('     "call_type": "inbound"')
    print("   }")
    
    print("\n   Outbound Call API Request:")
    print("   {") 
    print('     "text": "I need help with my debt",')
    print('     "call_type": "outbound"')
    print("   }")
    
    print(f"\n📁 Implementation Details:")
    print(f"   • inbound-call.txt: {len(prompt_loader.load_all_prompts().get('inbound-call', '')):,} chars")
    print(f"   • outbound-call.txt: {len(prompt_loader.load_all_prompts().get('outbound-call', '')):,} chars") 
    print(f"   • prompt-main.txt: {len(prompt_loader.load_all_prompts().get('prompt-main', '')):,} chars")
    print(f"   • Priority: call-specific prompt + main prompt as backup")

def show_prompt_differences():
    print("\n📋 Prompt Content Differences")
    print("=" * 50)
    
    prompts = prompt_loader.load_all_prompts()
    
    # Show outbound greeting
    if "outbound-call" in prompts:
        print("📞 OUTBOUND CALL Opening:")
        outbound_content = prompts["outbound-call"]
        # Extract the first few lines
        lines = outbound_content.split('\n')
        for line in lines[:8]:
            if line.strip():
                print(f"   {line}")
    
    print("\n📞 INBOUND CALL Opening:")
    if "inbound-call" in prompts:
        inbound_content = prompts["inbound-call"] 
        lines = inbound_content.split('\n')
        for line in lines[:8]:
            if line.strip():
                print(f"   {line}")

if __name__ == "__main__":
    test_call_types()
    show_prompt_differences()
