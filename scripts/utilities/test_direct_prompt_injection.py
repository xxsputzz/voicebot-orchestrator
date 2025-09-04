#!/usr/bin/env python3
"""
Direct test of prompt injection system without requiring running services
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from aws_microservices.prompt_loader import prompt_loader

def main():
    print("🧠 LLM Prompt Injection System - Direct Test")
    print("=" * 50)
    
    # Test prompt loading
    print("1. Loading prompts from docs/prompts/...")
    prompts = prompt_loader.load_all_prompts()
    print(f"   ✅ Found {len(prompts)} prompt files")
    
    for name, content in prompts.items():
        print(f"   📄 {name}: {len(content):,} characters")
    
    # Test system prompt generation
    print("\n2. Generating combined system prompt...")
    system_prompt = prompt_loader.get_system_prompt()
    print(f"   ✅ Generated system prompt: {len(system_prompt):,} characters")
    
    # Simulate LLM request processing
    print("\n3. Simulating LLM request processing...")
    user_input = "I need help with my debt consolidation options"
    
    # This is what happens inside the LLM service
    enhanced_context = system_prompt
    full_prompt = f"{enhanced_context}\n\nUser: {user_input}\nAssistant:"
    
    print(f"   📝 User input: '{user_input}'")
    print(f"   🎯 Full prompt length: {len(full_prompt):,} characters")
    print(f"   📊 System prompt portion: {len(system_prompt):,} chars ({len(system_prompt)/len(full_prompt)*100:.1f}%)")
    
    # Show prompt structure
    print("\n4. System prompt structure:")
    lines = system_prompt.split('\n')
    for i, line in enumerate(lines[:10]):  # Show first 10 lines
        print(f"   {line}")
    if len(lines) > 10:
        print(f"   ... ({len(lines)-10} more lines)")
    
    print("\n✅ Test complete!")
    print("\n💡 What happens in production:")
    print("   1. User sends request to LLM service")
    print("   2. Service loads system prompts from docs/prompts/")
    print("   3. System prompt is prepended to user input")
    print("   4. Enhanced prompt is sent to LLM model")
    print("   5. LLM generates response following prompt instructions")
    
    print("\n🎯 Expected behavior with current prompts:")
    print("   - LLM will identify as 'Alex' from 'Finally Payoff Debt'")
    print("   - Responses will follow banking/loan specialist format")
    print("   - Customer service tone will be helpful and professional")
    print("   - Qualifying questions will be asked for loan applications")

if __name__ == "__main__":
    main()
