#!/usr/bin/env python3
"""
Prompt Management Utility
Manage prompts for LLM services from the command line
"""
import sys
import os
from pathlib import Path
import requests
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from aws_microservices.prompt_loader import prompt_loader

def show_help():
    """Show help information"""
    print("🎯 LLM Prompt Management Utility")
    print("=" * 40)
    print("Commands:")
    print("  list       - List all available prompts")
    print("  show <name>- Show content of a specific prompt")
    print("  reload     - Reload prompts from disk")
    print("  test       - Test prompt loading")
    print("  services   - Check LLM services and reload their prompts")
    print("  help       - Show this help")
    print("\nExamples:")
    print("  python prompt_manager.py list")
    print("  python prompt_manager.py show prompt-main")
    print("  python prompt_manager.py reload")
    print("  python prompt_manager.py services")

def list_prompts():
    """List all available prompts"""
    print("📝 Available Prompts")
    print("-" * 20)
    
    prompts = prompt_loader.load_all_prompts()
    
    if not prompts:
        print("❌ No prompt files found in docs/prompts/")
        print(f"📁 Looking in: {prompt_loader.get_prompts_directory()}")
        return
    
    for name, content in prompts.items():
        print(f"  📄 {name:<20} ({len(content):,} characters)")
    
    print(f"\n📁 Prompts directory: {prompt_loader.get_prompts_directory()}")
    print(f"📊 Total prompts: {len(prompts)}")

def show_prompt(prompt_name):
    """Show content of a specific prompt"""
    print(f"📄 Prompt: {prompt_name}")
    print("=" * 40)
    
    prompts = prompt_loader.load_all_prompts()
    
    if prompt_name not in prompts:
        print(f"❌ Prompt '{prompt_name}' not found")
        print(f"Available prompts: {', '.join(prompts.keys())}")
        return
    
    content = prompts[prompt_name]
    print(f"📊 Length: {len(content):,} characters")
    print(f"📄 Content:\n")
    print(content)

def reload_prompts():
    """Reload prompts from disk"""
    print("🔄 Reloading prompts from disk...")
    
    prompts = prompt_loader.reload_prompts()
    
    print(f"✅ Reloaded {len(prompts)} prompt files")
    for name in prompts.keys():
        print(f"  📄 {name}")

def test_prompts():
    """Test prompt loading and system prompt generation"""
    print("🧪 Testing Prompt System")
    print("=" * 30)
    
    # Test 1: Load all prompts
    print("1. Loading all prompts...")
    prompts = prompt_loader.load_all_prompts()
    print(f"   ✅ Loaded {len(prompts)} prompts")
    
    # Test 2: Generate system prompt
    print("2. Generating system prompt...")
    system_prompt = prompt_loader.get_system_prompt()
    print(f"   ✅ System prompt: {len(system_prompt):,} characters")
    
    # Test 3: Show preview
    print("3. System prompt preview:")
    preview = system_prompt[:300] + "..." if len(system_prompt) > 300 else system_prompt
    print(f"   {preview}")
    
    print("\n✅ Prompt system test complete!")

def check_services():
    """Check LLM services and reload their prompts"""
    print("🔍 Checking LLM Services")
    print("=" * 30)
    
    services = {
        "Mistral LLM": "http://localhost:8021",
        "GPT LLM": "http://localhost:8022"
    }
    
    for service_name, base_url in services.items():
        print(f"\n🧠 {service_name}")
        print("-" * 15)
        
        # Check health
        try:
            health_response = requests.get(f"{base_url}/health", timeout=3)
            if health_response.status_code == 200:
                print("  ✅ Service is running")
            else:
                print(f"  ❌ Service unhealthy (Status: {health_response.status_code})")
                continue
        except requests.exceptions.RequestException:
            print("  ❌ Service not accessible")
            continue
        
        # Get prompts info
        try:
            prompts_response = requests.get(f"{base_url}/prompts", timeout=5)
            if prompts_response.status_code == 200:
                info = prompts_response.json()
                print(f"  📝 Prompts loaded: {info['total_prompts']}")
                print(f"  📊 System prompt: {info['system_prompt_length']:,} chars")
            else:
                print("  ❌ Could not get prompts info")
                continue
        except requests.exceptions.RequestException:
            print("  ❌ Could not get prompts info")
            continue
        
        # Reload prompts
        try:
            reload_response = requests.post(f"{base_url}/prompts/reload", timeout=10)
            if reload_response.status_code == 200:
                result = reload_response.json()
                print(f"  🔄 Prompts reloaded: {result['total_prompts']} files")
            else:
                print("  ❌ Could not reload prompts")
        except requests.exceptions.RequestException:
            print("  ❌ Could not reload prompts")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "help":
        show_help()
    elif command == "list":
        list_prompts()
    elif command == "show":
        if len(sys.argv) < 3:
            print("❌ Please specify a prompt name")
            print("Usage: python prompt_manager.py show <prompt_name>")
        else:
            show_prompt(sys.argv[2])
    elif command == "reload":
        reload_prompts()
    elif command == "test":
        test_prompts()
    elif command == "services":
        check_services()
    else:
        print(f"❌ Unknown command: {command}")
        show_help()

if __name__ == "__main__":
    main()
