"""
Service Startup Example Script
Shows how to programmatically initialize services
"""
import asyncio
import sys
import os

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from voicebot_orchestrator.modular_cli import VoicebotServices

async def startup_services_example():
    """Example of programmatically starting services"""
    
    # Create service manager
    services = VoicebotServices()
    
    print("🚀 Service Startup Example")
    print("="*40)
    
    # Method 1: Initialize individual services
    print("\n1️⃣ Initializing STT...")
    await services.initialize_stt()
    
    print("\n2️⃣ Initializing LLM...")
    await services.initialize_llm()
    
    print("\n3️⃣ Initializing TTS Kokoro...")
    await services.initialize_tts('kokoro')
    
    # Show status
    print("\n📊 Current Status:")
    status = services.get_status()
    for service, state in status.items():
        print(f"   {service}: {state}")
    
    # Optional: Initialize Nari Dia as well
    user_choice = input("\n🎭 Also initialize Nari Dia TTS? (y/N): ").strip().lower()
    if user_choice == 'y':
        print("\n4️⃣ Initializing TTS Nari Dia...")
        await services.initialize_tts('nari_dia')
    
    print("\n✅ Service initialization complete!")
    
    # Cleanup when done
    print("\n🧹 Cleaning up...")
    services.cleanup()
    print("✅ Cleanup complete!")

if __name__ == "__main__":
    asyncio.run(startup_services_example())
