"""
Service Startup and Shutdown Test
Demonstrates the new service management features
"""
import asyncio
import sys
import os

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from voicebot_orchestrator.modular_cli import VoicebotServices

async def test_service_lifecycle():
    """Test service startup and shutdown lifecycle"""
    
    services = VoicebotServices()
    
    print("🚀 SERVICE LIFECYCLE TEST")
    print("="*50)
    
    # Initial status
    print("\n1️⃣ Initial Status:")
    status = services.get_status()
    for service, state in status.items():
        print(f"   {service}: {state}")
    
    # Test TTS startup
    print("\n2️⃣ Testing TTS Kokoro startup...")
    await services.initialize_tts('kokoro')
    
    status = services.get_status()
    print("   Status after TTS startup:")
    for service, state in status.items():
        print(f"   {service}: {state}")
    
    # Test LLM startup (Mistral)
    print("\n3️⃣ Testing LLM (Mistral) startup...")
    await services.initialize_llm('mistral')
    
    status = services.get_status()
    print("   Status after LLM startup:")
    for service, state in status.items():
        print(f"   {service}: {state}")
    
    # Test LLM switching (GPT-OSS)
    print("\n4️⃣ Testing LLM switch to GPT-OSS...")
    await services.initialize_llm('gpt-oss')
    
    status = services.get_status()
    print("   Status after LLM switch:")
    for service, state in status.items():
        print(f"   {service}: {state}")
    
    # Test selective shutdown
    print("\n5️⃣ Testing TTS shutdown...")
    await services.shutdown_tts()
    
    status = services.get_status()
    print("   Status after TTS shutdown:")
    for service, state in status.items():
        print(f"   {service}: {state}")
    
    # Test full cleanup
    print("\n6️⃣ Testing full cleanup...")
    await services.cleanup_all()
    
    status = services.get_status()
    print("   Status after cleanup:")
    for service, state in status.items():
        print(f"   {service}: {state}")
    
    print("\n✅ Service lifecycle test complete!")

if __name__ == "__main__":
    asyncio.run(test_service_lifecycle())
