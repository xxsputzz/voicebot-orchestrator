#!/usr/bin/env python3
"""
Quick test to verify the service name cleanup logic
"""

import sys
from pathlib import Path

# Add the aws_microservices directory to path
sys.path.append(str(Path(__file__).parent))

from enhanced_service_manager import EnhancedServiceManager

def test_service_names():
    print("🧪 Testing Service Name Cleanup")
    print("=" * 50)
    
    manager = EnhancedServiceManager()
    
    # Get service status
    status = manager.get_service_status(fast_mode=True)
    
    print("📋 Current Service Names:")
    print("-" * 30)
    
    for service_name, info in status.items():
        # Apply the same logic as in show_status
        clean_name = info['description']
        
        # Clean up service names for consistent display
        service_name_map = {
            "orchestrator": "Orchestrator",
            "whisper_stt": "Whisper STT", 
            "kokoro_tts": "Kokoro TTS",
            "hira_dia_tts": "Hira Dia TTS",  # Default, may be overridden below
            "mistral_llm": "Mistral LLM",
            "gpt_llm": "GPT LLM"
        }
        
        # Use mapped name if available, otherwise clean the description
        if service_name in service_name_map:
            clean_name = service_name_map[service_name]
            
            # Special handling for Hira Dia TTS based on intended engine mode
            if service_name == "hira_dia_tts":
                intended_mode = manager.intended_engine_mode.get(service_name, "full")
                if intended_mode == "4bit":
                    clean_name = "Dia 4-bit TTS"
                else:
                    clean_name = "Hira Dia TTS"
        else:
            # Fallback: clean the description text
            clean_name = clean_name.replace(" Service", "").replace(" (OpenAI)", "").replace(" (Fast)", "").replace(" (High Quality)", "")
            clean_name = clean_name.replace(" (Quality)", "").replace(" (Speed)", "").replace("🎭 ", "").replace("⚡ ", "")
            clean_name = clean_name.replace(" (Quality + Speed)", "").replace("(FastAPI)", "").replace("Unified ", "")
        
        print(f"  {service_name}: '{info['description']}' → '{clean_name}'")
    
    print("\n✅ Service name cleanup test complete!")

if __name__ == "__main__":
    test_service_names()
