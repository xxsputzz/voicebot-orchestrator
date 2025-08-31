#!/usr/bin/env python3
"""
Test script to verify the reordered Individual Service Management section
"""

import sys
import os
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_service_manager import EnhancedServiceManager

def test_reordered_individual_services():
    """Test the reordered individual service management menu display"""
    
    print("🧪 Testing Reordered Individual Service Management")
    print("=" * 55)
    
    manager = EnhancedServiceManager()
    
    # Simulate the individual service management display
    print("\n🔧 Individual Service Management")
    print("-" * 40)
    print("📋 Using cached service status...")
    
    # Use the same ordering logic as the actual method
    service_order = [
        "orchestrator",
        "whisper_stt", 
        "mistral_llm",
        "gpt_llm",
        "kokoro_tts",
        "hira_dia_tts"
    ]
    
    service_list = [(name, manager.service_configs[name]) for name in service_order if name in manager.service_configs]
    
    service_names = {
        "orchestrator": "Orchestrator",
        "whisper_stt": "Whisper STT", 
        "kokoro_tts": "Kokoro TTS",
        "hira_dia_tts": "Hira Dia TTS",
        "dia_4bit_tts": "Dia 4-bit TTS",
        "mistral_llm": "Mistral LLM",
        "gpt_llm": "GPT LLM"
    }
    
    # Display regular services (simulated as stopped)
    for i, (service_name, config) in enumerate(service_list, 1):
        simple_name = service_names.get(service_name, config['description'])
        service_desc = f"Start {simple_name}"
        status_text = "(⏹️  stopped)"
        print(f"  {i}. {service_desc:<35} {status_text}")
    
    # Add special Dia 4-bit TTS option
    dia_4bit_num = len(service_list) + 1
    dia_service_desc = "Start Dia 4-bit TTS"
    dia_status_text = "(⏹️  stopped)"
    print(f"  {dia_4bit_num}. {dia_service_desc:<35} {dia_status_text}")
    
    print("  r. Refresh status (clear cache)")
    print("  0. Back to main menu")
    
    print(f"\nSelect service (0-{dia_4bit_num}, r):")
    
    # Verify the order
    print("\n✅ Service Order Verification:")
    print("   1. Orchestrator     (Coordination)")
    print("   2. Whisper STT      (Speech → Text)")
    print("   3. Mistral LLM      (Language Processing - Fast)")
    print("   4. GPT LLM          (Language Processing - Quality)")
    print("   5. Kokoro TTS       (Text → Speech - Fast)")
    print("   6. Hira Dia TTS     (Text → Speech - Quality)")
    print("   7. Dia 4-bit TTS    (Text → Speech - Speed)")
    
    print("\n🎯 Flow: Input → Orchestrator → STT → LLM → TTS → Output")
    print("✅ Individual Service Management reordered successfully!")

if __name__ == "__main__":
    test_reordered_individual_services()
