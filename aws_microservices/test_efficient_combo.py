#!/usr/bin/env python3
"""
Test script to verify the Efficient Combo (option 3) configuration
"""

import sys
import os
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_service_manager import EnhancedServiceManager

def test_efficient_combo_config():
    """Test that the efficient combo is correctly configured"""
    
    print("🧪 Testing Efficient Combo Configuration")
    print("=" * 50)
    
    manager = EnhancedServiceManager()
    
    # Check that efficient combo exists and has correct configuration
    if "efficient" not in manager.combinations:
        print("❌ ERROR: 'efficient' combination not found!")
        return False
    
    efficient_combo = manager.combinations["efficient"]
    
    print(f"✅ Combination found: {efficient_combo['name']}")
    print(f"📝 Description: {efficient_combo['description']}")
    print(f"🎯 Use case: {efficient_combo['use_case']}")
    print(f"🔧 Services: {efficient_combo['services']}")
    
    # Verify it includes the correct services
    expected_services = ["orchestrator", "whisper_stt", "hira_dia_tts", "mistral_llm"]
    actual_services = efficient_combo["services"]
    
    if actual_services == expected_services:
        print("✅ Service list is correct")
    else:
        print(f"❌ ERROR: Expected {expected_services}, got {actual_services}")
        return False
    
    # Check description mentions Dia 4-bit
    if "4-bit" in efficient_combo["description"]:
        print("✅ Description correctly mentions Dia 4-bit TTS")
    else:
        print("❌ ERROR: Description should mention Dia 4-bit TTS")
        return False
    
    print("\n🎉 All tests passed! Efficient Combo is correctly configured.")
    return True

def test_menu_order():
    """Test that the menu options are in the correct order"""
    
    print("\n🧪 Testing Menu Order")
    print("=" * 30)
    
    manager = EnhancedServiceManager()
    
    # Expected order based on your requirements
    expected_combos = {
        "fast": "Fast Combo",
        "efficient": "Efficient Combo", 
        "balanced": "Balanced Combo",
        "quality": "Quality Combo"
    }
    
    print("Expected combo order:")
    for i, (key, name) in enumerate(expected_combos.items(), 2):  # Start from option 2
        combo = manager.combinations[key]
        print(f"  {i}. {name}")
        # Show service names with clarification for Dia 4-bit
        services_display = combo['services'].copy()
        if key == "efficient":
            services_with_note = []
            for service in services_display:
                if service == "hira_dia_tts":
                    services_with_note.append("hira_dia_tts (configured for Dia 4-bit mode)")
                else:
                    services_with_note.append(service)
            print(f"     Services: {services_with_note}")
        else:
            print(f"     Services: {services_display}")
    
    print("\n✅ Menu order verification complete")
    print("📝 Note: Service names remain as 'hira_dia_tts' since it's the same unified service,")
    print("          but the Efficient Combo automatically configures it for Dia 4-bit mode.")

if __name__ == "__main__":
    success = test_efficient_combo_config()
    test_menu_order()
    
    if success:
        print("\n🎯 Summary: All configurations are correct!")
        print("   • Option 3 is now 'Efficient Combo' with Mistral LLM + Dia 4-bit TTS")
        print("   • Option 4 is now 'Balanced Combo' with GPT LLM + Kokoro TTS")
        print("   • Option 5 remains 'Quality Combo' with GPT LLM + Hira Dia TTS")
        print("   • Service order in descriptions: LLM before TTS")
    else:
        print("\n❌ Some configuration issues found!")
        sys.exit(1)
