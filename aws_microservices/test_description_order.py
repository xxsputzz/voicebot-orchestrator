#!/usr/bin/env python3
"""
Quick test to verify the combination descriptions show correctly during startup
"""

import sys
import os
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_service_manager import EnhancedServiceManager

def test_combination_descriptions():
    """Test that combination descriptions show the correct LLM + TTS order"""
    
    print("🧪 Testing Combination Description Display")
    print("=" * 50)
    
    manager = EnhancedServiceManager()
    
    # Test each combination's description as it would appear during startup
    for combo_name, combo_data in manager.combinations.items():
        print(f"\n🚀 {combo_data['name']}")
        print(f"   Description: {combo_data['description']}")
        print(f"   Use case: {combo_data['use_case']}")
        
        # Verify LLM comes before TTS in description
        desc = combo_data['description']
        if 'LLM' in desc and 'TTS' in desc:
            llm_pos = desc.find('LLM')
            tts_pos = desc.find('TTS')
            if llm_pos < tts_pos:
                print("   ✅ LLM appears before TTS in description")
            else:
                print("   ❌ TTS appears before LLM in description")
        else:
            print("   ℹ️  Description format check skipped")
    
    print("\n🎉 Description order verification complete!")

if __name__ == "__main__":
    test_combination_descriptions()
