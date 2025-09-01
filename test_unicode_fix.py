#!/usr/bin/env python3
"""
Test script to verify Unicode character handling in TTS synthesis
"""
import sys
import os
import asyncio
sys.path.append('.')
sys.path.append('./voicebot_orchestrator')

from voicebot_orchestrator.enhanced_tts_manager import EnhancedTTSManager

async def test_unicode_handling():
    """Test the text sanitization functionality"""
    print("🧪 Testing Unicode Character Handling for TTS")
    print("=" * 50)
    
    # Create TTS manager instance
    manager = EnhancedTTSManager()
    
    # Test cases with problematic Unicode characters
    test_cases = [
        "🎤 Hungrey",  # The actual failing case
        "Hello 🎵 World",
        "Testing emoji 🚀 in text",
        "Multiple emojis: 🎤🎵🔊🎧",
        "Banking symbols: 💰🏦💳💵",
        "Mixed: Hello 🌐 World! 💻📱",
        "Regular text without emojis",
        "Text with accents: café naïve résumé",
        "",  # Empty text
        "🎭🎭🎭",  # Only emojis
    ]
    
    print("Testing text sanitization:")
    print("-" * 30)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"  Original: \"{test_text}\"")
        
        try:
            sanitized = manager.sanitize_text_for_synthesis(test_text)
            print(f"  Sanitized: \"{sanitized}\"")
            
            # Test if the sanitized text can be encoded properly
            try:
                encoded = sanitized.encode('ascii')
                print(f"  ✅ ASCII encoding: OK")
            except UnicodeEncodeError as e:
                print(f"  ❌ ASCII encoding failed: {e}")
                
        except Exception as e:
            print(f"  ❌ Sanitization failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Unicode handling test completed!")

if __name__ == "__main__":
    asyncio.run(test_unicode_handling())
