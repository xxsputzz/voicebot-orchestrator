#!/usr/bin/env python3
"""
TTS Request Interceptor - Debug what text is actually being sent to TTS
This will help identify the source of emoji characters in TTS requests
"""

import requests
import json
import time
from datetime import datetime

def intercept_tts_requests():
    """Monitor TTS requests to see what text contains emojis"""
    
    print("🔍 TTS Request Interceptor - Debugging Emoji Sources")
    print("=" * 60)
    
    # Sample texts that might come from LLM responses
    test_cases = [
        # Test if the issue is in the LLM responses
        "I'm here to help with your banking needs. How can I assist you today?",
        "I can help you check your account balance. Please verify your identity first.",
        "Let me help you with that transfer. 🎤 What accounts would you like to use?",  # Intentional emoji test
        "Your balance is $1,234.56. Is there anything else I can help you with?",
        "✅ Your transaction has been processed successfully!",  # Another emoji test
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: Analyzing text content")
        print(f"📝 Original text: '{text}'")
        
        # Check for Unicode characters
        unicode_chars = []
        for char in text:
            if ord(char) > 127:  # Non-ASCII characters
                unicode_chars.append(f"'{char}' (U+{ord(char):04X})")
        
        if unicode_chars:
            print(f"⚠️  Unicode characters found: {', '.join(unicode_chars)}")
        else:
            print("✓ No Unicode characters detected")
        
        # Test the actual TTS request
        try:
            payload = {
                "text": text,
                "return_audio": False  # Don't return audio, just test processing
            }
            
            print(f"📤 Sending to TTS service...")
            response = requests.post(
                "http://localhost:8012/synthesize",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"✅ TTS accepted text successfully")
                result = response.json()
                metadata = result.get('metadata', {})
                print(f"⏱️  Processing time: {metadata.get('processing_time_seconds', 'unknown')}s")
            else:
                print(f"❌ TTS rejected text: {response.status_code}")
                print(f"💬 Error details: {response.text}")
                
                # Check if it's the specific emoji encoding error
                if "'charmap' codec can't encode character" in response.text:
                    print("🎯 FOUND THE ISSUE: Emoji encoding error in TTS request!")
                    # Extract the problematic character
                    import re
                    match = re.search(r"'\\\\U([0-9a-fA-F]{8})'", response.text)
                    if match:
                        unicode_code = match.group(1)
                        char_code = int(unicode_code, 16)
                        problematic_char = chr(char_code)
                        print(f"🐛 Problematic character: '{problematic_char}' (U+{unicode_code})")
                
        except Exception as e:
            print(f"💥 Request failed: {e}")
        
        print("-" * 40)

def monitor_live_requests():
    """Monitor what's happening in real TTS traffic"""
    print(f"\n🔄 Live Request Monitor")
    print("This would require instrumenting the TTS service to log incoming requests...")
    print("For now, check the TTS service logs for incoming text content.")

if __name__ == "__main__":
    intercept_tts_requests()
    
    print(f"\n📊 Analysis Complete")
    print("🎯 Key findings to check:")
    print("1. Are LLM responses including emojis in the text content?")
    print("2. Is the orchestrator adding emojis to responses before sending to TTS?")
    print("3. Are there emojis in system prompts or pre-configured responses?")
    print("\n💡 Next steps:")
    print("1. Check LLM service responses for emoji content")
    print("2. Check orchestrator message processing")
    print("3. Review system prompts and default responses")
