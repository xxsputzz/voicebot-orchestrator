#!/usr/bin/env python3
"""
Test TTS with output suppression to isolate emoji contamination
"""
import requests
import json
import sys
import io
import os
from contextlib import redirect_stdout, redirect_stderr

def test_tts_with_output_suppression():
    """Test TTS while suppressing all output that might contain emojis"""
    print("🧪 TESTING TTS WITH OUTPUT SUPPRESSION")
    print("=" * 50)
    
    # Clean ASCII text
    clean_text = "I understand you're asking about: steal ball bearings. How can I help you with this?"
    
    print(f"📝 Input text: {repr(clean_text)}")
    print(f"✅ Input is clean ASCII: {all(ord(char) <= 127 for char in clean_text)}")
    
    # Test with complete output suppression
    print("\n🔄 Testing TTS with output suppression...")
    
    try:
        # Suppress all output during TTS request
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                response = requests.post(
                    'http://localhost:8012/synthesize',
                    json={
                        'text': clean_text,
                        'voice': 'af_bella',
                        'return_audio': True
                    },
                    timeout=30
                )
        
        print(f"🌐 Response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ TTS request succeeded with output suppression!")
            result = response.json()
            print(f"📊 Metadata: {result.get('metadata', {})}")
        else:
            print(f"❌ TTS request failed: {response.status_code}")
            print(f"💥 Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_tts_with_output_suppression()
