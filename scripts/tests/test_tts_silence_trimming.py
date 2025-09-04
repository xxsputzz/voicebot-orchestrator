#!/usr/bin/env python3
"""
Test script to verify TTS silence trimming functionality
"""

import requests
import json
import sys

def test_tts_silence_trimming():
    """Test the TTS service with silence trimming"""
    
    # Test with a short phrase
    test_text = "Hello world, this is a test of silence trimming."
    
    print(f"🧪 Testing TTS with text: '{test_text}'")
    
    try:
        # Send request to TTS service
        response = requests.post(
            'http://localhost:8012/synthesize',
            json={
                'text': test_text,
                'high_quality': True,
                'return_audio': True
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ TTS request successful")
            print(f"📊 Audio file: {result.get('file_path', 'Not provided')}")
            print(f"🎵 Duration: {result.get('duration', 'Not provided')}s")
            print(f"⚡ Generation time: {result.get('generation_time', 'Not provided')}s")
            
            if 'audio_base64' in result:
                print(f"📦 Audio data included: {len(result['audio_base64'])} characters")
            
            return True
        else:
            print(f"❌ TTS request failed: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🎤 TTS Silence Trimming Test")
    print("=" * 50)
    
    # Check if service is running
    try:
        health_response = requests.get('http://localhost:8012/status', timeout=5)
        if health_response.status_code == 200:
            print("✅ TTS service is running")
        else:
            print("⚠️ TTS service may not be healthy")
    except:
        print("❌ TTS service is not running. Please start it first.")
        sys.exit(1)
    
    # Run the test
    success = test_tts_silence_trimming()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
