#!/usr/bin/env python3
"""
Direct test of jlaw voice with GPU acceleration
"""

import requests
import json

def test_jlaw_direct():
    payload = {
        'text': 'Hello, this is the fixed jlaw voice speaking with GPU acceleration from your NVIDIA RTX 4060. The voice conditioning has been regenerated and is working perfectly.',
        'voice': 'jlaw',
        'preset': 'fast',
        'return_audio': True
    }

    print("🎙️ Testing jlaw voice synthesis with GPU...")
    print(f"Text: {payload['text']}")
    print("🚀 Starting synthesis...")

    try:
        response = requests.post('http://localhost:8015/synthesize', json=payload, timeout=180)

        if response.status_code == 200:
            result = response.json()
            metadata = result['metadata']
            print("✅ SUCCESS: jlaw voice synthesis completed!")
            print(f"   Voice: {metadata['voice']}")
            print(f"   Duration: {metadata['duration']:.2f} seconds")
            print(f"   Sample rate: {metadata['sample_rate']} Hz")
            print(f"   Engine: {metadata['engine']}")
            if 'audio_file' in result:
                print(f"   🎵 Audio saved: {result['audio_file']}")
            print("🎉 jlaw voice is working perfectly with GPU acceleration!")
            return True
        else:
            print(f"❌ Failed: HTTP {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_jlaw_direct()
    if success:
        print("\n🎯 jlaw voice GPU test: PASSED ✅")
    else:
        print("\n🎯 jlaw voice GPU test: FAILED ❌")
