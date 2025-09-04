#!/usr/bin/env python3
"""Test Zonos TTS endpoints to verify format consistency."""

import requests
import json
import base64

def test_zonos_endpoints():
    """Test all Zonos TTS endpoints for proper formats."""
    base_url = "http://localhost:8014"
    
    print("=== Testing Zonos TTS Endpoints ===")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health endpoint: {response.status_code}")
        print(f"Health response: {response.json()}")
    except Exception as e:
        print(f"Health endpoint error: {e}")
    
    # Test voices endpoint
    try:
        response = requests.get(f"{base_url}/voices")
        voices = response.json()
        print(f"\nVoices endpoint: {response.status_code}")
        print(f"Voices type: {type(voices)}")
        print(f"Voices content: {voices}")
        if isinstance(voices, list) and all(isinstance(v, str) for v in voices):
            print("✅ Voices format is correct (list of strings)")
        else:
            print("❌ Voices format is incorrect")
    except Exception as e:
        print(f"Voices endpoint error: {e}")
    
    # Test models endpoint
    try:
        response = requests.get(f"{base_url}/models")
        models = response.json()
        print(f"\nModels endpoint: {response.status_code}")
        print(f"Models type: {type(models)}")
        print(f"Models content: {models}")
        if isinstance(models, list) and all(isinstance(m, str) for m in models):
            print("✅ Models format is correct (list of strings)")
        else:
            print("❌ Models format is incorrect")
    except Exception as e:
        print(f"Models endpoint error: {e}")
    
    # Test synthesis endpoint
    try:
        synthesis_data = {
            "text": "Hello, this is a test of Zonos TTS synthesis.",
            "voice": "default",
            "model": "zonos-v1",
            "emotion": "neutral",
            "seed": 42
        }
        
        response = requests.post(f"{base_url}/synthesize", json=synthesis_data)
        print(f"\nSynthesis endpoint: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Synthesis response keys: {list(result.keys())}")
            
            if 'audio_data' in result:
                audio_data = result['audio_data']
                if audio_data and len(audio_data) > 0:
                    try:
                        # Try to decode base64 to verify it's valid
                        decoded = base64.b64decode(audio_data)
                        print(f"✅ Audio data present: {len(audio_data)} chars, {len(decoded)} bytes")
                    except Exception as decode_error:
                        print(f"❌ Audio data invalid base64: {decode_error}")
                else:
                    print("❌ Audio data is empty")
            else:
                print("❌ No audio_data in response")
                
            # Show sample of response
            print(f"Sample response: {str(result)[:200]}...")
        else:
            print(f"Synthesis failed: {response.text}")
            
    except Exception as e:
        print(f"Synthesis endpoint error: {e}")

if __name__ == "__main__":
    test_zonos_endpoints()
