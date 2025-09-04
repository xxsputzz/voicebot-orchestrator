#!/usr/bin/env python3
"""
Test script for the Real Tortoise TTS Service
"""
import requests
import base64
import json
import time

def test_tortoise_service():
    """Test the real Tortoise TTS service"""
    base_url = "http://localhost:8016"
    
    print("🧪 Testing Real Tortoise TTS Service...")
    
    # Test 1: Check if service is alive
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Get available voices
    try:
        response = requests.get(f"{base_url}/voices")
        voices_data = response.json()
        print(f"✅ Available voices: {len(voices_data['voices'])} voices")
        print(f"   First 5 voices: {voices_data['voices'][:5]}")
    except Exception as e:
        print(f"❌ Voices check failed: {e}")
        return
    
    # Test 3: Test speech synthesis
    test_text = "Hello, this is a test of the real Tortoise TTS system. The quality should be much better now!"
    
    for voice in ['angie', 'freeman', 'halle']:
        try:
            print(f"\n🔄 Testing voice '{voice}'...")
            start_time = time.time()
            
            response = requests.post(f"{base_url}/synthesize", json={
                "text": test_text,
                "voice": voice,
                "preset": "fast"
            })
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                audio_b64 = result.get('audio_base64', '')
                
                if audio_b64:
                    # Save audio file
                    audio_bytes = base64.b64decode(audio_b64)
                    filename = f"real_tortoise_test_{voice}.wav"
                    
                    with open(filename, 'wb') as f:
                        f.write(audio_bytes)
                    
                    print(f"✅ Voice '{voice}' test successful!")
                    print(f"   Generation time: {end_time - start_time:.2f} seconds")
                    print(f"   Audio file saved: {filename}")
                    print(f"   Audio size: {len(audio_bytes)} bytes")
                else:
                    print(f"❌ Voice '{voice}' test failed: No audio data")
            else:
                print(f"❌ Voice '{voice}' test failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Voice '{voice}' test error: {e}")
    
    print("\n🎉 Real Tortoise TTS Service test completed!")

if __name__ == "__main__":
    test_tortoise_service()
