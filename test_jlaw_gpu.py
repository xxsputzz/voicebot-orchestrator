#!/usr/bin/env python3
"""
Test jlaw voice with GPU-accelerated Tortoise TTS
"""

import requests
import time
import json

def test_jlaw_gpu():
    print("🚀 Testing jlaw voice with GPU-accelerated Tortoise TTS...")
    
    # Wait for service to be ready
    time.sleep(10)
    
    try:
        # Check service health
        print("📊 Checking service health...")
        health_response = requests.get('http://localhost:8015/health', timeout=10)
        print(f"Service status: {health_response.status_code}")
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            voices = health_data.get('voices', [])
            engine = health_data.get('engine', 'unknown')
            
            print(f"Engine: {engine}")
            print(f"Voices available: {len(voices)}")
            
            if 'jlaw' in voices:
                print("✅ jlaw voice found in service!")
                
                # Test synthesis with jlaw voice
                synthesis_data = {
                    'text': 'Hello, this is the fixed jlaw voice speaking with GPU acceleration. The voice conditioning has been regenerated and is working perfectly.',
                    'voice': 'jlaw',
                    'preset': 'fast',
                    'return_audio': True
                }
                
                print("🎙️ Starting jlaw voice synthesis...")
                print(f"Text: {synthesis_data['text'][:50]}...")
                
                synthesis_response = requests.post(
                    'http://localhost:8015/synthesize',
                    json=synthesis_data,
                    timeout=180  # Allow extra time for first synthesis
                )
                
                if synthesis_response.status_code == 200:
                    result = synthesis_response.json()
                    metadata = result['metadata']
                    
                    print("✅ jlaw synthesis successful!")
                    print(f"   Voice: {metadata['voice']}")
                    print(f"   Duration: {metadata['duration']:.2f} seconds")
                    print(f"   Sample rate: {metadata['sample_rate']} Hz")
                    print(f"   Engine: {metadata['engine']}")
                    
                    if 'audio_file' in result:
                        print(f"   Audio saved: {result['audio_file']}")
                    
                    print("🎉 jlaw voice is working perfectly with GPU acceleration!")
                    return True
                    
                else:
                    print(f"❌ Synthesis failed: HTTP {synthesis_response.status_code}")
                    print(f"Error: {synthesis_response.text}")
                    return False
                    
            else:
                print("⚠️ jlaw voice not found in available voices")
                print(f"Available voices: {voices[:10]}...")  # Show first 10
                return False
                
        else:
            print(f"❌ Service not ready: HTTP {health_response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the Tortoise TTS service is running on port 8015")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_jlaw_gpu()
    if success:
        print("\n🎯 jlaw voice GPU acceleration test: PASSED ✅")
    else:
        print("\n🎯 jlaw voice GPU acceleration test: FAILED ❌")
