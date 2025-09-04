#!/usr/bin/env python3
"""
Quick Zonos TTS Test
===================

Simple test to verify Zonos TTS service is working properly
"""

import requests
import json
import base64
import time
from pathlib import Path

def test_zonos_tts():
    """Test Zonos TTS service with a simple request"""
    
    zonos_url = "http://localhost:8014"
    
    print("🧠 Testing Zonos TTS Service")
    print("=" * 40)
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{zonos_url}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Health check: {health_data.get('status', 'unknown')}")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        return False
    
    # Test voices endpoint
    try:
        voices_response = requests.get(f"{zonos_url}/voices", timeout=5)
        if voices_response.status_code == 200:
            voices_data = voices_response.json()
            voices = voices_data.get('voices', [])
            print(f"🎭 Available voices: {len(voices)} found")
            
            # Check if voices is a list of strings (new format) or objects (old format)
            if voices and isinstance(voices[0], str):
                print(f"   ✅ New format: {voices}")
            else:
                print(f"   ⚠️ Old format detected - need service restart")
                # Extract voice IDs for compatibility
                if isinstance(voices[0], dict):
                    voices = [voice.get('id', 'unknown') for voice in voices]
                    print(f"   🔄 Extracted IDs: {voices}")
        else:
            print(f"❌ Voices check failed: {voices_response.status_code}")
            voices = ['default']
    except Exception as e:
        print(f"⚠️ Voices check error: {e}")
        voices = ['default']
    
    # Test models endpoint
    try:
        models_response = requests.get(f"{zonos_url}/models", timeout=5)
        if models_response.status_code == 200:
            models_data = models_response.json()
            models = models_data.get('models', [])
            print(f"🤖 Available models: {len(models)} found")
            
            # Check format
            if models and isinstance(models[0], str):
                print(f"   ✅ New format: {models}")
            else:
                print(f"   ⚠️ Old format detected - need service restart")
                if isinstance(models[0], dict):
                    models = [model.get('id', 'unknown') for model in models]
                    print(f"   🔄 Extracted IDs: {models}")
        else:
            print(f"❌ Models check failed: {models_response.status_code}")
            models = ['zonos-v1']
    except Exception as e:
        print(f"⚠️ Models check error: {e}")
        models = ['zonos-v1']
    
    # Test synthesis
    print(f"\n🔄 Testing synthesis...")
    
    test_text = "Hello! This is a test of Zonos neural TTS synthesis. The quick brown fox jumps over the lazy dog."
    
    synthesis_request = {
        "text": test_text,
        "voice": voices[0] if voices else "default",
        "model": models[0] if models else "zonos-v1",
        "emotion": "neutral",
        "seed": 12345,
        "format": "wav"
    }
    
    print(f"📊 Request: {synthesis_request['voice']}/{synthesis_request['model']}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{zonos_url}/synthesize",
            json=synthesis_request,
            timeout=30
        )
        
        synthesis_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            if 'audio_data' in result:
                # Save audio
                audio_bytes = base64.b64decode(result['audio_data'])
                filename = f"zonos_test_{int(time.time())}.wav"
                
                with open(filename, 'wb') as f:
                    f.write(audio_bytes)
                
                duration = result.get('duration', 0)
                print(f"✅ Synthesis successful!")
                print(f"   📊 Duration: {duration:.2f}s")
                print(f"   ⏱️ Generation time: {synthesis_time:.2f}s")
                print(f"   🚀 Speed: {duration/synthesis_time:.2f}x real-time")
                print(f"   💾 Saved: {filename}")
                return True
            else:
                print(f"❌ No audio data in response")
                return False
        else:
            print(f"❌ Synthesis failed: {response.status_code}")
            try:
                error = response.json()
                print(f"   Error: {error.get('detail', 'Unknown')}")
            except:
                print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        return False

if __name__ == "__main__":
    success = test_zonos_tts()
    if success:
        print(f"\n🎉 Zonos TTS test PASSED!")
    else:
        print(f"\n💥 Zonos TTS test FAILED!")
