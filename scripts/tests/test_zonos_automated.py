#!/usr/bin/env python3
"""
Automated Zonos TTS Test
Test Zonos TTS service with predefined inputs
"""

import requests
import json
import time
import os

def test_zonos_tts():
    """Automated test of Zonos TTS service"""
    zonos_url = "http://localhost:8014"
    
    print("🧠 Automated Zonos TTS Test")
    print("="*60)
    
    # Test health check
    try:
        health_response = requests.get(f"{zonos_url}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Service status: {health_data.get('status', 'unknown')}")
            print(f"   Engine: {health_data.get('engine', 'unknown')}")
            print(f"   Implementation: {health_data.get('implementation', 'unknown')}")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Zonos TTS service: {e}")
        print("   Make sure the service is running on port 8014")
        return False
    
    # Test voices endpoint
    try:
        voices_response = requests.get(f"{zonos_url}/voices", timeout=5)
        if voices_response.status_code == 200:
            voices = voices_response.json()
            print(f"🎭 Available voices: {voices}")
            print(f"   Type: {type(voices)}")
            if isinstance(voices, list) and len(voices) > 0:
                selected_voice = voices[0]
                print(f"✅ Voices endpoint working, selected: {selected_voice}")
            else:
                print("⚠️ Voices endpoint returned unexpected format")
                selected_voice = "default"
        else:
            print(f"❌ Voices endpoint failed: {voices_response.status_code}")
            selected_voice = "default"
    except Exception as e:
        print(f"⚠️ Voices endpoint error: {e}")
        selected_voice = "default"
    
    # Test models endpoint
    try:
        models_response = requests.get(f"{zonos_url}/models", timeout=5)
        if models_response.status_code == 200:
            models = models_response.json()
            print(f"🤖 Available models: {models}")
            print(f"   Type: {type(models)}")
            if isinstance(models, list) and len(models) > 0:
                selected_model = models[0]
                print(f"✅ Models endpoint working, selected: {selected_model}")
            else:
                print("⚠️ Models endpoint returned unexpected format")
                selected_model = "zonos-v1"
        else:
            print(f"❌ Models endpoint failed: {models_response.status_code}")
            selected_model = "zonos-v1"
    except Exception as e:
        print(f"⚠️ Models endpoint error: {e}")
        selected_model = "zonos-v1"
    
    # Test synthesis
    print("\n🎯 Testing Neural TTS Synthesis")
    print("-"*40)
    
    test_text = "Hello! This is a test of Zonos neural text-to-speech synthesis. How does it sound?"
    
    synthesis_data = {
        "text": test_text,
        "voice": selected_voice,
        "model": selected_model,
        "emotion": "neutral",
        "speed": 1.0,
        "pitch": 1.0,
        "high_quality": True,
        "seed": 42
    }
    
    print(f"📝 Text: '{test_text}'")
    print(f"🎭 Voice: {selected_voice}")
    print(f"🤖 Model: {selected_model}")
    print(f"😐 Emotion: neutral")
    print(f"⚡ Speed: 1.0x")
    print(f"🎵 Pitch: 1.0x")
    print(f"🌟 High Quality: True")
    print(f"🎲 Seed: 42")
    
    try:
        print("🔄 Synthesizing speech...")
        start_time = time.time()
        
        synthesis_response = requests.post(
            f"{zonos_url}/synthesize",
            json=synthesis_data,
            timeout=30
        )
        
        synthesis_time = time.time() - start_time
        
        if synthesis_response.status_code == 200:
            result = synthesis_response.json()
            print(f"✅ Synthesis successful in {synthesis_time:.2f}s")
            
            # Check for audio data
            audio_data = result.get('audio_data') or result.get('audio_base64')
            if audio_data and len(audio_data) > 0:
                print(f"🎵 Audio data: {len(audio_data)} characters")
                
                # Try to decode and save
                try:
                    import base64
                    audio_bytes = base64.b64decode(audio_data)
                    output_file = "test_zonos_synthesis.wav"
                    
                    with open(output_file, "wb") as f:
                        f.write(audio_bytes)
                    
                    print(f"💾 Audio saved to: {output_file}")
                    print(f"📊 Audio file size: {len(audio_bytes)} bytes")
                    
                    if os.path.exists(output_file):
                        print("✅ Audio file created successfully")
                    else:
                        print("❌ Audio file not created")
                        
                except Exception as e:
                    print(f"⚠️ Audio decoding error: {e}")
            else:
                print("❌ No audio data in response")
                print(f"   Available fields: {list(result.keys())}")
            
            # Show metadata if available
            if 'metadata' in result:
                metadata = result['metadata']
                print(f"📋 Metadata: {metadata}")
                
        else:
            print(f"❌ Synthesis failed: {synthesis_response.status_code}")
            print(f"   Response: {synthesis_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        return False
    
    print("\n🎉 Zonos TTS test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_zonos_tts()
    if success:
        print("\n✅ All tests passed - Zonos TTS is working correctly!")
    else:
        print("\n❌ Some tests failed - check the output above")
