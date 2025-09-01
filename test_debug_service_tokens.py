#!/usr/bin/env python3
"""
Debug Service Token Handling vs Direct Model
Test to understand why service generates shorter audio than direct model testing
"""

import requests
import json
import time
import numpy as np
import soundfile as sf
import base64
import io

def test_service_tts_with_debug():
    """Test TTS service with detailed debugging"""
    
    # Use the same test text that worked in direct model
    test_text = """The ancient library held secrets beyond imagination. Dusty tomes lined endless shelves, each containing knowledge from civilizations long forgotten. As Sarah walked through the narrow aisles, her footsteps echoed in the vast silence. The smell of old parchment and leather bindings filled the air. Somewhere in these depths lay the answer she sought - a cure for the mysterious plague that had befallen her village. Time was running out, and the keeper of this knowledge was said to be both wise and dangerous."""
    
    print(f"🔤 Test text length: {len(test_text)} characters")
    print(f"📄 Text content: {test_text[:100]}...")
    
    # Calculate expected tokens like the service does
    expected_tokens = len(test_text) * 12  # Latest calculation from service
    max_tokens = min(65536, max(4096, expected_tokens))
    
    print(f"🧮 Expected tokens calculation: {len(test_text)} * 12 = {expected_tokens}")
    print(f"🎯 Final max_tokens: {max_tokens}")
    
    # Test service
    url = "http://localhost:8012/synthesize"
    
    payload = {
        "text": test_text,
        "voice": "default",
        "speed": 1.0,
        "output_format": "wav",
        "return_audio": True,
        "high_quality": True,
        "engine_preference": "full"
    }
    
    print(f"\n🌐 Sending request to {url}")
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        duration = time.time() - start_time
        
        print(f"⏱️ Request completed in {duration:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"✅ Service response keys: {list(result.keys())}")
                
                if 'audio_base64' in result:
                    # Decode base64 audio
                    audio_data = base64.b64decode(result['audio_base64'])
                    print(f"🔊 Audio data size: {len(audio_data)} bytes")
                    
                    # Save and analyze audio
                    filename = f"debug_service_audio_{int(time.time())}.wav"
                    with open(filename, 'wb') as f:
                        f.write(audio_data)
                    
                    # Load with soundfile to get actual duration
                    try:
                        audio_array, sample_rate = sf.read(filename)
                        actual_duration = len(audio_array) / sample_rate
                        print(f"🎵 Actual audio duration: {actual_duration:.2f} seconds")
                        print(f"🎵 Sample rate: {sample_rate} Hz")
                        print(f"🎵 Audio samples: {len(audio_array)}")
                        print(f"🎵 Audio shape: {audio_array.shape}")
                        
                        # Check if audio has content or is silence
                        max_amplitude = np.max(np.abs(audio_array))
                        rms = np.sqrt(np.mean(audio_array**2))
                        print(f"🔉 Max amplitude: {max_amplitude:.6f}")
                        print(f"🔉 RMS level: {rms:.6f}")
                        
                        if max_amplitude < 0.001:
                            print("⚠️ Audio appears to be silent or very quiet!")
                        else:
                            print("✅ Audio has audible content")
                            
                    except Exception as e:
                        print(f"❌ Error analyzing audio file: {e}")
                
                # Print any other response data
                for key, value in result.items():
                    if key != 'audio_base64':
                        if key == 'metadata' and isinstance(value, dict):
                            print(f"📋 metadata:")
                            for meta_key, meta_value in value.items():
                                print(f"    {meta_key}: {meta_value}")
                        else:
                            print(f"📋 {key}: {value}")
                        
            except json.JSONDecodeError as e:
                print(f"❌ Failed to decode JSON response: {e}")
                print(f"📝 Raw response: {response.text[:500]}...")
                
        else:
            print(f"❌ Service error: {response.status_code}")
            print(f"📝 Error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out after 300 seconds")
    except requests.exceptions.RequestException as e:
        print(f"🌐 Request failed: {e}")

if __name__ == "__main__":
    print("🔍 Debug Service Token Handling Test")
    print("=" * 50)
    test_service_tts_with_debug()
    print("\n" + "=" * 50)
    print("🏁 Debug test completed")
