#!/usr/bin/env python3
"""
Simple test script to verify TTS audio generation with debugging
"""

import requests
import json
import base64
import wave
import os

def test_tts_audio():
    """Test TTS audio generation with debugging"""
    
    print("🧪 Testing TTS Audio Generation")
    print("=" * 40)
    
    # Wait for service to be ready
    print("⏳ Waiting for TTS service...")
    import time
    time.sleep(2)
    
    # Test text
    test_text = "Hello! This is a test of the TTS audio generation system."
    
    print(f"📝 Text: '{test_text}'")
    
    try:
        # Make TTS request
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
            print("✅ TTS Request Successful!")
            
            # Check response data
            file_path = result.get('file_path')
            duration = result.get('duration')
            generation_time = result.get('generation_time')
            
            print(f"📁 File: {file_path}")
            print(f"⏱️ Duration: {duration}s")
            print(f"🕐 Generation Time: {generation_time}s")
            
            # Check if file exists and has content
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"📊 File Size: {file_size} bytes")
                
                if file_size > 1000:  # More than 1KB suggests real audio
                    print("✅ Audio file appears to contain data")
                else:
                    print("⚠️ Audio file is very small - may be empty")
                
                # Try to analyze with wave module
                try:
                    with wave.open(file_path, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        sample_rate = wav_file.getframerate()
                        channels = wav_file.getnchannels()
                        
                        actual_duration = frames / sample_rate
                        
                        print(f"🎵 Audio Analysis:")
                        print(f"   - Frames: {frames}")
                        print(f"   - Sample Rate: {sample_rate} Hz")
                        print(f"   - Channels: {channels}")
                        print(f"   - Calculated Duration: {actual_duration:.2f}s")
                        
                        if frames > 0:
                            print("✅ Audio file has valid audio data!")
                        else:
                            print("❌ Audio file is empty!")
                            
                except Exception as e:
                    print(f"⚠️ Could not analyze audio file: {e}")
            else:
                print("❌ Audio file not found!")
                
            # Check base64 audio if included
            if 'audio_base64' in result:
                audio_b64 = result['audio_base64']
                print(f"📦 Base64 Audio: {len(audio_b64)} characters")
                
                if len(audio_b64) > 1000:
                    print("✅ Base64 audio data appears substantial")
                else:
                    print("⚠️ Base64 audio data is very small")
                    
            return True
        else:
            print(f"❌ TTS Request Failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_tts_audio()
