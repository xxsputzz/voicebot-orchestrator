#!/usr/bin/env python3
"""
Test script to check TTS service voice availability
"""

import requests
import json
import time

def test_tts_service_voices():
    """Test the TTS service voices endpoint"""
    service_url = "http://localhost:5000"
    
    print("🧪 Testing TTS Service Voice Availability...")
    
    # Wait a moment for service to be ready
    print("⏳ Waiting for service to be ready...")
    time.sleep(5)
    
    try:
        # Test voices endpoint
        print(f"📡 Requesting voices from {service_url}/voices")
        response = requests.get(f"{service_url}/voices", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            voices = data.get('voices', [])
            
            print(f"✅ SUCCESS: Found {len(voices)} voices!")
            print(f"📋 Available voices:")
            for i, voice in enumerate(voices, 1):
                print(f"  {i:2d}. {voice}")
            
            # Check for specific official voices
            official_voices = ['angie', 'daniel', 'deniro', 'emma', 'freeman', 'geralt', 'halle', 'jlaw']
            found_official = [v for v in official_voices if v in voices]
            
            print(f"\n🎯 Official voices found: {len(found_official)}/{len(official_voices)}")
            for voice in found_official:
                print(f"  ✅ {voice}")
            
            missing_official = [v for v in official_voices if v not in voices]
            if missing_official:
                print(f"\n⚠️  Missing official voices:")
                for voice in missing_official:
                    print(f"  ❌ {voice}")
            
            return len(voices)
            
        else:
            print(f"❌ Failed to get voices: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return 0
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to TTS service - is it running on port 5000?")
        return 0
    except Exception as e:
        print(f"❌ Error testing TTS service: {e}")
        return 0

if __name__ == "__main__":
    voice_count = test_tts_service_voices()
    print(f"\n🏁 Final Result: {voice_count} voices detected")
    
    if voice_count >= 19:
        print("🎉 SUCCESS: All official voices are working!")
    elif voice_count > 3:
        print("✅ PROGRESS: More voices than before, but still missing some")
    else:
        print("❌ ISSUE: Still only showing few voices")
