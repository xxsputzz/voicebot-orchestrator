#!/usr/bin/env python3

import requests
import json
import time

def test_unicode_fix():
    """Test that the Unicode fix is working in the live TTS service"""
    
    # Test with the original problematic text that had emoji
    test_text = "🎤 Hungrey"
    print(f"🧪 Testing TTS with Unicode text: '{test_text}'")
    
    # Try TTS service
    tts_url = "http://localhost:8004/synthesize"
    tts_data = {
        "text": test_text,
        "engine": "DIA_4BIT",
        "voice": "default"
    }
    
    try:
        print("📡 Sending request to TTS service...")
        response = requests.post(tts_url, json=tts_data, timeout=30)
        
        if response.status_code == 200:
            print("✅ SUCCESS! TTS service handled Unicode text correctly")
            print(f"📊 Response headers: {dict(response.headers)}")
            print(f"📏 Audio data size: {len(response.content)} bytes")
            
            # Save the audio
            with open("test_unicode_success.wav", "wb") as f:
                f.write(response.content)
            print("💾 Audio saved as test_unicode_success.wav")
            
        else:
            print(f"❌ TTS service error: {response.status_code}")
            print(f"🔍 Error response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("⚠️  TTS service not running on port 8004")
    except Exception as e:
        print(f"❌ Error testing TTS: {e}")

def test_more_unicode_cases():
    """Test additional Unicode cases"""
    test_cases = [
        "Hello 🎵 world",
        "🚀 Launch sequence",
        "Time ⏰ is running out",
        "Simple text without emoji"
    ]
    
    print("\n🧪 Testing additional Unicode cases...")
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: '{text}' ---")
        
        tts_url = "http://localhost:8004/synthesize"
        tts_data = {
            "text": text,
            "engine": "DIA_4BIT",
            "voice": "default"
        }
        
        try:
            response = requests.post(tts_url, json=tts_data, timeout=15)
            
            if response.status_code == 200:
                print(f"✅ SUCCESS! Audio: {len(response.content)} bytes")
            else:
                print(f"❌ FAILED: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("⚠️  Service not available")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🎯 Unicode Fix Validation Test")
    print("=" * 50)
    
    test_unicode_fix()
    test_more_unicode_cases()
    
    print("\n🏁 Test complete!")
