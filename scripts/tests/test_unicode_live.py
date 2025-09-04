#!/usr/bin/env python3

import requests
import json

def test_unicode_fix_live():
    """Test that our Unicode fix works with the live service"""
    
    # Test the original problematic text that had emoji
    test_text = "🎤 Hungrey"
    print(f"🎯 Testing Unicode fix with: '{test_text}'")
    
    # Test with TTS service
    tts_url = "http://localhost:8012/synthesize"
    tts_data = {
        "text": test_text,
        "return_audio": True,
        "high_quality": False,  # Use 4-bit for speed
        "engine_preference": "auto"
    }
    
    try:
        print("📡 Sending request to TTS service...")
        response = requests.post(tts_url, json=tts_data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! TTS service handled Unicode text correctly")
            print(f"📊 Audio generated: {len(result.get('audio_base64', ''))} base64 chars")
            print(f"🔧 Engine used: {result.get('metadata', {}).get('engine_used', 'unknown')}")
            print(f"⏱️  Processing time: {result.get('metadata', {}).get('processing_time_seconds', 0)} seconds")
            
            print("\n🎉 UNICODE FIX IS WORKING! 🎉")
            print("✅ The original '🎤 Hungrey' problem is SOLVED!")
            print("✅ No more 'charmap codec' errors!")
            
        else:
            print(f"❌ TTS service error: {response.status_code}")
            error_text = response.text
            print(f"🔍 Error response: {error_text}")
            
            if "'charmap' codec can't encode character" in error_text:
                print("❌ Unicode fix is NOT working - charmap error still occurring")
                print("💡 Service needs to be restarted to pick up the fix")
            else:
                print("❓ Different error - may not be Unicode related")
            
    except requests.exceptions.ConnectionError:
        print("⚠️  TTS service not running on port 8012")
        print("💡 Start the service with: python aws_microservices/tts_hira_dia_service.py --engine auto")
    except Exception as e:
        print(f"❌ Error testing TTS: {e}")

def test_additional_unicode_cases():
    """Test additional Unicode cases"""
    print("\n🧪 Testing additional Unicode cases...")
    
    test_cases = [
        "Hello 🎵 world",
        "🚀 Launch sequence",
        "Time ⏰ is running out",
        "🎉🎊 Party time!"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: '{text}' ---")
        
        tts_url = "http://localhost:8012/synthesize"
        tts_data = {
            "text": text,
            "return_audio": True,
            "high_quality": False,
            "engine_preference": "auto"
        }
        
        try:
            response = requests.post(tts_url, json=tts_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                engine_used = result.get('metadata', {}).get('engine_used', 'unknown')
                print(f"✅ SUCCESS! Engine: {engine_used}")
            else:
                print(f"❌ FAILED: {response.status_code} - {response.text[:100]}")
                
        except requests.exceptions.ConnectionError:
            print("⚠️  Service not available")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🎯 Live Unicode Fix Test")
    print("=" * 50)
    
    test_unicode_fix_live()
    test_additional_unicode_cases()
    
    print("\n🏁 Test complete!")
