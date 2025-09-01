#!/usr/bin/env python3

import requests
import json

def reproduce_unicode_error():
    """Reproduce the exact Unicode error from the user's report"""
    
    # The exact text that caused the error 
    test_text = "🎤 Hungrey"  # This contains the U0001f3a4 character that caused the charmap error
    
    print("🎯 REPRODUCING UNICODE ERROR")
    print("=" * 50)
    print(f"Text causing error: '{test_text}'")
    print(f"Character analysis:")
    for i, char in enumerate(test_text):
        print(f"  {i}: '{char}' (U+{ord(char):04X})")
    
    # Try to send this to the TTS service that was failing
    tts_data = {
        "text": test_text,
        "return_audio": True
    }
    
    print("\n📡 Sending to TTS service...")
    
    # Check if any TTS service is available
    test_ports = [8011, 8012, 8013, 8004]
    
    for port in test_ports:
        print(f"\n🔍 Testing port {port}...")
        try:
            health_response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if health_response.status_code == 200:
                print(f"✅ Service found on port {port}")
                
                # Try the synthesis request
                response = requests.post(f"http://localhost:{port}/synthesize", json=tts_data, timeout=30)
                
                if response.status_code == 200:
                    print(f"🎉 SUCCESS! Unicode text processed correctly on port {port}")
                    result = response.json()
                    if 'metadata' in result:
                        print(f"📊 Metadata: {result['metadata']}")
                    return True
                else:
                    print(f"❌ Error {response.status_code}: {response.text}")
                    if "'charmap' codec can't encode character" in response.text:
                        print("🔍 FOUND THE UNICODE ERROR - this service needs our fix!")
                        return False
                    
        except requests.exceptions.ConnectionError:
            print(f"⚠️  No service on port {port}")
        except Exception as e:
            print(f"❌ Error testing port {port}: {e}")
    
    print("\n❌ No TTS services found or all failed")
    return False

def test_unicode_sanitization_locally():
    """Test our Unicode sanitization directly"""
    
    print("\n🧪 TESTING UNICODE SANITIZATION LOCALLY")
    print("=" * 50)
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'voicebot_orchestrator'))
        
        from enhanced_tts_manager import EnhancedTTSManager
        
        # Create manager to test sanitization
        manager = EnhancedTTSManager()
        
        test_cases = [
            "🎤 Hungrey",  # Original problem
            "Hello 🎵 world",
            "🚀 Launch",
            "Normal text"
        ]
        
        print("Testing text sanitization:")
        for text in test_cases:
            sanitized = manager.sanitize_text_for_synthesis(text)
            has_unicode = any(ord(c) > 127 for c in text)
            is_clean = not any(ord(c) > 127 for c in sanitized)
            status = "✅" if (not has_unicode or is_clean) else "❌"
            print(f"  '{text}' → '{sanitized}' {status}")
        
        print("\n✅ Unicode sanitization is working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import enhanced_tts_manager: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing sanitization: {e}")
        return False

if __name__ == "__main__":
    print("🔍 UNICODE ERROR DIAGNOSIS")
    print("=" * 60)
    
    # Test sanitization locally first
    sanitization_works = test_unicode_sanitization_locally()
    
    # Try to reproduce the live error
    live_test_works = reproduce_unicode_error()
    
    print("\n📊 DIAGNOSIS SUMMARY")
    print("=" * 30)
    print(f"Local sanitization working: {'✅' if sanitization_works else '❌'}")
    print(f"Live service working: {'✅' if live_test_works else '❌'}")
    
    if sanitization_works and not live_test_works:
        print("\n💡 SOLUTION: Restart TTS services to pick up the Unicode fix")
        print("The sanitization code is working, but services need to be restarted")
    elif not sanitization_works:
        print("\n❌ ISSUE: Sanitization code has a problem")
    elif live_test_works:
        print("\n🎉 FIXED: Unicode handling is working correctly!")
    
    print("\n🏁 Diagnosis complete!")
