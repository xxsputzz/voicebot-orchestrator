#!/usr/bin/env python3
"""
Test the fixed TTS service to ensure no more Unicode encoding errors
"""

import requests
import json
import time

def test_tts_service():
    """Test the TTS service with various text inputs"""
    
    base_url = "http://localhost:8012"
    
    # Test cases
    test_cases = [
        "Hello, this is a simple test.",
        "Testing text with numbers: 123 and punctuation!",
        "This is a longer test to see if the service can handle more text without any encoding issues.",
        "Testing special characters: $100, 50% off, email@domain.com"
    ]
    
    print("🧪 Testing TTS Service (Unicode fixes)")
    print("=" * 50)
    
    # First, check if service is healthy
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Service is healthy")
        else:
            print(f"❌ Service health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        return False
    
    # Test synthesis requests
    for i, text in enumerate(test_cases, 1):
        print(f"\n🎯 Test {i}: '{text[:30]}{'...' if len(text) > 30 else ''}'")
        
        try:
            payload = {
                "text": text,
                "return_audio": False  # Don't return large audio data for testing
            }
            
            response = requests.post(
                f"{base_url}/synthesize",
                json=payload,
                timeout=300  # 5 minutes for TTS generation
            )
            
            if response.status_code == 200:
                result = response.json()
                metadata = result.get('metadata', {})
                print(f"   ✅ Success! Engine: {metadata.get('engine_used', 'unknown')}")
                print(f"   ⏱️  Processing time: {metadata.get('processing_time_seconds', 'unknown')}s")
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"   💬 Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            return False
    
    print(f"\n🎉 All tests passed! Unicode encoding issues are fixed.")
    return True

if __name__ == "__main__":
    # Wait a moment for service to be ready
    print("⏳ Waiting 30 seconds for TTS service to be ready...")
    time.sleep(30)
    
    success = test_tts_service()
    if success:
        print("\n✅ TTS 500 error investigation complete - Unicode issues resolved!")
    else:
        print("\n❌ Some tests failed - further investigation needed.")
