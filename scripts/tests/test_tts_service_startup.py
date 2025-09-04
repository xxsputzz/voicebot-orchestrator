#!/usr/bin/env python3
"""
Quick test to verify TTS service startup and basic functionality
"""

import requests
import time
import sys
from pathlib import Path

def test_tts_service():
    """Test TTS service startup and health check"""
    print("🧪 Testing TTS Service Startup")
    print("=" * 50)
    
    service_url = "http://localhost:8015"
    
    # Test 1: Health Check
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{service_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Service is healthy")
            print(f"   Engine: {health_data.get('engine', 'unknown')}")
            print(f"   Device: {health_data.get('device', 'unknown')}")
            print(f"   Voice count: {health_data.get('voice_count', 'unknown')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Service not running on {service_url}")
        print(f"   Start the service first with: python aws_microservices/tts_tortoise_service.py --direct")
        return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 2: Voices endpoint
    print("\n2️⃣ Testing voices endpoint...")
    try:
        response = requests.get(f"{service_url}/voices", timeout=10)
        if response.status_code == 200:
            voices_data = response.json()
            print(f"   ✅ Found {voices_data.get('total', 0)} voices")
            voices = voices_data.get('voices', [])[:5]  # First 5 voices
            print(f"   Sample voices: {', '.join(voices)}")
        else:
            print(f"   ❌ Voices endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Voices endpoint error: {e}")
        return False
    
    # Test 3: GPU Status
    print("\n3️⃣ Testing GPU status...")
    try:
        response = requests.get(f"{service_url}/gpu-status", timeout=10)
        if response.status_code == 200:
            gpu_data = response.json()
            print(f"   ✅ GPU manager active")
            print(f"   Device: {gpu_data.get('device', 'unknown')}")
            print(f"   Tracked models: {gpu_data.get('tracked_models', 0)}")
        else:
            print(f"   ⚠️ GPU status unavailable: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️ GPU status error: {e}")
    
    # Test 4: Timeout Stats
    print("\n4️⃣ Testing timeout configuration...")
    try:
        response = requests.get(f"{service_url}/timeout-stats", timeout=10)
        if response.status_code == 200:
            timeout_data = response.json()
            config = timeout_data.get('timeout_config', {})
            print(f"   ✅ Timeout system configured")
            print(f"   Base char time: {config.get('char_processing_time', 'unknown')}s per char")
            print(f"   Min timeout: {config.get('min_timeout', 'unknown')}s")
            print(f"   Safety buffer: {config.get('safety_buffer', 'unknown')}x")
        else:
            print(f"   ❌ Timeout stats failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Timeout stats error: {e}")
    
    print(f"\n✅ Service startup test completed successfully!")
    print(f"🎯 The TTS service is ready for synthesis requests")
    
    return True

def test_quick_synthesis():
    """Test a very short synthesis to verify the pipeline works"""
    print(f"\n🎤 Testing Quick Synthesis...")
    
    service_url = "http://localhost:8015"
    
    synthesis_data = {
        "text": "Hello",  # Very short text for quick test
        "voice": "angie",
        "preset": "ultra_fast"
    }
    
    try:
        print(f"   Synthesizing: '{synthesis_data['text']}'")
        start_time = time.time()
        
        response = requests.post(
            f"{service_url}/synthesize", 
            json=synthesis_data, 
            timeout=60  # 1 minute timeout for quick test
        )
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Synthesis completed in {processing_time:.1f}s")
            print(f"   Service processing time: {result.get('processing_time', 'unknown'):.1f}s")
            
            if result.get('audio_base64'):
                print(f"   ✅ Audio generated successfully")
            else:
                print(f"   ⚠️ No audio data returned")
            
            return True
        else:
            print(f"   ❌ Synthesis failed: {response.status_code}")
            try:
                error = response.json()
                print(f"   Error: {error.get('detail', 'unknown')}")
            except:
                print(f"   Raw error: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   ⏰ Synthesis timed out after 60 seconds")
        print(f"   This is expected for longer texts, but 'Hello' should be quick")
        return False
    except Exception as e:
        print(f"   ❌ Synthesis error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TTS Service Test Suite")
    print("Make sure the service is running first:")
    print("python aws_microservices/tts_tortoise_service.py --direct")
    print()
    
    if test_tts_service():
        if len(sys.argv) > 1 and sys.argv[1] == "--with-synthesis":
            test_quick_synthesis()
        else:
            print("\nTo test synthesis, run with: --with-synthesis")
    else:
        print("\n❌ Basic service tests failed")
        sys.exit(1)
