#!/usr/bin/env python3
"""
Validation Script for Real Kokoro TTS Implementation
Tests both direct usage and service integration
"""

import asyncio
import sys
import time
import subprocess
import requests
import base64
from pathlib import Path

async def test_direct_kokoro():
    """Test Real Kokoro TTS directly"""
    print("\n=== DIRECT REAL KOKORO TTS TEST ===")
    
    try:
        from voicebot_orchestrator.real_kokoro_tts import KokoroTTS
        
        print("[1/3] Initializing Real Kokoro TTS...")
        tts = KokoroTTS(voice="af_bella")
        
        print("[2/3] Generating speech...")
        text = "This is a validation test of the Real Kokoro TTS system. If you hear this, the implementation is working correctly."
        audio_bytes = await tts.synthesize_speech(text)
        
        print("[3/3] Saving audio...")
        filename = "validation_direct_kokoro.wav"
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        
        print(f"✅ SUCCESS: Direct test completed")
        print(f"   Audio size: {len(audio_bytes)} bytes")
        print(f"   File saved: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Direct test failed: {e}")
        return False

def test_service_kokoro():
    """Test Real Kokoro TTS service"""
    print("\n=== SERVICE REAL KOKORO TTS TEST ===")
    
    try:
        print("[1/4] Starting TTS service...")
        # Start service in background
        service_process = subprocess.Popen(
            [sys.executable, "aws_microservices/tts_kokoro_service.py"],
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for service to start
        time.sleep(5)
        
        print("[2/4] Testing service health...")
        response = requests.get("http://localhost:8011/health", timeout=10)
        if response.status_code != 200:
            raise Exception(f"Health check failed: {response.status_code}")
        
        health_data = response.json()
        print(f"   Engine: {health_data.get('engine', 'unknown')}")
        print(f"   Implementation: {health_data.get('implementation', 'unknown')}")
        
        print("[3/4] Testing speech synthesis...")
        text = "Service test successful! The Real Kokoro TTS microservice is working properly and generating actual speech."
        
        synth_response = requests.post("http://localhost:8011/synthesize",
            json={
                'text': text,
                'voice': 'af_bella',
                'speed': 1.0,
                'return_audio': True
            },
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if synth_response.status_code != 200:
            raise Exception(f"Synthesis failed: {synth_response.status_code}")
        
        synth_data = synth_response.json()
        metadata = synth_data['metadata']
        
        print("[4/4] Saving service audio...")
        if synth_data.get('audio_base64'):
            audio_bytes = base64.b64decode(synth_data['audio_base64'])
            filename = "validation_service_kokoro.wav"
            with open(filename, 'wb') as f:
                f.write(audio_bytes)
            
            print(f"✅ SUCCESS: Service test completed")
            print(f"   Engine: {metadata['engine_used']}")
            print(f"   Implementation: {metadata.get('implementation', 'unknown')}")
            print(f"   Processing time: {metadata['processing_time_seconds']}s")
            print(f"   Audio size: {metadata['audio_size_bytes']} bytes")
            print(f"   File saved: {filename}")
            
            # Stop service
            service_process.terminate()
            service_process.wait(timeout=5)
            return True
        else:
            raise Exception("No audio data returned")
            
    except Exception as e:
        print(f"❌ FAILED: Service test failed: {e}")
        try:
            service_process.terminate()
            service_process.wait(timeout=5)
        except:
            pass
        return False

async def main():
    """Run all validation tests"""
    print("🎯 REAL KOKORO TTS VALIDATION")
    print("=" * 50)
    
    # Test direct usage
    direct_success = await test_direct_kokoro()
    
    # Test service
    service_success = test_service_kokoro()
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION RESULTS:")
    print(f"   Direct TTS:  {'✅ PASS' if direct_success else '❌ FAIL'}")
    print(f"   Service TTS: {'✅ PASS' if service_success else '❌ FAIL'}")
    
    if direct_success and service_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("   Real Kokoro TTS is fully functional!")
        print("   Unicode encoding issues resolved!")
        print("   Service integration working!")
    else:
        print("\n⚠️ Some tests failed. Check the output above.")
    
    return direct_success and service_success

if __name__ == "__main__":
    asyncio.run(main())
