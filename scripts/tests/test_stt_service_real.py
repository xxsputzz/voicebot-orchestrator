#!/usr/bin/env python3
"""
Test STT Service with Real Audio
===============================

Test the updated STT service with real microphone recordings.
"""

import requests
from pathlib import Path

def test_stt_service_with_real_audio():
    """Test STT service with real recorded audio"""
    print("🧪 TESTING STT SERVICE WITH REAL AUDIO")
    print("=" * 50)
    
    # Find test audio files
    audio_dir = Path("tests/audio_samples/microphone_tests")
    audio_files = list(audio_dir.glob("*.wav"))
    
    if not audio_files:
        print("❌ No test audio files found")
        return
    
    # Test each audio file
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n📄 Testing file {i}: {audio_file.name}")
        print(f"   Size: {audio_file.stat().st_size:,} bytes")
        
        try:
            # Send to STT service
            print("   🔄 Sending to STT service...")
            
            with open(audio_file, 'rb') as f:
                files = {'audio': f}
                response = requests.post("http://localhost:8002/transcribe", files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                transcript = result.get("text", "").strip()
                confidence = result.get("confidence", 0.0)
                processing_time = result.get("processing_time_seconds", 0.0)
                
                print(f"   ✅ SUCCESS!")
                print(f"   📝 Transcript: '{transcript}'")
                print(f"   🎯 Confidence: {confidence:.2f}")
                print(f"   ⏱️  Processing: {processing_time:.3f}s")
                
                if transcript and "I would like to check my account balance please" not in transcript:
                    print(f"   🎉 REAL TRANSCRIPTION WORKING!")
                else:
                    print(f"   ⚠️  Still getting mock response")
                    
            else:
                print(f"   ❌ Request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🏁 Test completed!")

if __name__ == "__main__":
    test_stt_service_with_real_audio()
