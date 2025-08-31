#!/usr/bin/env python3
"""
Test STT service with proper file upload to trigger debug logs
"""
import requests
import os
import sys

def test_stt_with_file_upload():
    """Test STT service by uploading audio file properly"""
    
    # Test file
    audio_file = "benchmark_kokoro_1.wav"
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    file_size = os.path.getsize(audio_file)
    print(f"🧪 Testing STT service with file upload: {audio_file} ({file_size} bytes)")
    
    try:
        # Upload file to STT service
        with open(audio_file, 'rb') as f:
            files = {'audio': (audio_file, f, 'audio/wav')}
            response = requests.post(
                'http://localhost:8003/transcribe',
                files=files,
                timeout=30
            )
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📋 Full Response: {result}")
            print(f"📝 Text: \"{result.get('text', '')}\"")
            print(f"📏 Length: {len(result.get('text', ''))}")
            print(f"🎯 Confidence: {result.get('confidence', 'N/A')}")
            print(f"⏱️  Processing time: {result.get('processing_time_seconds', 0):.2f}s")
            
            if result.get('text', '').strip():
                print("✅ SUCCESS: Got transcript!")
                return True
            else:
                print("❌ FAILED: Empty transcript!")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔊 STT File Upload Test")
    print("=" * 50)
    success = test_stt_with_file_upload()
    sys.exit(0 if success else 1)
