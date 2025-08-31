#!/usr/bin/env python3
"""
Direct STT Service Test
======================
"""
import requests
from pathlib import Path

def test_stt_service():
    """Test STT service directly with detailed output"""
    
    # Find an audio file with real speech 
    audio_files = list(Path('.').glob('*kokoro*.wav'))
    if not audio_files:
        audio_files = list(Path('.').glob('*.wav'))

    if audio_files:
        audio_file = audio_files[0]
        print(f'🎵 Testing STT with: {audio_file.name}')
        
        try:
            with open(audio_file, 'rb') as f:
                files = {'audio': f}
                response = requests.post('http://localhost:8003/transcribe', files=files, timeout=60)
            
            print(f'📊 Status Code: {response.status_code}')
            
            if response.status_code == 200:
                result = response.json()
                transcript = result.get('text', '').strip()
                processing_time = result.get('processing_time_seconds', 0.0)
                confidence = result.get('confidence', 0.0)
                
                print(f'📋 Response Keys: {list(result.keys())}')
                print(f'📝 Raw Response: {result}')
                print(f'🎙️  Transcript: "{transcript}"')
                print(f'📏 Transcript Length: {len(transcript)}')
                print(f'🎯 Confidence: {confidence:.2f}')
                print(f'⏱️  Processing time: {processing_time:.2f}s')
                
                if transcript and len(transcript) > 3:
                    print('✅ SUCCESS: STT SERVICE WORKING OVER NETWORK!')
                    return True
                else:
                    print('❌ PROBLEM: Empty transcript received')
                    return False
            else:
                print(f'❌ Request failed: {response.status_code}')
                print(f'📄 Response: {response.text}')
                return False
        except Exception as e:
            print(f'❌ Test failed: {e}')
            import traceback
            traceback.print_exc()
            return False
    else:
        print('❌ No audio files found')
        return False

# Test health endpoint first
def test_health():
    """Test health endpoint"""
    try:
        response = requests.get('http://localhost:8003/health', timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"🟢 Health Status: {health}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing STT Service Direct Connection")
    print("=" * 50)
    
    # Test health first
    print("\n1. Testing health endpoint...")
    health_ok = test_health()
    
    if health_ok:
        print("\n2. Testing transcription...")
        success = test_stt_service()
        
        if success:
            print("\n✅ ALL TESTS PASSED!")
        else:
            print("\n❌ TRANSCRIPTION TEST FAILED!")
    else:
        print("\n❌ SERVICE NOT HEALTHY!")
