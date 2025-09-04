#!/usr/bin/env python3
"""
Test STT service network communication
"""
import requests
import time
from pathlib import Path

def test_stt_service():
    print("Testing STT service network communication...")
    
    # Test health endpoint first
    try:
        response = requests.get('http://localhost:8002/health', timeout=10)
        if response.status_code == 200:
            health = response.json()
            print('✅ STT Service Health Check:')
            print(f'   Status: {health.get("status")}')
            print(f'   Implementation: {health.get("implementation")}')
            print(f'   Ready: {health.get("ready")}')
        else:
            print(f'❌ Health check failed: {response.status_code}')
            return False
    except Exception as e:
        print(f'❌ Health check failed: {e}')
        return False

    # Find a test audio file
    audio_files = list(Path('.').glob('*.wav'))
    if not audio_files:
        print('❌ No audio files found')
        return False

    audio_file = audio_files[0]
    print(f'\nTesting STT service with: {audio_file.name}')

    try:
        with open(audio_file, 'rb') as f:
            files = {'audio': f}
            print("Sending request to STT service...")
            response = requests.post('http://localhost:8002/transcribe', files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            transcript = result.get('text', '').strip()
            confidence = result.get('confidence', 0.0)
            processing_time = result.get('processing_time_seconds', 0.0)
            
            print('\n✅ STT SERVICE TEST RESULT:')
            print(f'   Transcript: "{transcript}"')
            print(f'   Confidence: {confidence:.2f}')
            print(f'   Processing Time: {processing_time:.2f}s')
            
            if transcript and len(transcript) > 3:
                print('\n🎉 STT SERVICE IS NOW WORKING OVER NETWORK!')
                return True
            else:
                print('\n❌ Still getting empty/short transcript')
                return False
        else:
            print(f'❌ Request failed: {response.status_code}')
            print(f'   Response: {response.text}')
            return False
            
    except Exception as e:
        print(f'❌ Test failed: {e}')
        return False

if __name__ == "__main__":
    success = test_stt_service()
    exit(0 if success else 1)
