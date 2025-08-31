#!/usr/bin/env python3
"""
Test STT service on port 8003
"""
import requests
import time
from pathlib import Path

def test_stt_service():
    time.sleep(1)
    
    # Test health endpoint
    try:
        response = requests.get('http://localhost:8003/health', timeout=5)
        if response.status_code == 200:
            health = response.json()
            print('✅ STT Service (port 8003) Health Check:')
            print(f'   Status: {health.get("status")}')
            print(f'   Implementation: {health.get("implementation")}')
            print(f'   Ready: {health.get("ready")}')
            
            # Quick test with an audio file
            audio_files = list(Path('.').glob('*.wav'))
            if audio_files:
                audio_file = audio_files[0]
                print(f'\nTesting with: {audio_file.name}')
                
                with open(audio_file, 'rb') as f:
                    files = {'audio': f}
                    response = requests.post('http://localhost:8003/transcribe', files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    transcript = result.get('text', '').strip()
                    processing_time = result.get('processing_time_seconds', 0.0)
                    print(f'   Transcript: "{transcript}"')
                    print(f'   Processing time: {processing_time:.2f}s')
                    print('\n🎉 STT SERVICE WORKING ON PORT 8003!')
                else:
                    print(f'❌ Transcribe failed: {response.status_code}')
                    print(f'   Response: {response.text}')
        else:
            print(f'❌ Health check failed: {response.status_code}')
    except Exception as e:
        print(f'❌ Test failed: {e}')

if __name__ == "__main__":
    test_stt_service()
