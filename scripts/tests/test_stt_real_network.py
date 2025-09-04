#!/usr/bin/env python3
"""
Test real STT network transcription on port 8003
"""
import requests
from pathlib import Path

def test_real_stt():
    # Find an audio file with real speech 
    audio_files = list(Path('.').glob('*kokoro*.wav'))
    if not audio_files:
        audio_files = list(Path('.').glob('*.wav'))

    if audio_files:
        audio_file = audio_files[0]
        print(f'Testing STT with: {audio_file.name}')
        
        try:
            with open(audio_file, 'rb') as f:
                files = {'audio': f}
                response = requests.post('http://localhost:8003/transcribe', files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                transcript = result.get('text', '').strip()
                processing_time = result.get('processing_time_seconds', 0.0)
                confidence = result.get('confidence', 0.0)
                
                print('\n✅ REAL STT NETWORK TRANSCRIPTION:')
                print(f'   File: {audio_file.name}')
                print(f'   Transcript: "{transcript}"')
                print(f'   Confidence: {confidence:.2f}')
                print(f'   Processing time: {processing_time:.2f}s')
                
                if transcript and len(transcript) > 3:
                    print('\n🎉 SUCCESS: STT SERVICE WORKING OVER NETWORK!')
                    return True
                else:
                    print('\n⚠️  Empty transcript received')
                    return False
            else:
                print(f'❌ Request failed: {response.status_code}')
                print(f'   Response: {response.text}')
                return False
        except Exception as e:
            print(f'❌ Test failed: {e}')
            return False
    else:
        print('❌ No audio files found')
        return False

if __name__ == "__main__":
    test_real_stt()
