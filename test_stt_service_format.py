#!/usr/bin/env python3
"""
Test Whisper STT Service Response Format
"""
import requests
import json

def test_whisper_service():
    print("🧪 Testing Whisper STT Service Response Format")
    print("=" * 50)
    
    try:
        # Use one of the recorded audio files
        audio_file = 'tests/audio_samples/interactive_pipeline/recorded_audio_20250830_224100.wav'
        
        print(f"Testing file: {audio_file}")
        
        with open(audio_file, 'rb') as f:
            files = {'audio': f}
            response = requests.post('http://localhost:8002/transcribe', files=files, timeout=30)
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Service response:")
            print(json.dumps(result, indent=2))
            
            # Check the field names
            text_field = result.get('text', 'NOT_FOUND')
            transcript_field = result.get('transcript', 'NOT_FOUND')
            
            print(f"\nField analysis:")
            print(f'  "text" field: "{text_field}"')
            print(f'  "transcript" field: "{transcript_field}"')
            
            if text_field != 'NOT_FOUND' and len(text_field.strip()) > 0:
                print("🎉 Found transcription in 'text' field!")
            elif transcript_field != 'NOT_FOUND' and len(transcript_field.strip()) > 0:
                print("🎉 Found transcription in 'transcript' field!")
            else:
                print("❌ No transcription found in either field")
                
        else:
            print(f"❌ Service error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_whisper_service()
