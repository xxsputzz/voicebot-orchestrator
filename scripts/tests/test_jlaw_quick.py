#!/usr/bin/env python3
import requests

payload = {
    'text': 'Hello, this is jlaw voice with GPU acceleration.',
    'voice': 'jlaw',
    'preset': 'ultra_fast',
    'return_audio': True
}

print("Testing jlaw voice with ultra_fast preset...")
response = requests.post('http://localhost:8015/synthesize', json=payload, timeout=60)

if response.status_code == 200:
    result = response.json()
    metadata = result['metadata']
    print("SUCCESS: jlaw voice working!")
    print(f"Duration: {metadata['duration']:.2f}s")
    print(f"Engine: {metadata['engine']}")
    if 'audio_file' in result:
        print(f"Audio: {result['audio_file']}")
    print("🎉 jlaw voice working with GPU!")
else:
    print(f"Failed: {response.status_code} - {response.text}")
