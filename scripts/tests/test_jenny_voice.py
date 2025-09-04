#!/usr/bin/env python3
"""
Simple test for voice synthesis with updated names
"""

from aws_microservices.tts_zonos_service import app
from fastapi.testclient import TestClient

def test_voice_synthesis():
    client = TestClient(app)
    
    # Test Jenny voice
    print("🧪 Testing Jenny voice synthesis...")
    response = client.post('/synthesize', json={
        'text': 'Hello, this is Jenny from Microsoft Edge Neural TTS speaking with professional voice quality.',
        'voice': 'jenny',
        'emotion': 'professional'
    })
    
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    
    if response.status_code == 200:
        print(f"Audio size: {len(response.content)} bytes")
        
        # Save audio file
        with open('test_jenny_voice.wav', 'wb') as f:
            f.write(response.content)
        print("✅ Audio saved as test_jenny_voice.wav")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

if __name__ == "__main__":
    success = test_voice_synthesis()
    if success:
        print("🎉 Voice synthesis test passed!")
    else:
        print("❌ Voice synthesis test failed!")
