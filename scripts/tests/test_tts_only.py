#!/usr/bin/env python3
"""
Simple TTS Test
"""
import requests
import base64
from datetime import datetime

def test_tts_only():
    """Test TTS service only"""
    print("🔊 Testing TTS Service")
    print("=" * 30)
    
    test_text = "Hello world, this is a test of the improved text to speech system."
    
    try:
        tts_response = requests.post(
            'http://localhost:8011/synthesize', 
            json={
                'text': test_text, 
                'voice': 'af_bella', 
                'return_audio': True
            },
            timeout=30
        )
        
        if tts_response.status_code == 200:
            result = tts_response.json()
            if result.get('audio_base64'):
                print("✅ SUCCESS: Audio generated!")
                print(f"Input text: {test_text}")
                print(f"Audio size: {len(result['audio_base64'])} characters (base64)")
                print(f"Metadata: {result['metadata']}")
                
                # Save audio file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_tts_improved_{timestamp}.wav"
                
                audio_bytes = base64.b64decode(result['audio_base64'])
                with open(filename, 'wb') as f:
                    f.write(audio_bytes)
                print(f"💾 Audio saved as: {filename}")
                
                return True
            else:
                print("❌ No audio data received")
                return False
        else:
            print(f"❌ TTS failed: {tts_response.status_code}")
            print(f"Response: {tts_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_tts_only()
    if success:
        print("\n🎉 TTS test completed successfully!")
        print("The audio should now sound more speech-like instead of a simple beep.")
    else:
        print("\n💥 TTS test failed!")
