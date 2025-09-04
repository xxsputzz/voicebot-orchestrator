#!/usr/bin/env python3
"""
Simple LLM → TTS Pipeline Test
"""
import requests
import json
import base64
from datetime import datetime

def test_llm_to_tts():
    """Test LLM → TTS pipeline"""
    print("🧠 Testing LLM → TTS Pipeline")
    print("=" * 40)
    
    # Test LLM
    print("Step 1: Testing LLM...")
    try:
        llm_response = requests.post(
            'http://localhost:8023/generate', 
            json={
                'text': 'What is the weather like today?', 
                'temperature': 0.7, 
                'max_tokens': 100
            },
            timeout=30
        )
        
        if llm_response.status_code == 200:
            llm_result = llm_response.json()
            llm_text = llm_result.get('response', '')
            print(f"✅ LLM Response: {llm_text[:100]}...")
            
            # Test TTS with LLM output
            print("\nStep 2: Testing TTS...")
            tts_response = requests.post(
                'http://localhost:8011/synthesize', 
                json={
                    'text': llm_text, 
                    'voice': 'af_bella', 
                    'return_audio': True
                },
                timeout=60
            )
            
            if tts_response.status_code == 200:
                result = tts_response.json()
                if result.get('audio_base64'):
                    print("✅ SUCCESS: Audio generated!")
                    print(f"Audio size: {len(result['audio_base64'])} characters (base64)")
                    print(f"Metadata: {result['metadata']}")
                    
                    # Save audio file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"test_pipeline_{timestamp}.wav"
                    
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
        else:
            print(f"❌ LLM failed: {llm_response.status_code}")
            print(f"Response: {llm_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_llm_to_tts()
    if success:
        print("\n🎉 Pipeline test completed successfully!")
        print("Check the generated audio file to hear the result.")
    else:
        print("\n💥 Pipeline test failed!")
