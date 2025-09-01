#!/usr/bin/env python3
"""
Simple LLM to TTS Test to Reproduce Emoji Issue
"""
import requests
import json

def test_specific_llm_to_tts():
    """Test the specific case that's failing"""
    print("🧪 Testing LLM to TTS with 'steal ball bearings'")
    print("=" * 50)
    
    # Step 1: Test LLM
    print("📝 LLM Input: 'steal ball bearings'")
    print("🔄 Sending to LLM: http://localhost:8022")
    
    try:
        llm_response = requests.post(
            'http://localhost:8022/generate', 
            json={
                'text': 'steal ball bearings',
                'temperature': 0.7,
                'max_tokens': 100
            },
            timeout=30
        )
        
        if llm_response.status_code == 200:
            llm_result = llm_response.json()
            llm_text = llm_result.get('response', '')
            print(f"✅ LLM Output: '{llm_text[:100]}...'")
            print(f"Full LLM text repr: {repr(llm_text)}")
            
            # Check if LLM text contains emojis
            if any(ord(char) > 127 for char in llm_text):
                print("⚠️ WARNING: LLM output contains non-ASCII characters!")
                for i, char in enumerate(llm_text):
                    if ord(char) > 127:
                        print(f"   Position {i}: {repr(char)} (U+{ord(char):04X})")
            else:
                print("✅ LLM output is clean ASCII")
            
            # Step 2: Test TTS with clean text
            print("\n🔄 Sending to TTS: http://localhost:8012")
            
            # Make sure we're sending clean text
            clean_llm_text = llm_text
            
            tts_response = requests.post(
                'http://localhost:8012/synthesize', 
                json={
                    'text': clean_llm_text,
                    'voice': 'af_bella', 
                    'return_audio': True
                },
                timeout=60
            )
            
            if tts_response.status_code == 200:
                result = tts_response.json()
                if result.get('audio_base64'):
                    print("✅ SUCCESS: TTS worked!")
                    print(f"Audio size: {len(result['audio_base64'])} characters (base64)")
                else:
                    print("❌ No audio data received from TTS")
            else:
                print(f"❌ TTS request failed: {tts_response.status_code}")
                print(f"   Response: {tts_response.text}")
                
                # Check if the request body has emojis
                request_text = clean_llm_text
                print(f"   Request text repr: {repr(request_text)}")
                
        else:
            print(f"❌ LLM failed: {llm_response.status_code}")
            print(f"Response: {llm_response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_specific_llm_to_tts()
