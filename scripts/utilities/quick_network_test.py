#!/usr/bin/env python3
"""
Quick STT Network Test
=====================
"""
import requests

print('🧪 Testing STT service with benchmark_kokoro_1.wav...')

try:
    with open('benchmark_kokoro_1.wav', 'rb') as f:
        files = {'audio': f}
        response = requests.post('http://localhost:8003/transcribe', files=files, timeout=60)
    
    print(f'📊 Status: {response.status_code}')
    
    if response.status_code == 200:
        result = response.json()
        print(f'📋 Full Response: {result}')
        
        text = result.get('text', '')
        confidence = result.get('confidence', 0.0)
        processing_time = result.get('processing_time_seconds', 0.0)
        
        print(f'📝 Text: "{text}"')
        print(f'📏 Length: {len(text)}')
        print(f'🎯 Confidence: {confidence}')
        print(f'⏱️  Processing time: {processing_time:.2f}s')
        
        if text and len(text) > 0:
            print('✅ SUCCESS: STT working!')
        else:
            print('❌ FAILED: Empty transcript!')
            
    else:
        print(f'❌ Error {response.status_code}: {response.text}')
        
except Exception as e:
    print(f'❌ Test failed: {e}')
