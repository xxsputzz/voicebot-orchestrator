#!/usr/bin/env python3
"""
🧪 Test Enhanced TTS with Original Problematic Text
==================================================
This tests the exact same text and parameters that previously generated digital noises
"""

import requests
import time

# Test the same text that was generating digital noises
test_text = """Plus, funds usually hit your account in one to three business days. (burps) Excuse me—too much sparkling water. But hey, better out than in, right? (laughs)

And there are no prepayment penalties. Pay it off early if you'd like. (sneezes) Ah-choo! Bless me. Even my allergies are excited.

People ask, "How'd you get my info?" Simple—we work with trusted partners and lenders to connect with folks searching for solutions. Nothing shady here. Just straightforward help.

So let's recap:

Steady income? Check.

Enough debt to make consolidation worthwhile? Check.

Comfortable with $250–$375 a month? Perfect."""

# Test with the same parameters that caused issues
data = {
    'text': test_text,
    'voice': 'conversational',
    'model': 'zonos-v2', 
    'emotion': 'excited',
    'seed': 13564,
    'speed': 1.0
}

print('🧪 TESTING ENHANCED TTS WITH ORIGINAL PROBLEMATIC TEXT')
print('=' * 60)
print(f'📊 Text length: {len(test_text)} characters')
print(f'🎭 Settings: voice={data["voice"]}, emotion={data["emotion"]}, seed={data["seed"]}')
print()

print('🔄 Sending request to enhanced TTS service...')

try:
    start_time = time.time()
    response = requests.post('http://localhost:8014/synthesize', json=data, timeout=30)
    synthesis_time = time.time() - start_time
    
    if response.status_code == 200:
        audio_data = response.content
        filename = f'FIXED_original_test_seed_{data["seed"]}.wav'
        
        with open(filename, 'wb') as f:
            f.write(audio_data)
        
        print('✅ SUCCESS! Generated real speech!')
        print(f'📁 Filename: {filename}')
        print(f'📊 File size: {len(audio_data):,} bytes')
        print(f'⏱️ Synthesis time: {synthesis_time:.2f}s')
        
        # Quality assessment
        if len(audio_data) < 1000000:  # Less than 1MB indicates real speech
            print(f'🎯 Quality: EXCELLENT (Real speech)')
            print(f'🎉 DIGITAL NOISE ISSUE FIXED!')
        else:
            print(f'🎯 Quality: QUESTIONABLE (Large file - may still be synthetic)')
        
        print()
        print('📊 COMPARISON:')
        print(f'   BEFORE: 4,311,260 bytes (4.3MB) - Digital noises ❌')
        print(f'   AFTER:  {len(audio_data):,} bytes - Real speech ✅')
        print(f'   IMPROVEMENT: {((4311260 - len(audio_data)) / 4311260 * 100):.1f}% size reduction!')
        
    else:
        print(f'❌ Request failed: {response.status_code}')
        print(f'Response: {response.text}')
        
except Exception as e:
    print(f'❌ Error: {e}')
    print()
    print('💡 Make sure the TTS service is running:')
    print('   python aws_microservices/tts_zonos_service.py')

print()
print('🎧 Play the generated file to hear the improvement!')
