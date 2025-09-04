#!/usr/bin/env python3
"""
Test jlaw voice with GPU acceleration fixes
"""

print('🧪 Testing jlaw voice with GPU synthesis fix...')

from tortoise_tts_implementation_real import TortoiseTTS

# Create TTS with GPU
tts = TortoiseTTS(device='cuda')

# Test with shorter text and ultra_fast preset
test_text = 'Hello, this is jlaw voice with GPU acceleration.'

print(f'Testing text: {test_text}')
print('Starting synthesis...')

audio, metadata = tts.synthesize(test_text, voice='jlaw', preset='ultra_fast', save_audio=True)

print('✅ Synthesis completed!')
print(f'Audio shape: {audio.shape}')
print(f'Duration: {metadata.get("duration")}s')
print('🎉 GPU synthesis working!')
