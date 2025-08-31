#!/usr/bin/env python3
"""
Test Direct Whisper Transcription
=================================
"""
import asyncio
import sys, os
sys.path.append('.')

async def test_direct_whisper():
    try:
        from voicebot_orchestrator.real_whisper_stt import WhisperSTT
        
        # Initialize the service exactly like the STT service does
        stt = WhisperSTT(model_name='base', device='cpu')
        print(f'✅ WhisperSTT created: _use_real = {stt._use_real}')
        
        # Test with the same file
        audio_file = 'benchmark_kokoro_1.wav'
        print(f'🎵 Testing with: {audio_file}')
        
        # Check if file exists
        if not os.path.exists(audio_file):
            print(f'❌ File not found: {audio_file}')
            return
        
        # Get file size for debugging
        file_size = os.path.getsize(audio_file)
        print(f'📊 File size: {file_size} bytes')
        
        # Transcribe the file
        print('🔄 Starting transcription...')
        result = await stt.transcribe_file(audio_file)
        print(f'📝 Result: "{result}"')
        print(f'📏 Length: {len(result)}')
        
        if result and len(result) > 0:
            print('✅ SUCCESS: Direct transcription works!')
            return True
        else:
            print('❌ PROBLEM: Direct transcription returns empty!')
            return False
            
    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_direct_whisper())
    print(f"\n📊 Final Result: {'SUCCESS' if success else 'FAILED'}")
