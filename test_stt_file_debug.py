#!/usr/bin/env python3
"""
Debug STT Service File Handling
===============================
"""
import tempfile
import os
import shutil
import asyncio
import sys
sys.path.append('.')

async def test_stt_file_handling():
    """Test exactly what the STT service does with file uploads"""
    
    try:
        from voicebot_orchestrator.real_whisper_stt import WhisperSTT
        
        # Initialize the service exactly like the STT service does
        whisper_stt = WhisperSTT(model_name='base', device='cpu')
        print(f'✅ WhisperSTT created: _use_real = {whisper_stt._use_real}')
        
        # Original file
        original_file = 'benchmark_kokoro_1.wav'
        if not os.path.exists(original_file):
            print(f'❌ Original file not found: {original_file}')
            return False
        
        print(f'📂 Original file: {original_file}')
        print(f'📊 Original size: {os.path.getsize(original_file)} bytes')
        
        # Read the file like FastAPI would
        with open(original_file, 'rb') as f:
            audio_data = f.read()
        
        print(f'📊 Read data size: {len(audio_data)} bytes')
        
        # Save to temp file exactly like the STT service does
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        print(f'📂 Temp file: {temp_path}')
        print(f'📊 Temp size: {os.path.getsize(temp_path)} bytes')
        
        # Compare file contents
        with open(temp_path, 'rb') as f:
            temp_data = f.read()
        
        if audio_data == temp_data:
            print('✅ File contents match perfectly')
        else:
            print('❌ File contents DO NOT match!')
            print(f'   Original: {len(audio_data)} bytes')
            print(f'   Temp: {len(temp_data)} bytes')
        
        # Now transcribe using the temp file
        print('🔄 Transcribing temp file...')
        result = await whisper_stt.transcribe_file(temp_path)
        print(f'📝 Transcription result: "{result}"')
        print(f'📏 Result length: {len(result)}')
        
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print('🗑️  Temp file cleaned up')
        
        if result and len(result) > 0:
            print('✅ SUCCESS: Temp file transcription works!')
            return True
        else:
            print('❌ PROBLEM: Temp file transcription returns empty!')
            return False
        
    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Testing STT Service File Handling")
    print("=" * 40)
    
    success = asyncio.run(test_stt_file_handling())
    print(f"\n📊 Final Result: {'SUCCESS' if success else 'FAILED'}")
