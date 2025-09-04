#!/usr/bin/env python3
"""
Comprehensive STT Debug Test
============================
Test the specific recorded audio file to see what's happening
"""
import asyncio
import sys
import os
from pathlib import Path
import requests
import wave

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

async def test_recorded_audio():
    """Test the specific recorded audio file"""
    
    # Find the most recent recorded audio file
    audio_files = list(Path('.').glob('recorded_audio_*.wav'))
    if not audio_files:
        print("❌ No recorded audio files found")
        return
    
    # Get the most recent file
    latest_file = max(audio_files, key=lambda f: f.stat().st_mtime)
    print(f"🎵 Testing latest recorded file: {latest_file}")
    
    # Check file properties
    try:
        file_size = latest_file.stat().st_size
        print(f"📊 File size: {file_size} bytes")
        
        # Check WAV properties
        with wave.open(str(latest_file), 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            duration = frames / sample_rate
            
            print(f"🎵 Audio properties:")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Sample rate: {sample_rate} Hz")
            print(f"   Channels: {channels}")
            print(f"   Sample width: {sample_width} bytes")
            print(f"   Frames: {frames}")
            
            if duration < 0.5:
                print("⚠️  WARNING: Audio too short (< 0.5 seconds)")
            if sample_rate != 16000:
                print(f"⚠️  WARNING: Sample rate is {sample_rate}, expected 16000")
            if channels != 1:
                print(f"⚠️  WARNING: Not mono audio ({channels} channels)")
                
    except Exception as e:
        print(f"❌ Error reading WAV file: {e}")
        return
    
    # Test 1: Direct Whisper transcription
    print(f"\n🧪 Test 1: Direct Whisper Transcription")
    print("-" * 40)
    
    try:
        from voicebot_orchestrator.real_whisper_stt import WhisperSTT
        
        stt = WhisperSTT(model_name='base', device='cpu')
        print(f"✅ STT created: _use_real = {stt._use_real}")
        
        if stt._use_real:
            result = await stt.transcribe_file(str(latest_file))
            print(f"📝 Direct result: '{result}' (length: {len(result)})")
            
            if result and len(result) > 0:
                print("✅ SUCCESS: Direct transcription works!")
            else:
                print("❌ FAILED: Direct transcription returns empty!")
        else:
            print("❌ STT using mock implementation!")
            
    except Exception as e:
        print(f"❌ Direct transcription failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Network STT service
    print(f"\n🧪 Test 2: Network STT Service")
    print("-" * 40)
    
    try:
        # Check if service is running
        health_response = requests.get('http://localhost:8003/health', timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Service healthy: {health_data}")
            
            # Test transcription
            with open(latest_file, 'rb') as f:
                files = {'audio': f}
                response = requests.post('http://localhost:8003/transcribe', files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                transcript = result.get('text', '').strip()
                confidence = result.get('confidence', 0.0)
                processing_time = result.get('processing_time_seconds', 0.0)
                
                print(f"📝 Network result: '{transcript}' (length: {len(transcript)})")
                print(f"🎯 Confidence: {confidence}")
                print(f"⏱️  Processing time: {processing_time:.2f}s")
                
                if transcript and len(transcript) > 0:
                    print("✅ SUCCESS: Network transcription works!")
                else:
                    print("❌ FAILED: Network transcription returns empty!")
                    print(f"📋 Full response: {result}")
            else:
                print(f"❌ Network request failed: {response.status_code}")
                print(f"   Response: {response.text}")
        else:
            print(f"❌ Service not healthy: {health_response.status_code}")
            
    except Exception as e:
        print(f"❌ Network test failed: {e}")
        
    # Test 3: Alternative audio files
    print(f"\n🧪 Test 3: Test with known good audio")
    print("-" * 40)
    
    # Find other audio files to compare
    other_files = list(Path('.').glob('*kokoro*.wav')) + list(Path('.').glob('benchmark*.wav'))
    if other_files:
        test_file = other_files[0]
        print(f"🎵 Testing known good file: {test_file}")
        
        try:
            # Test direct transcription
            from voicebot_orchestrator.real_whisper_stt import WhisperSTT
            stt = WhisperSTT(model_name='base', device='cpu')
            
            if stt._use_real:
                result = await stt.transcribe_file(str(test_file))
                print(f"📝 Known good result: '{result}' (length: {len(result)})")
                
                if result and len(result) > 0:
                    print("✅ SUCCESS: Known good audio works!")
                    print("🔍 CONCLUSION: Issue is with recorded audio quality/format")
                else:
                    print("❌ FAILED: Even known good audio fails!")
                    print("🔍 CONCLUSION: Issue is with Whisper setup")
            
        except Exception as e:
            print(f"❌ Known good test failed: {e}")
    else:
        print("⚠️  No other audio files found for comparison")

if __name__ == "__main__":
    print("🔍 Comprehensive STT Debug Test")
    print("=" * 50)
    
    asyncio.run(test_recorded_audio())
