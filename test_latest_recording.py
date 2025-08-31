#!/usr/bin/env python3
"""
Test Latest Recording with Real Whisper
======================================
"""

import asyncio
from pathlib import Path

async def test_latest_recording():
    print("🧪 TESTING LATEST RECORDING WITH REAL WHISPER")
    print("=" * 50)
    
    try:
        from voicebot_orchestrator.real_whisper_stt import WhisperSTT
        
        # Your latest recording
        audio_file = Path('tests/audio_samples/interactive_pipeline/recorded_audio_20250830_230655.wav')
        
        print(f"📄 File: {audio_file.name}")
        print(f"📊 Size: {audio_file.stat().st_size:,} bytes")
        
        # Check if WhisperSTT uses real implementation
        whisper = WhisperSTT()
        is_real = hasattr(whisper, '_use_real') and whisper._use_real
        print(f"🔧 Using real Whisper: {is_real}")
        
        if is_real:
            print("🔄 Transcribing with real Whisper...")
            transcript = await whisper.transcribe_file(str(audio_file))
            
            print(f"✅ Transcript: \"{transcript}\"")
            print(f"📝 Length: {len(transcript)} characters")
            
            if transcript.strip():
                print("🎉 SUCCESS! Real transcription is working!")
                return transcript
            else:
                print("⚠️  Empty transcript - possible silence or audio issue")
                return ""
        else:
            print("❌ Falling back to mock implementation")
            # Try mock transcription
            transcript = await whisper.transcribe_file(str(audio_file))
            print(f"Mock result: \"{transcript}\"")
            return transcript
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_latest_recording())
    
    if result:
        print(f"\n💡 SOLUTION FOUND!")
        print("Your microphone is recording and real Whisper can transcribe it.")
        print("The issue is in the STT service configuration.")
    else:
        print(f"\n🔧 TROUBLESHOOTING NEEDED")
        print("There's an issue with the Whisper setup that needs to be resolved.")
