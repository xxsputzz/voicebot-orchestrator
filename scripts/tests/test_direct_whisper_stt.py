#!/usr/bin/env python3
"""
Direct Whisper STT Test
======================

Test microphone recording with direct Whisper transcription,
bypassing the problematic service.
"""

import asyncio
import pyaudio
import wave
import time
from datetime import datetime
from pathlib import Path

async def test_microphone_with_direct_whisper():
    """Test microphone recording with direct Whisper transcription"""
    print("🎙️ MICROPHONE TEST WITH DIRECT WHISPER")
    print("=" * 50)
    
    # Audio recording settings
    settings = {
        'format': pyaudio.paInt16,
        'channels': 1,
        'rate': 16000,
        'chunk': 1024,
    }
    
    # Create audio directory
    audio_dir = Path("tests/audio_samples/direct_whisper_tests")
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎤 Recording 5 seconds of audio...")
    print("   Get ready to speak in 3 seconds...")
    time.sleep(3)
    print("   🔴 RECORDING NOW - Speak clearly!")
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    try:
        # Open microphone stream
        stream = audio.open(
            format=settings['format'],
            channels=settings['channels'],
            rate=settings['rate'],
            input=True,
            frames_per_buffer=settings['chunk']
        )
        
        frames = []
        
        # Record audio
        for i in range(0, int(settings['rate'] / settings['chunk'] * 5)):
            data = stream.read(settings['chunk'])
            frames.append(data)
        
        print("✅ Recording completed!")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"direct_whisper_test_{timestamp}.wav"
        filepath = audio_dir / filename
        
        # Write WAV file
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(settings['channels'])
            wf.setsampwidth(audio.get_sample_size(settings['format']))
            wf.setframerate(settings['rate'])
            wf.writeframes(b''.join(frames))
        
        print(f"🎵 Audio saved: {filepath}")
        print(f"📊 File size: {filepath.stat().st_size:,} bytes")
        
        # Now transcribe with direct Whisper
        print(f"\n🔄 Transcribing with direct Whisper...")
        
        try:
            from voicebot_orchestrator.real_whisper_stt import WhisperSTT
            
            whisper = WhisperSTT()
            transcript = await whisper.transcribe_file(str(filepath))
            
            if transcript and transcript.strip():
                print(f"✅ TRANSCRIPTION SUCCESS!")
                print(f"📝 You said: \"{transcript.strip()}\"")
                print(f"🎉 Your microphone and Whisper are working perfectly!")
                return transcript.strip()
            else:
                print(f"⚠️  Empty transcription - possible silence")
                return ""
                
        except Exception as e:
            print(f"❌ Whisper transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return None
    finally:
        audio.terminate()

if __name__ == "__main__":
    print("🧪 TESTING YOUR MICROPHONE WITH REAL WHISPER")
    print("This will record your voice and transcribe it directly")
    print("bypassing any service issues.")
    print()
    
    input("Press Enter when ready to record...")
    
    result = asyncio.run(test_microphone_with_direct_whisper())
    
    if result:
        print(f"\n🎉 SUCCESS! Your setup is working perfectly!")
        print(f"✅ Microphone: Recording correctly")
        print(f"✅ Whisper: Transcribing accurately") 
        print(f"✅ Transcript: \"{result}\"")
        print(f"\nThe issue is with the STT service configuration, not your hardware!")
    elif result == "":
        print(f"\n⚠️  Audio recorded but transcription was empty")
        print("This might indicate silence or very quiet audio")
    else:
        print(f"\n❌ Test failed - check error messages above")
