#!/usr/bin/env python3
"""
Simple microphone recording test
"""
import pyaudio
import wave
import time
from pathlib import Path

def test_microphone_recording():
    """Test microphone recording functionality"""
    print("🎤 Testing Microphone Recording")
    print("=" * 40)
    
    # Audio settings
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 3
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    try:
        print(f"🔊 Available audio devices: {audio.get_device_count()}")
        
        # Get default input device
        default_input = audio.get_default_input_device_info()
        print(f"🎙️ Default microphone: {default_input['name']}")
        
        # Open microphone stream
        print(f"\n🎤 Recording for {RECORD_SECONDS} seconds...")
        print("   Speak now!")
        
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        frames = []
        
        # Record audio
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        print("✅ Recording completed!")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Save to file
        filename = "mic_test_recording.wav"
        
        # Write WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        
        print(f"🎵 Audio saved: {filename}")
        
        # Check file size
        file_size = Path(filename).stat().st_size
        print(f"📊 File size: {file_size} bytes")
        
        if file_size > 1000:  # Should be at least 1KB for 3 seconds
            print("✅ Microphone recording test PASSED!")
            return True
        else:
            print("❌ File too small, recording may have failed")
            return False
            
    except Exception as e:
        print(f"❌ Recording test failed: {e}")
        return False
    finally:
        audio.terminate()

if __name__ == "__main__":
    test_microphone_recording()
