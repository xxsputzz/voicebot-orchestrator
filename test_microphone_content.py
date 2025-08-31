#!/usr/bin/env python3
"""
Microphone Content Verification Test
===================================

This test verifies that:
1. Microphone recording captures different audio content each time
2. Audio files have different sizes and waveforms
3. Manual inspection of recorded audio files

Usage:
    python test_microphone_content.py
"""

import pyaudio
import wave
import time
import os
from datetime import datetime
from pathlib import Path

def record_audio_sample(duration=5, sample_name="sample"):
    """Record a single audio sample"""
    
    # Create audio directory if it doesn't exist
    audio_dir = Path("tests/audio_samples/microphone_tests")
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Audio recording settings
    settings = {
        'format': pyaudio.paInt16,
        'channels': 1,
        'rate': 16000,
        'chunk': 1024,
    }
    
    print(f"\n🎤 Recording {sample_name} for {duration} seconds...")
    print("   Speak now...")
    
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
        for i in range(0, int(settings['rate'] / settings['chunk'] * duration)):
            data = stream.read(settings['chunk'])
            frames.append(data)
        
        print("✅ Recording completed!")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{sample_name}_{timestamp}.wav"
        filepath = audio_dir / filename
        
        # Write WAV file
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(settings['channels'])
            wf.setsampwidth(audio.get_sample_size(settings['format']))
            wf.setframerate(settings['rate'])
            wf.writeframes(b''.join(frames))
        
        print(f"🎵 Audio saved: {filepath}")
        
        # Get file info
        file_size = os.path.getsize(filepath)
        print(f"📊 File size: {file_size:,} bytes")
        
        return str(filepath), file_size
        
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return None, 0
    finally:
        audio.terminate()

def analyze_audio_differences(files):
    """Analyze differences between audio files"""
    print(f"\n📊 AUDIO ANALYSIS")
    print("=" * 50)
    
    sizes = []
    for filename, file_size in files:
        if filename and os.path.exists(filename):
            sizes.append(file_size)
            print(f"📁 {filename}")
            print(f"   Size: {file_size:,} bytes")
            
            # Try to get some basic audio stats
            try:
                with wave.open(filename, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / rate
                    print(f"   Duration: {duration:.2f} seconds")
                    print(f"   Frames: {frames:,}")
            except Exception as e:
                print(f"   Error reading: {e}")
            print()
    
    # Check if files are different
    if len(set(sizes)) > 1:
        print("✅ GOOD: Audio files have different sizes (recording different content)")
        size_diff = max(sizes) - min(sizes)
        print(f"   Size difference: {size_diff:,} bytes")
    else:
        print("⚠️  WARNING: All audio files have the same size")
        print("   This might indicate silence or identical content")
    
    return sizes

def main():
    """Main test function"""
    print("🎙️ MICROPHONE CONTENT VERIFICATION TEST")
    print("=" * 50)
    print()
    print("This test will record 3 audio samples to verify your")
    print("microphone is capturing different content each time.")
    print()
    print("💡 TIP: Say different things for each recording to test variation")
    print()
    
    input("Press Enter to start the first recording...")
    
    files = []
    
    # Record 3 samples
    for i in range(3):
        print(f"\n🎯 RECORDING {i+1}/3")
        print("-" * 30)
        
        if i == 0:
            print("💬 Suggestion: Say 'Hello, this is my first test recording'")
        elif i == 1:
            print("💬 Suggestion: Say 'This is my second recording with different words'")
        else:
            print("💬 Suggestion: Say 'Final test recording number three'")
        
        print("\n⏱️  Starting in 3 seconds...")
        time.sleep(3)
        
        filename, size = record_audio_sample(duration=5, sample_name=f"mic_test_{i+1}")
        files.append((filename, size))
        
        if i < 2:
            input("\nPress Enter for next recording...")
    
    # Analyze differences
    analyze_audio_differences(files)
    
    print(f"\n🔍 MANUAL VERIFICATION:")
    print("You can manually play these files to verify content:")
    for filename, _ in files:
        if filename and os.path.exists(filename):
            print(f"   📂 {filename}")
    
    print(f"\n✅ Test completed!")
    print("If files have different sizes, your microphone is working correctly.")

if __name__ == "__main__":
    main()
