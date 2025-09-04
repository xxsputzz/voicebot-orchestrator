#!/usr/bin/env python3
"""
Debug Audio Length - Analyze TTS output files
"""
import wave
import os
from pathlib import Path

def analyze_audio_file(filepath):
    """Analyze WAV file properties"""
    try:
        with wave.open(str(filepath), 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            duration_seconds = frames / sample_rate
            file_size = os.path.getsize(filepath)
            
            print(f"📁 File: {filepath.name}")
            print(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"🎵 Duration: {duration_seconds:.2f} seconds")
            print(f"🔊 Sample rate: {sample_rate:,} Hz")
            print(f"📻 Channels: {channels}")
            print(f"🎚️ Sample width: {sample_width} bytes")
            print(f"📈 Total frames: {frames:,}")
            print(f"💾 Bitrate: {(file_size * 8) / duration_seconds / 1000:.1f} kbps")
            
            return duration_seconds
            
    except Exception as e:
        print(f"❌ Error analyzing {filepath}: {e}")
        return None

def main():
    """Analyze recent TTS audio files"""
    audio_dir = Path("tests/audio_samples/interactive_pipeline")
    
    if not audio_dir.exists():
        print(f"❌ Directory not found: {audio_dir}")
        return
    
    # Find all WAV files sorted by modification time (newest first)
    wav_files = list(audio_dir.glob("*.wav"))
    wav_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not wav_files:
        print("❌ No WAV files found")
        return
    
    print("🎧 Audio File Analysis")
    print("=" * 50)
    
    # Analyze the 3 most recent files
    for i, wav_file in enumerate(wav_files[:3]):
        print(f"\n{i+1}. Recent File Analysis:")
        duration = analyze_audio_file(wav_file)
        
        if duration and duration < 10:
            print(f"⚠️  WARNING: Short duration ({duration:.2f}s) - possible truncation!")
        
        print("-" * 40)

if __name__ == "__main__":
    main()
