#!/usr/bin/env python3
"""
Analyze Zonos audio files to identify digital artifacts
"""
import os
import wave
import numpy as np

def analyze_audio_file(filename):
    """Analyze audio file for artifacts"""
    if not os.path.exists(filename):
        print(f"File {filename} not found")
        return
    
    print(f'\n=== Analysis of {filename} ===')
    
    try:
        with wave.open(filename, 'rb') as w:
            frames = w.getnframes()
            rate = w.getframerate()
            duration = frames / rate
            channels = w.getnchannels()
            sample_width = w.getsampwidth()
            
            print(f'Duration: {duration:.2f} seconds')
            print(f'Sample rate: {rate} Hz')
            print(f'Channels: {channels}')
            print(f'Sample width: {sample_width} bytes')
            print(f'Total frames: {frames}')
            
            # Read audio data
            audio_data = w.readframes(frames)
        
        # Convert to numpy array for analysis
        if sample_width == 2:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        else:
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
        # Check beginning and end for silence/artifacts
        start_samples = audio_array[:min(2000, len(audio_array))]
        end_samples = audio_array[-min(2000, len(audio_array)):]
        
        start_max = np.max(np.abs(start_samples))
        end_max = np.max(np.abs(end_samples))
        overall_max = np.max(np.abs(audio_array))
        
        print(f'Start amplitude (first 2000): {start_max} ({start_max/overall_max*100:.1f}% of max)')
        print(f'End amplitude (last 2000): {end_max} ({end_max/overall_max*100:.1f}% of max)')
        print(f'Overall max amplitude: {overall_max}')
        
        # Check for sudden jumps that might indicate artifacts
        if len(audio_array) > 100:
            start_rms = np.sqrt(np.mean(start_samples**2))
            end_rms = np.sqrt(np.mean(end_samples**2))
            overall_rms = np.sqrt(np.mean(audio_array**2))
            
            print(f'Start RMS: {start_rms:.1f} ({start_rms/overall_rms*100:.1f}% of avg)')
            print(f'End RMS: {end_rms:.1f} ({end_rms/overall_rms*100:.1f}% of avg)')
            
            # Look for silence threshold
            silence_threshold = overall_max * 0.01  # 1% of max
            start_silence = np.sum(np.abs(start_samples) < silence_threshold)
            end_silence = np.sum(np.abs(end_samples) < silence_threshold)
            
            print(f'Silent samples at start: {start_silence}/{len(start_samples)} ({start_silence/len(start_samples)*100:.1f}%)')
            print(f'Silent samples at end: {end_silence}/{len(end_samples)} ({end_silence/len(end_samples)*100:.1f}%)')
            
            # Check for digital artifacts (sudden spikes)
            diff = np.diff(audio_array.astype(float))
            large_jumps = np.abs(diff) > (overall_max * 0.5)
            if np.any(large_jumps):
                jump_positions = np.where(large_jumps)[0]
                print(f'⚠️  Found {len(jump_positions)} large amplitude jumps (possible digital artifacts)')
                if len(jump_positions) <= 5:
                    for pos in jump_positions[:5]:
                        print(f'   Jump at sample {pos} ({pos/rate:.3f}s)')
            else:
                print('✅ No large amplitude jumps detected')
        
    except Exception as e:
        print(f'Error analyzing {filename}: {e}')

def main():
    """Main analysis function"""
    print("🎵 Zonos Audio Artifact Analysis")
    print("=" * 50)
    
    # Check for recent test files
    test_files = [
        'test_zonos_direct.wav',
        'test_zonos_phonemes.wav', 
        'test_zonos_enhanced_phonemes.wav',
        'zonos_neural_seed_755633_professional_happy_20250901_135910.wav'
    ]
    
    found_files = []
    for filename in test_files:
        if os.path.exists(filename):
            found_files.append(filename)
    
    if not found_files:
        print("No Zonos test files found. Looking for any .wav files...")
        wav_files = [f for f in os.listdir('.') if f.endswith('.wav') and 'zonos' in f.lower()]
        found_files = wav_files[:3]  # Analyze first 3
    
    if not found_files:
        print("No Zonos audio files found for analysis")
        return
    
    for filename in found_files:
        analyze_audio_file(filename)
    
    print("\n" + "=" * 50)
    print("🔍 ARTIFACT DETECTION SUMMARY")
    print("=" * 50)
    print("Look for:")
    print("- High amplitude at start/end (>50% of max) = Header artifacts")
    print("- Large amplitude jumps = Digital noise/glitches")
    print("- Very low silent percentage = No proper padding")
    print("- RMS spikes at start/end = Encoding artifacts")

if __name__ == "__main__":
    main()
