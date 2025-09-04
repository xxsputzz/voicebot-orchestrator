#!/usr/bin/env python3
"""
Very simple speech-like generator using basic tone patterns
"""

import numpy as np
import wave

def generate_speech_tones(text, sample_rate=22050):
    """Generate very basic speech-like tones"""
    
    # Define basic frequency patterns for common sounds
    sound_patterns = {
        'a': [440, 880],      # A note and octave
        'e': [330, 660],      # E note
        'i': [523, 1047],     # C note  
        'o': [294, 588],      # D note
        'u': [262, 524],      # C note lower
        'h': [200],           # Low rumble
        'l': [400, 800],      # Medium tones
        'w': [300, 600],      # Lower tones
        'r': [350, 700],      # Rolling sound
        'd': [250],           # Short low tone
        'n': [380, 760],      # Nasal sound
        't': [150],           # Short click
        's': [2000],          # High hiss
        'default': [220, 440] # Default tone
    }
    
    words = text.lower().split()
    audio_segments = []
    
    for word in words:
        word_audio = []
        
        for char in word:
            if char.isalpha():
                # Get frequency pattern for this character
                frequencies = sound_patterns.get(char, sound_patterns['default'])
                
                # Generate tone for this character
                duration = 0.15  # 150ms per character
                samples = int(duration * sample_rate)
                t = np.linspace(0, duration, samples)
                
                char_audio = np.zeros(samples)
                
                # Add each frequency component
                for freq in frequencies:
                    amplitude = 0.3 / len(frequencies)
                    char_audio += amplitude * np.sin(2 * np.pi * freq * t)
                
                # Add envelope for natural sound
                envelope = np.exp(-3 * t) * (1 - np.exp(-10 * t))
                char_audio *= envelope
                
                word_audio.append(char_audio)
        
        # Combine character sounds for this word
        if word_audio:
            word_combined = np.concatenate(word_audio)
            audio_segments.append(word_combined)
            
            # Add pause between words
            pause_duration = 0.2
            pause_samples = int(pause_duration * sample_rate)
            audio_segments.append(np.zeros(pause_samples))
    
    # Combine all segments
    if audio_segments:
        full_audio = np.concatenate(audio_segments)
    else:
        full_audio = np.zeros(int(1.0 * sample_rate))
    
    # Normalize
    max_val = np.max(np.abs(full_audio))
    if max_val > 0:
        full_audio = full_audio / max_val * 0.6
    
    return full_audio

def save_wav(audio_data, sample_rate, filename):
    """Save audio as WAV file"""
    audio_16bit = (audio_data * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_16bit.tobytes())

if __name__ == "__main__":
    print("Generating simple tone-based speech...")
    
    test_text = "hello world test"
    audio = generate_speech_tones(test_text)
    
    save_wav(audio, 22050, "test_tone_speech.wav")
    print(f"Audio saved as test_tone_speech.wav")
    print("This uses musical tones to represent speech sounds")
    print("Should at least sound more melodic and speech-like than digital noise")
