#!/usr/bin/env python3
"""
Simple phoneme-based speech synthesis that actually sounds like speech
"""

import numpy as np
import wave

def text_to_phonemes(text):
    """Convert text to simple phonemes"""
    # Very basic phoneme mapping
    phoneme_map = {
        'a': 'ah', 'e': 'eh', 'i': 'ih', 'o': 'oh', 'u': 'uh',
        'b': 'buh', 'c': 'kuh', 'd': 'duh', 'f': 'fuh', 'g': 'guh',
        'h': 'huh', 'j': 'juh', 'k': 'kuh', 'l': 'luh', 'm': 'muh',
        'n': 'nuh', 'p': 'puh', 'q': 'kuh', 'r': 'ruh', 's': 'suh',
        't': 'tuh', 'v': 'vuh', 'w': 'wuh', 'x': 'eks', 'y': 'yuh', 'z': 'zuh'
    }
    
    phonemes = []
    for char in text.lower():
        if char in phoneme_map:
            phonemes.append(phoneme_map[char])
        elif char == ' ':
            phonemes.append('_pause_')
    
    return phonemes

def generate_phoneme_audio(phoneme, duration=0.2, sample_rate=22050):
    """Generate audio for a single phoneme"""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    if phoneme == '_pause_':
        return np.zeros(samples)
    
    # Vowel sounds (use harmonics)
    if phoneme in ['ah', 'eh', 'ih', 'oh', 'uh']:
        # Different formant frequencies for different vowels
        formants = {
            'ah': [730, 1090, 2440],  # 'a' sound
            'eh': [530, 1840, 2480],  # 'e' sound  
            'ih': [390, 1990, 2550],  # 'i' sound
            'oh': [570, 840, 2410],   # 'o' sound
            'uh': [460, 1100, 2240]   # 'u' sound
        }
        
        f0 = 150  # Base frequency
        audio = np.zeros(samples)
        
        # Generate harmonics at formant frequencies
        for i, formant in enumerate(formants[phoneme]):
            amplitude = 0.3 / (i + 1)  # Decreasing amplitude
            audio += amplitude * np.sin(2 * np.pi * formant * t / 100)
        
        # Add fundamental frequency
        audio += 0.5 * np.sin(2 * np.pi * f0 * t)
        
    else:
        # Consonant sounds (use filtered noise)
        audio = np.random.normal(0, 0.2, samples)
        
        # Different filtering for different consonants
        if phoneme.startswith(('s', 'f', 'sh')):
            # Fricatives - high frequency noise
            freq_mod = np.sin(2 * np.pi * 4000 * t)
        elif phoneme.startswith(('b', 'p', 'd', 't', 'g', 'k')):
            # Plosives - short burst
            envelope = np.exp(-10 * t)
            audio *= envelope
            freq_mod = np.sin(2 * np.pi * 200 * t)
        else:
            # Other consonants
            freq_mod = np.sin(2 * np.pi * 800 * t)
        
        audio *= (0.5 + 0.5 * freq_mod)
    
    # Apply envelope
    envelope = np.ones(samples)
    fade_len = min(samples // 10, int(0.01 * sample_rate))
    if fade_len > 0:
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    
    return audio * envelope

def text_to_speech(text, sample_rate=22050):
    """Convert text to speech audio"""
    print(f"Converting to speech: '{text}'")
    
    # Simple word-based approach
    words = text.lower().split()
    audio_segments = []
    
    for word in words:
        print(f"Processing word: {word}")
        
        # Generate audio for each letter/sound
        for char in word:
            if char.isalpha():
                # Generate phoneme audio
                if char in 'aeiou':
                    # Vowels get longer duration
                    phoneme_audio = generate_phoneme_audio(char + 'h', duration=0.15)
                else:
                    # Consonants get shorter duration
                    phoneme_audio = generate_phoneme_audio(char + 'uh', duration=0.1)
                
                audio_segments.append(phoneme_audio)
        
        # Add pause between words
        pause = np.zeros(int(0.1 * sample_rate))
        audio_segments.append(pause)
    
    # Combine all segments
    if audio_segments:
        full_audio = np.concatenate(audio_segments)
    else:
        full_audio = np.zeros(int(1.0 * sample_rate))
    
    # Normalize
    max_val = np.max(np.abs(full_audio))
    if max_val > 0:
        full_audio = full_audio / max_val * 0.7
    
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
    print("Generating phoneme-based speech synthesis...")
    
    test_text = "hello world"
    audio = text_to_speech(test_text)
    
    save_wav(audio, 22050, "test_phoneme_speech.wav")
    print(f"Audio saved as test_phoneme_speech.wav")
    print("This should sound much more like actual speech!")
