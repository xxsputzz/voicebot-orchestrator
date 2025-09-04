#!/usr/bin/env python3
"""
Test script to generate realistic speech audio using a different approach
"""

import numpy as np
import wave
import io

def generate_speech_like_audio(text, duration=3.0):
    """Generate more realistic speech using noise-based synthesis"""
    
    sample_rate = 22050
    samples = int(sample_rate * duration)
    
    # Instead of pure sine waves, use filtered noise which sounds more like speech
    # This is similar to how vocoder synthesizers work
    
    # Generate white noise as base
    noise = np.random.normal(0, 0.2, samples)
    
    # Create speech-like filtering
    # Human speech is mostly in 100-4000 Hz range
    t = np.linspace(0, duration, samples)
    
    # Create pitch contour (like intonation in speech)
    base_pitch = 150  # Hz, typical speech fundamental
    pitch_contour = base_pitch * (1 + 0.2 * np.sin(2 * np.pi * 0.5 * t))
    
    # Create formant-like resonances by amplitude modulating the noise
    # These frequencies are characteristic of human vowels
    formant1 = 800   # First formant
    formant2 = 1200  # Second formant
    formant3 = 2800  # Third formant
    
    # Apply formant filtering by modulating amplitude at formant frequencies
    speech_signal = noise.copy()
    
    # Add formant resonances
    for formant_freq in [formant1, formant2, formant3]:
        formant_mod = 1 + 0.5 * np.sin(2 * np.pi * formant_freq * t / sample_rate)
        speech_signal *= formant_mod
    
    # Add pitch harmonics to give voice-like quality
    for harmonic in range(1, 4):
        harmonic_signal = 0.3 / harmonic * np.sin(2 * np.pi * pitch_contour * harmonic * t)
        speech_signal += harmonic_signal
    
    # Create word-like amplitude envelope
    words = text.split()
    word_duration = duration / max(len(words), 1)
    
    envelope = np.zeros(samples)
    for i, word in enumerate(words[:8]):  # Max 8 words
        word_start = i * word_duration
        word_end = min((i + 1) * word_duration, duration)
        
        start_sample = int(word_start * sample_rate)
        end_sample = int(word_end * sample_rate * 0.9)  # 90% on, 10% pause
        end_sample = min(end_sample, samples)
        
        if end_sample > start_sample:
            word_samples = end_sample - start_sample
            # Create natural word envelope
            word_env = np.ones(word_samples)
            
            # Soft attack/decay
            attack_len = min(word_samples // 10, int(0.02 * sample_rate))
            decay_len = min(word_samples // 8, int(0.03 * sample_rate))
            
            if attack_len > 0:
                word_env[:attack_len] = np.linspace(0.1, 1.0, attack_len)
            if decay_len > 0:
                word_env[-decay_len:] = np.linspace(1.0, 0.1, decay_len)
            
            envelope[start_sample:end_sample] = word_env
    
    # Apply envelope
    speech_signal *= envelope
    
    # Simple low-pass filter to remove harsh frequencies
    # Speech is mostly below 4kHz
    alpha = 0.1  # Simple exponential smoothing
    for i in range(1, len(speech_signal)):
        speech_signal[i] = alpha * speech_signal[i] + (1 - alpha) * speech_signal[i-1]
    
    # Normalize
    max_val = np.max(np.abs(speech_signal))
    if max_val > 0:
        speech_signal = speech_signal / max_val * 0.8
    
    return speech_signal, sample_rate

def save_wav(audio_data, sample_rate, filename):
    """Save audio as WAV file"""
    audio_16bit = (audio_data * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_16bit.tobytes())

if __name__ == "__main__":
    print("Generating speech-like audio using noise-based synthesis...")
    
    test_text = "Hello world, this is a test of speech synthesis"
    audio, sr = generate_speech_like_audio(test_text, duration=4.0)
    
    save_wav(audio, sr, "test_speech_noise_based.wav")
    print(f"Audio saved as test_speech_noise_based.wav")
    print("Please play this file and see if it sounds more like speech!")
    print("")
    print("This uses filtered noise instead of pure sine waves,")
    print("which should sound much more like human speech.")
