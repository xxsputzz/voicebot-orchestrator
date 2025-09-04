#!/usr/bin/env python3

import numpy as np
import io
import wave

def generate_simple_speech_audio(text, voice_config, sample_rate=22050):
    """Generate simple but realistic speech-like audio"""
    words = text.split()
    if not words:
        words = ["hello"]
    
    # Calculate duration: ~2.5 words per second + pauses
    words_per_second = 2.5
    duration = len(words) / words_per_second + 0.5
    duration = max(1.0, min(duration, 20.0))
    
    samples = int(sample_rate * duration)
    audio = np.zeros(samples)
    
    # Voice characteristics
    base_freq = voice_config.get("base_freq", 200)
    if "female" in voice_config.get("style", ""):
        base_freq = max(base_freq, 180)
    else:
        base_freq = min(base_freq, 160)
    
    # Generate audio for each word
    word_duration = duration / len(words)
    
    for word_idx, word in enumerate(words):
        word_start = int(word_idx * word_duration * sample_rate)
        word_samples = int(word_duration * 0.9 * sample_rate)  # 90% sound, 10% silence
        word_end = min(word_start + word_samples, samples)
        
        if word_end <= word_start:
            continue
            
        # Generate word audio
        t = np.linspace(0, word_samples/sample_rate, word_samples)
        
        # Check if word has vowels (more musical) or consonants (more noisy)
        vowels = set('aeiouAEIOU')
        has_vowels = any(c in vowels for c in word)
        
        if has_vowels:
            # Vowel-rich words: generate harmonic content
            f0 = base_freq * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))  # Add vibrato
            
            # Generate fundamental + harmonics
            word_audio = np.zeros(len(t))
            for harmonic in range(1, 5):
                amplitude = 0.8 / harmonic
                word_audio += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)
            
            # Add slight noise for realism
            word_audio += np.random.normal(0, 0.05, len(t))
            
        else:
            # Consonant-heavy words: more noise-based
            word_audio = np.random.normal(0, 0.4, len(t))
            # Add some tonal component
            tone = 0.3 * np.sin(2 * np.pi * base_freq * 0.8 * t)
            word_audio += tone
        
        # Apply envelope (attack and decay)
        envelope = np.ones(len(t))
        fade_len = min(len(t) // 10, int(0.05 * sample_rate))
        if fade_len > 0:
            envelope[:fade_len] = np.linspace(0.2, 1.0, fade_len)
            envelope[-fade_len:] = np.linspace(1.0, 0.2, fade_len)
        
        word_audio *= envelope
        
        # Add to main audio
        audio[word_start:word_end] = word_audio[:word_end-word_start]
    
    # Apply simple low-pass filter to sound more natural
    alpha = 0.3
    for i in range(1, len(audio)):
        audio[i] = alpha * audio[i] + (1 - alpha) * audio[i-1]
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.8
    
    # Convert to WAV
    audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.read()

# Test
if __name__ == "__main__":
    voice_config = {"base_freq": 200, "style": "female"}
    audio_data = generate_simple_speech_audio("Hello world testing speech", voice_config)
    
    with open("test_speech.wav", "wb") as f:
        f.write(audio_data)
    
    print(f"Generated {len(audio_data)} bytes of audio")
    print("Saved as test_speech.wav")
