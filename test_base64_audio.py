#!/usr/bin/env python3
"""
Test script to decode and verify TTS audio from base64
"""

import requests
import base64
import wave
import os

def test_and_save_tts_audio():
    """Test TTS and save base64 audio to file"""
    
    print("🧪 Testing TTS Audio and Decoding Base64")
    print("=" * 50)
    
    # Test text
    test_text = "Testing audio generation with base64 decoding."
    
    print(f"📝 Text: '{test_text}'")
    
    try:
        # Make TTS request
        response = requests.post(
            'http://localhost:8012/synthesize',
            json={
                'text': test_text,
                'high_quality': True,
                'return_audio': True
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ TTS Request Successful!")
            
            # Get base64 audio
            audio_b64 = result.get('audio_base64')
            if audio_b64:
                print(f"📦 Base64 Audio: {len(audio_b64)} characters")
                
                # Decode base64 to binary
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    print(f"🔧 Decoded to {len(audio_bytes)} bytes")
                    
                    # Save to file
                    output_file = "test_decoded_audio.wav"
                    with open(output_file, 'wb') as f:
                        f.write(audio_bytes)
                    
                    print(f"💾 Saved to: {output_file}")
                    
                    # Analyze the file
                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        print(f"📊 File Size: {file_size} bytes")
                        
                        # Try to analyze with wave module
                        try:
                            with wave.open(output_file, 'rb') as wav_file:
                                frames = wav_file.getnframes()
                                sample_rate = wav_file.getframerate()
                                channels = wav_file.getnchannels()
                                sample_width = wav_file.getsampwidth()
                                
                                duration = frames / sample_rate
                                
                                print(f"🎵 Audio Analysis:")
                                print(f"   - Frames: {frames:,}")
                                print(f"   - Sample Rate: {sample_rate} Hz")
                                print(f"   - Channels: {channels}")
                                print(f"   - Sample Width: {sample_width} bytes")
                                print(f"   - Duration: {duration:.2f} seconds")
                                
                                if frames > 0:
                                    print("✅ Audio file contains valid audio data!")
                                    
                                    # Check if it's mostly silence
                                    audio_data = wav_file.readframes(frames)
                                    import struct
                                    
                                    # Convert to samples (assuming 16-bit audio)
                                    if sample_width == 2:  # 16-bit
                                        samples = struct.unpack(f'<{frames * channels}h', audio_data)
                                        max_amplitude = max(abs(s) for s in samples[:min(10000, len(samples))])  # Check first 10k samples
                                        print(f"   - Max amplitude (first 10k samples): {max_amplitude}")
                                        
                                        if max_amplitude > 100:  # Reasonable threshold for non-silence
                                            print("✅ Audio contains actual speech/sound!")
                                        else:
                                            print("⚠️ Audio appears to be mostly silent")
                                    
                                else:
                                    print("❌ Audio file is empty!")
                                    
                        except Exception as e:
                            print(f"⚠️ Could not analyze audio file: {e}")
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ Failed to decode base64 audio: {e}")
                    return False
            else:
                print("❌ No base64 audio in response")
                return False
        else:
            print(f"❌ TTS Request Failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_and_save_tts_audio()
    print(f"\n🏆 Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
