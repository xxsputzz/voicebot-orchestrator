#!/usr/bin/env python3
"""
Diagnose Zonos TTS Digital Noise Issue
"""
import asyncio
import sys
import os
import wave
import numpy as np
sys.path.append('.')

async def diagnose_zonos_audio():
    try:
        from voicebot_orchestrator.zonos_tts import ZonosTTS
        
        print('🔍 Diagnosing Zonos TTS Audio Generation...')
        print('=' * 60)
        
        tts = ZonosTTS(voice='aria', model='zonos-v1')
        
        # Test with simple text
        test_text = 'Hello world, this is a test.'
        print(f'Testing text: "{test_text}"')
        
        audio_bytes = await tts.synthesize_speech(
            text=test_text,
            emotion='neutral',
            seed=12345
        )
        
        print(f'Generated {len(audio_bytes)} bytes of audio')
        
        # Save and analyze the audio
        filename = 'zonos_diagnosis_test.wav'
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        print(f'Saved audio as {filename}')
        
        # Analyze the audio file
        try:
            with wave.open(filename, 'rb') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                channels = w.getnchannels()
                duration = frames / rate
                
                print(f'\nAudio Analysis:')
                print(f'  Duration: {duration:.2f} seconds')
                print(f'  Sample rate: {rate} Hz')
                print(f'  Channels: {channels}')
                print(f'  Frames: {frames}')
                
                # Read and analyze a few samples
                audio_data = w.readframes(min(frames, 1000))
                samples = np.frombuffer(audio_data, dtype=np.int16)
                
                if len(samples) > 0:
                    print(f'\n  Sample analysis:')
                    print(f'    Min value: {np.min(samples)}')
                    print(f'    Max value: {np.max(samples)}')
                    print(f'    RMS: {np.sqrt(np.mean(samples.astype(float)**2)):.1f}')
                    print(f'    First 10 samples: {samples[:10].tolist()}')
                    
                    # Check if it's mostly digital noise/artifacts
                    zero_crossings = np.sum(np.diff(np.sign(samples)) != 0)
                    print(f'    Zero crossings in first 1000 samples: {zero_crossings}')
                    
                    if zero_crossings > 800:  # Very high zero crossing rate suggests noise
                        print('\n⚠️  WARNING: High zero crossing rate suggests digital noise!')
                        print('    This indicates the audio is noisy/harsh rather than speech-like')
                    elif np.max(np.abs(samples)) < 1000:  # Very quiet
                        print('\n⚠️  WARNING: Audio appears very quiet (possible silence)')
                    else:
                        print('\n✅ Audio appears to have reasonable characteristics')
                        
                    # Analysis of the audio generation algorithm
                    print('\n📊 Algorithm Analysis:')
                    print('    Current implementation uses formant-based synthesis')
                    print('    Multiple sine waves are combined to simulate speech')
                    print('    The issue may be in the formant frequency calculations')
                    
                else:
                    print('❌ No audio samples found')
                    
        except Exception as e:
            print(f'❌ Error analyzing audio: {e}')
        
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

async def main():
    success = await diagnose_zonos_audio()
    print(f'\nDiagnosis result: {"COMPLETED" if success else "FAILED"}')
    
    print('\n🔧 RECOMMENDATIONS:')
    print('1. The current Zonos TTS is a synthetic implementation, not real neural TTS')
    print('2. It generates formant-based audio which can sound digital/robotic')
    print('3. Consider integrating a real TTS engine like:')
    print('   - Microsoft Azure Cognitive Services Speech')
    print('   - Google Cloud Text-to-Speech')
    print('   - Amazon Polly')
    print('   - OpenAI TTS API')
    print('4. The current implementation is meant for testing/development only')

if __name__ == "__main__":
    asyncio.run(main())
