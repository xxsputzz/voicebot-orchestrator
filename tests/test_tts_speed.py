"""
Test TTS speed for real-time conversation requirements
"""
import asyncio
import sys
import os
import time
import tempfile
sys.path.append('..')

async def test_tts_speed():
    """Test current TTS speed for real-time requirements"""
    print('🚀 TTS Speed Analysis for Real-Time Conversation')
    print('=' * 60)
    print()
    
    # Target: Sub-second generation for conversational responses
    target_time = 1.0  # 1 second maximum for real-time feel
    
    test_phrases = [
        "Hello, how can I help you today?",  # Short response
        "I understand your concern. Let me check your account balance for you.",  # Medium response
        "Thank you for calling our banking services. Your current account balance is two thousand four hundred and fifty-seven dollars and thirty-two cents.",  # Long response
    ]
    
    print('1️⃣ Testing Kokoro TTS Speed...')
    print()
    
    try:
        from voicebot_orchestrator.tts import KokoroTTS
        
        # Initialize TTS
        init_start = time.time()
        kokoro = KokoroTTS(voice="af_bella")
        init_time = time.time() - init_start
        
        print(f'✅ Kokoro initialization: {init_time:.3f}s')
        print()
        
        total_gen_time = 0
        total_audio_duration = 0
        
        for i, text in enumerate(test_phrases, 1):
            print(f'{i}. Testing: "{text}"')
            
            gen_start = time.time()
            
            # Generate speech using async method
            audio_bytes = await kokoro.synthesize_speech(text)
            
            gen_time = time.time() - gen_start
            total_gen_time += gen_time
            
            # Save to temp file to analyze duration
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            # Analyze audio duration
            import wave
            try:
                with wave.open(tmp_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / rate
                    total_audio_duration += duration
                    
                    realtime_factor = gen_time / duration
                    
                    print(f'   ⏱️  Generation time: {gen_time:.3f}s')
                    print(f'   🎵 Audio duration: {duration:.3f}s')
                    print(f'   ⚡ Realtime factor: {realtime_factor:.2f}x')
                    
                    if gen_time <= target_time:
                        print(f'   ✅ REAL-TIME CAPABLE (under {target_time}s)')
                    else:
                        print(f'   ❌ TOO SLOW for real-time ({gen_time:.3f}s > {target_time}s)')
                    
                    print()
                    
            except Exception as e:
                print(f'   ❌ Could not analyze audio: {e}')
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        # Summary
        avg_gen_time = total_gen_time / len(test_phrases)
        avg_audio_duration = total_audio_duration / len(test_phrases)
        avg_realtime_factor = avg_gen_time / avg_audio_duration
        
        print('📊 KOKORO SPEED SUMMARY:')
        print('=' * 40)
        print(f'🏁 Average generation time: {avg_gen_time:.3f}s')
        print(f'🎵 Average audio duration: {avg_audio_duration:.3f}s')
        print(f'⚡ Average realtime factor: {avg_realtime_factor:.2f}x')
        print(f'🎯 Real-time capable: {"✅ YES" if avg_gen_time <= target_time else "❌ NO"}')
        
        if avg_gen_time > target_time:
            print(f'💡 Need {target_time/avg_gen_time:.1f}x speedup for real-time')
        
        print()
        
    except Exception as e:
        print(f'❌ Kokoro test failed: {e}')
        import traceback
        traceback.print_exc()
    
    print('🔍 ALTERNATIVE FAST TTS OPTIONS:')
    print('=' * 40)
    print('1. 🚀 Edge-TTS (Microsoft): ~0.1-0.5s generation')
    print('2. 🌐 Google TTS API: ~0.2-0.8s generation')
    print('3. ⚡ Coqui TTS (optimized): ~0.3-1.0s generation')
    print('4. 🏭 Azure Cognitive Services: ~0.1-0.3s generation')
    print('5. 📱 System TTS (Windows SAPI): ~0.05-0.2s generation')
    print()
    print('💡 For sub-second response times, consider cloud TTS APIs')
    print('   or lightweight local models like Edge-TTS.')

if __name__ == "__main__":
    asyncio.run(test_tts_speed())
