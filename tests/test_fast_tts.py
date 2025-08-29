"""
Test Edge-TTS for ultra-fast voice generation
Edge-TTS is Microsoft's cloud TTS service with excellent speed and quality
"""
import asyncio
import time
import tempfile
import os

async def test_edge_tts():
    """Test Edge-TTS for ultra-fast voice generation"""
    print('🚀 Testing Edge-TTS for Ultra-Fast Voice Generation')
    print('=' * 60)
    
    try:
        # Install edge-tts if not available
        try:
            import edge_tts
        except ImportError:
            print('📦 Installing edge-tts...')
            import subprocess
            import sys
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'edge-tts'])
            import edge_tts
        
        print('✅ Edge-TTS available')
        
        # List available female voices
        print('\n🎤 Available Female Voices:')
        voices = await edge_tts.list_voices()
        female_voices = [v for v in voices if 'Female' in v['Gender']]
        
        # Show some good English female voices
        good_voices = [
            'en-US-JennyNeural',    # Natural, conversational
            'en-US-AriaNeural',     # Professional, clear
            'en-US-SaraNeural',     # Friendly, warm
            'en-GB-SoniaNeural',    # British accent
            'en-AU-NatashaNeural',  # Australian accent
        ]
        
        for voice in good_voices[:3]:  # Test top 3
            voice_info = next((v for v in voices if v['ShortName'] == voice), None)
            if voice_info:
                print(f"   • {voice}: {voice_info['LocalName']}")
        
        print()
        
        # Test phrases
        test_phrases = [
            "Hello, how can I help you today?",
            "I understand your concern. Let me check your account balance.",
            "Your current balance is two thousand four fifty-seven dollars.",
        ]
        
        # Test each voice
        for voice_name in good_voices[:2]:  # Test 2 voices for speed
            print(f'🎭 Testing voice: {voice_name}')
            
            total_time = 0
            
            for i, text in enumerate(test_phrases, 1):
                print(f'   {i}. "{text}"')
                
                start_time = time.time()
                
                # Generate speech
                communicate = edge_tts.Communicate(text, voice_name)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                await communicate.save(tmp_path)
                
                gen_time = time.time() - start_time
                total_time += gen_time
                
                # Analyze file
                file_size = os.path.getsize(tmp_path) / 1024  # KB
                
                print(f'      ⏱️  Generation: {gen_time:.3f}s')
                print(f'      📁 File size: {file_size:.1f}KB')
                
                if gen_time <= 0.5:
                    print(f'      ✅ ULTRA-FAST (under 0.5s)')
                elif gen_time <= 1.0:
                    print(f'      ✅ REAL-TIME (under 1.0s)')
                else:
                    print(f'      ❌ TOO SLOW ({gen_time:.3f}s)')
                
                # Clean up
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            avg_time = total_time / len(test_phrases)
            print(f'   📊 Average: {avg_time:.3f}s per phrase')
            print()
        
        print('🎯 EDGE-TTS ADVANTAGES:')
        print('   ✅ Ultra-fast generation (0.1-0.5s typically)')
        print('   ✅ High-quality neural voices')
        print('   ✅ No local model storage needed')
        print('   ✅ Many voice options and languages')
        print('   ✅ Free Microsoft service')
        print()
        print('⚠️  CONSIDERATIONS:')
        print('   • Requires internet connection')
        print('   • Depends on Microsoft service availability')
        print('   • Usage limits may apply')
        
    except Exception as e:
        print(f'❌ Edge-TTS test failed: {e}')
        import traceback
        traceback.print_exc()

async def test_system_tts():
    """Test Windows built-in TTS (SAPI) for ultimate speed"""
    print('\n🖥️  Testing Windows System TTS (SAPI)')
    print('=' * 50)
    
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Set voice to female
        voices = engine.getProperty('voices')
        female_voices = [v for v in voices if 'female' in v.name.lower() or 'zira' in v.name.lower()]
        
        if female_voices:
            engine.setProperty('voice', female_voices[0].id)
            print(f'✅ Using voice: {female_voices[0].name}')
        
        # Set speed
        engine.setProperty('rate', 200)  # Words per minute
        
        test_text = "Hello, how can I help you today?"
        print(f'Testing: "{test_text}"')
        
        start_time = time.time()
        
        # Save to file for timing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        engine.save_to_file(test_text, tmp_path)
        engine.runAndWait()
        
        gen_time = time.time() - start_time
        
        print(f'⏱️  Generation: {gen_time:.3f}s')
        print(f'🎯 Result: {"✅ ULTRA-FAST" if gen_time < 0.2 else "✅ FAST" if gen_time < 0.5 else "MODERATE"}')
        
        # Clean up
        try:
            os.unlink(tmp_path)
        except:
            pass
            
        print('✅ Windows SAPI is the fastest local option')
        
    except ImportError:
        print('📦 Installing pyttsx3 for system TTS...')
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyttsx3'])
        print('✅ Installed pyttsx3, please run again')
    except Exception as e:
        print(f'❌ System TTS test failed: {e}')

if __name__ == "__main__":
    asyncio.run(test_edge_tts())
    asyncio.run(test_system_tts())
