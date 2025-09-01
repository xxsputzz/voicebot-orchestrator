#!/usr/bin/env python3
"""
🧪 ZONOS TTS FIX VERIFICATION
============================
Tests the enhanced Zonos TTS to verify it produces real speech instead of digital noises
"""

import asyncio
import os
import sys
import time
from pathlib import Path

def test_enhanced_zonos():
    """Test the enhanced Zonos TTS functionality"""
    
    print("🧪 TESTING ENHANCED ZONOS TTS")
    print("=" * 50)
    
    try:
        # Import the enhanced TTS
        from voicebot_orchestrator.zonos_tts import ZonosTTS, get_tts_status
        
        print("✅ Enhanced Zonos TTS imported successfully!")
        
        # Check TTS status
        status = get_tts_status()
        print(f"\n📊 TTS Status:")
        print(f"   Real TTS Available: {status['real_tts_available']}")
        print(f"   Available Engines: {status['engines']}")
        print(f"   Recommendation: {status['recommendation']}")
        
        # Initialize TTS
        tts = ZonosTTS()
        
        # Get available options
        voices = tts.get_available_voices()
        emotions = tts.get_available_emotions()
        
        print(f"\n🎭 Available Options:")
        print(f"   Voices: {len(voices)} ({', '.join(voices[:5])}...)")
        print(f"   Emotions: {len(emotions)} ({', '.join(emotions[:5])}...)")
        
        return tts, status
        
    except ImportError as e:
        print(f"❌ Failed to import enhanced TTS: {e}")
        print("\n💡 Run 'fix_zonos_tts.bat' to install the enhancement")
        return None, None
    except Exception as e:
        print(f"❌ Error initializing TTS: {e}")
        return None, None

async def test_synthesis(tts):
    """Test actual speech synthesis"""
    
    print("\n🎵 TESTING SPEECH SYNTHESIS")
    print("-" * 30)
    
    test_cases = [
        {
            "text": "Hello! This is a test of the enhanced Zonos TTS. You should hear natural speech, not digital noises.",
            "voice": "aria",
            "emotion": "conversational",
            "description": "Basic test with female voice"
        },
        {
            "text": "The quick brown fox jumps over the lazy dog. This sentence contains many different sounds.",
            "voice": "professional",
            "emotion": "calm",
            "description": "Male professional voice test"
        },
        {
            "text": "Excellent! The TTS is working perfectly with real neural speech synthesis!",
            "voice": "sophia",
            "emotion": "excited",
            "description": "Excited female voice test"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['description']}")
        print(f"   Voice: {test_case['voice']}")
        print(f"   Emotion: {test_case['emotion']}")
        print(f"   Text: {test_case['text'][:50]}...")
        
        try:
            start_time = time.time()
            
            audio_data = await tts.synthesize_speech(
                text=test_case['text'],
                voice=test_case['voice'],
                emotion=test_case['emotion'],
                speed=1.0,
                seed=12345
            )
            
            synthesis_time = time.time() - start_time
            
            # Save test file
            filename = f"test_enhanced_zonos_{i}_{test_case['voice']}_{int(time.time())}.wav"
            filepath = Path("tests") / "audio_samples" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            
            # Analyze result
            file_size = len(audio_data)
            duration_estimate = len(test_case['text']) * 0.08  # Rough estimate
            
            result = {
                "test": i,
                "success": True,
                "file_size": file_size,
                "synthesis_time": synthesis_time,
                "filename": str(filepath),
                "quality_check": "PASS" if file_size > 50000 else "QUESTIONABLE"  # Real speech should be larger
            }
            
            print(f"   ✅ Synthesis successful!")
            print(f"   📁 Saved: {filename}")
            print(f"   📊 Size: {file_size:,} bytes")
            print(f"   ⏱️ Time: {synthesis_time:.2f}s")
            print(f"   🎯 Quality: {result['quality_check']}")
            
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ Synthesis failed: {e}")
            results.append({
                "test": i,
                "success": False,
                "error": str(e)
            })
    
    return results

def analyze_results(results, status):
    """Analyze test results and provide recommendations"""
    
    print("\n📊 ANALYSIS & RECOMMENDATIONS")
    print("=" * 50)
    
    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]
    
    print(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed tests: {len(failed_tests)}")
    
    if successful_tests:
        avg_size = sum(r['file_size'] for r in successful_tests) / len(successful_tests)
        avg_time = sum(r['synthesis_time'] for r in successful_tests) / len(successful_tests)
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Average file size: {avg_size:,.0f} bytes")
        print(f"   Average synthesis time: {avg_time:.2f}s")
        
        # Quality assessment
        if avg_size > 100000:  # Larger files suggest real audio
            print(f"   🎯 Quality: EXCELLENT (Real neural TTS)")
        elif avg_size > 50000:
            print(f"   🎯 Quality: GOOD (Enhanced synthesis)")
        else:
            print(f"   🎯 Quality: POOR (May still be synthetic)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if status and status['real_tts_available']:
        print(f"   ✅ Real TTS is active - you should hear natural speech")
        print(f"   🎧 Listen to the generated files to verify quality")
    else:
        print(f"   ⚠️ Real TTS not available - still using synthetic speech")
        print(f"   📥 Run 'install_real_tts.bat' to install real TTS engines")
    
    if failed_tests:
        print(f"   🔧 Some tests failed - check error messages above")
        print(f"   🔄 Try restarting the TTS service")
    
    # File locations
    if successful_tests:
        print(f"\n📁 Generated test files:")
        for result in successful_tests:
            print(f"   {result['filename']}")

async def main():
    """Main test function"""
    
    print("🎙️ ZONOS TTS ENHANCEMENT VERIFICATION")
    print("=" * 60)
    print()
    
    # Test import and initialization
    tts, status = test_enhanced_zonos()
    
    if not tts:
        print("\n❌ Cannot proceed with synthesis tests")
        return
    
    # Test synthesis
    results = await test_synthesis(tts)
    
    # Analyze results
    analyze_results(results, status)
    
    print("\n🎉 TESTING COMPLETE!")
    print("=" * 60)
    print("\n🎧 Listen to the generated audio files to verify the quality improvement!")

if __name__ == "__main__":
    asyncio.run(main())
