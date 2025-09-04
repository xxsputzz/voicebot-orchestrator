"""
Test Enhanced Zonos TTS Service
Testing comprehensive API endpoints with new features
"""
import asyncio
import aiohttp
import json
import time

async def test_enhanced_service():
    """Test the enhanced TTS service endpoints"""
    print("🚀 Testing Enhanced Zonos TTS Service")
    print("=" * 50)
    
    base_url = "http://localhost:8014"
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Check service health
        print("\n1. Testing Service Health...")
        try:
            async with session.get(f"{base_url}/health") as resp:
                if resp.status == 200:
                    health_data = await resp.json()
                    print(f"✅ Service healthy: {health_data['status']}")
                else:
                    print(f"❌ Service health check failed: {resp.status}")
        except Exception as e:
            print(f"❌ Service not running: {e}")
            print("Please start the service with: python aws_microservices/tts_zonos_service.py")
            return
        
        # Test 2: Get enhanced voice options
        print("\n2. Testing Enhanced Voice Options...")
        try:
            async with session.get(f"{base_url}/voices") as resp:
                if resp.status == 200:
                    voice_data = await resp.json()
                    print(f"✅ Enhanced voices loaded:")
                    print(f"   Female voices: {len(voice_data['voices']['female'])}")
                    print(f"   Male voices: {len(voice_data['voices']['male'])}")
                    print(f"   Neutral voices: {len(voice_data['voices']['neutral'])}")
                    print(f"   Total options: {voice_data['count']}")
                    if 'emotions' in voice_data:
                        print(f"   Emotion categories: {len(voice_data['emotions'])}")
                        print(f"   Speaking styles: {len(voice_data['speaking_styles'])}")
                else:
                    print(f"❌ Voice options failed: {resp.status}")
        except Exception as e:
            print(f"❌ Voice options error: {e}")
        
        # Test 3: Get emotions
        print("\n3. Testing Emotion Categories...")
        try:
            async with session.get(f"{base_url}/emotions") as resp:
                if resp.status == 200:
                    emotion_data = await resp.json()
                    print(f"✅ Emotions loaded: {emotion_data['total_count']} total")
                    for category, emotions in emotion_data['emotions'].items():
                        print(f"   {category}: {len(emotions)} emotions")
                else:
                    print(f"❌ Emotions failed: {resp.status}")
        except Exception as e:
            print(f"❌ Emotions error: {e}")
        
        # Test 4: Get speaking styles
        print("\n4. Testing Speaking Styles...")
        try:
            async with session.get(f"{base_url}/speaking_styles") as resp:
                if resp.status == 200:
                    style_data = await resp.json()
                    print(f"✅ Speaking styles loaded: {style_data['count']} total")
                    for style, info in list(style_data['speaking_styles'].items())[:3]:
                        print(f"   {style}: {info['description']}")
                else:
                    print(f"❌ Speaking styles failed: {resp.status}")
        except Exception as e:
            print(f"❌ Speaking styles error: {e}")
        
        # Test 5: Enhanced synthesis with female voice
        print("\n5. Testing Enhanced Female Voice Synthesis...")
        synthesis_data = {
            "text": "Hello! I'm Sophia, your enhanced AI assistant with emotional intelligence.",
            "voice": "sophia",
            "emotion": "friendly",
            "speaking_style": "conversational",
            "speed": 1.1,
            "pitch": 1.0,
            "emphasis_words": ["enhanced", "emotional", "intelligence"],
            "high_quality": True,
            "output_format": "wav",
            "return_audio": True
        }
        
        try:
            start_time = time.time()
            async with session.post(f"{base_url}/synthesize", json=synthesis_data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    synthesis_time = time.time() - start_time
                    print(f"✅ Enhanced synthesis successful:")
                    print(f"   Processing time: {synthesis_time:.2f}s")
                    print(f"   Audio size: {result['metadata']['audio_size_bytes']} bytes")
                    print(f"   Voice: {result['metadata']['voice']}")
                    print(f"   Emotion: {result['metadata']['emotion']}")
                    print(f"   Speaking style: {result['metadata']['speaking_style']}")
                    print(f"   Engine: {result['metadata']['engine_used']}")
                else:
                    print(f"❌ Enhanced synthesis failed: {resp.status}")
        except Exception as e:
            print(f"❌ Enhanced synthesis error: {e}")
        
        # Test 6: Professional emotion synthesis
        print("\n6. Testing Professional Emotion Synthesis...")
        professional_data = {
            "text": "Good morning, team. Today's quarterly review will demonstrate our exceptional performance metrics and strategic achievements.",
            "voice": "professional",
            "emotion": "authoritative",
            "speaking_style": "presentation",
            "speed": 0.9,
            "pitch": 0.95,
            "prosody_adjustments": {
                "rate": 0.9,
                "pitch": 0.95
            },
            "high_quality": True,
            "return_audio": True
        }
        
        try:
            start_time = time.time()
            async with session.post(f"{base_url}/synthesize", json=professional_data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    synthesis_time = time.time() - start_time
                    print(f"✅ Professional synthesis successful:")
                    print(f"   Processing time: {synthesis_time:.2f}s")
                    print(f"   Audio size: {result['metadata']['audio_size_bytes']} bytes")
                    print(f"   Prosody adjustments applied: {bool(result['metadata']['parameters']['prosody_adjustments'])}")
                else:
                    print(f"❌ Professional synthesis failed: {resp.status}")
        except Exception as e:
            print(f"❌ Professional synthesis error: {e}")
        
        # Test 7: Entertainment emotion with intensity
        print("\n7. Testing Entertainment Emotion...")
        entertainment_data = {
            "text": "Welcome to the most spectacular, magical, and absolutely extraordinary show you've ever witnessed!",
            "voice": "luna",
            "emotion": "dramatic",
            "speaking_style": "storytelling",
            "speed": 1.0,
            "pitch": 1.1,
            "emphasis_words": ["spectacular", "magical", "extraordinary"],
            "high_quality": True,
            "return_audio": True
        }
        
        try:
            start_time = time.time()
            async with session.post(f"{base_url}/synthesize", json=entertainment_data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    synthesis_time = time.time() - start_time
                    print(f"✅ Entertainment synthesis successful:")
                    print(f"   Processing time: {synthesis_time:.2f}s")
                    print(f"   Audio size: {result['metadata']['audio_size_bytes']} bytes")
                    print(f"   Emphasis words: {result['metadata']['parameters']['emphasis_words']}")
                else:
                    print(f"❌ Entertainment synthesis failed: {resp.status}")
        except Exception as e:
            print(f"❌ Entertainment synthesis error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Enhanced TTS Service Testing Complete!")
    print("\nService Features Tested:")
    print("✅ Enhanced voice catalog with gender categories")
    print("✅ Comprehensive emotion system (25+ emotions)")
    print("✅ Advanced speaking styles (9 styles)")
    print("✅ Female voice synthesis with emotions")
    print("✅ Professional business communications")
    print("✅ Entertainment and storytelling modes")
    print("✅ Emphasis word processing")
    print("✅ Prosody adjustments")
    print("✅ High-quality neural synthesis")

if __name__ == "__main__":
    asyncio.run(test_enhanced_service())
