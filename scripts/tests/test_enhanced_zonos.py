"""
Test Enhanced Zonos TTS Implementation
Testing comprehensive voice/emotion/style system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from voicebot_orchestrator.zonos_tts import ZonosTTS

async def test_enhanced_zonos():
    """Test the enhanced Zonos TTS system"""
    print("🎤 Testing Enhanced Zonos TTS System")
    print("=" * 50)
    
    # Initialize TTS
    tts = ZonosTTS()
    
    # Test 1: Get available options
    print("\n1. Testing Available Options...")
    options = tts.get_available_options()
    
    print(f"Female voices: {list(options['voices']['female'].keys())}")
    print(f"Male voices: {list(options['voices']['male'].keys())}")
    print(f"Neutral voices: {list(options['voices']['neutral'].keys())}")
    print(f"Total voices: {sum(len(voices) for voices in options['voices'].values())}")
    
    print(f"\nEmotion categories: {list(options['emotions'].keys())}")
    print(f"Speaking styles: {options['speaking_styles']}")
    print(f"Output formats: {options['output_formats']}")
    
    # Test 2: Female voice synthesis
    print("\n2. Testing Female Voice Synthesis...")
    test_text = "Hello! I'm Sophia, your AI assistant. How can I help you today?"
    
    try:
        audio_bytes = await tts.synthesize_speech(
            text=test_text,
            voice="sophia",
            emotion="friendly",
            speaking_style="conversational",
            speed=1.1,
            pitch=1.0
        )
        print(f"✅ Female voice synthesis successful: {len(audio_bytes)} bytes")
        
        # Save test file
        with open("test_sophia_voice.wav", "wb") as f:
            f.write(audio_bytes)
        print("   Saved as: test_sophia_voice.wav")
        
    except Exception as e:
        print(f"❌ Female voice synthesis failed: {e}")
    
    # Test 3: Professional emotion
    print("\n3. Testing Professional Emotion...")
    business_text = "Welcome to our quarterly review meeting. Today we'll discuss our performance metrics."
    
    try:
        audio_bytes = await tts.synthesize_speech(
            text=business_text,
            voice="professional",
            emotion="authoritative",
            speaking_style="presentation",
            speed=0.9,
            pitch=0.95
        )
        print(f"✅ Professional emotion synthesis successful: {len(audio_bytes)} bytes")
        
        # Save test file
        with open("test_professional_voice.wav", "wb") as f:
            f.write(audio_bytes)
        print("   Saved as: test_professional_voice.wav")
        
    except Exception as e:
        print(f"❌ Professional emotion synthesis failed: {e}")
    
    # Test 4: Entertainment emotion with emphasis
    print("\n4. Testing Entertainment Emotion with Emphasis...")
    story_text = "Once upon a time, in a magical kingdom, there lived a brave princess who could speak to dragons!"
    
    try:
        audio_bytes = await tts.synthesize_speech(
            text=story_text,
            voice="luna",
            emotion="whimsical",
            speaking_style="storytelling",
            emphasis_words=["magical", "brave", "dragons"],
            speed=1.0,
            pitch=1.1
        )
        print(f"✅ Entertainment emotion synthesis successful: {len(audio_bytes)} bytes")
        
        # Save test file
        with open("test_entertainment_voice.wav", "wb") as f:
            f.write(audio_bytes)
        print("   Saved as: test_entertainment_voice.wav")
        
    except Exception as e:
        print(f"❌ Entertainment emotion synthesis failed: {e}")
    
    # Test 5: Intensity variants
    print("\n5. Testing Emotion Intensity Variants...")
    test_variants = [
        ("very_happy", "I'm absolutely thrilled to announce our success!"),
        ("slightly_sad", "Take a deep breath and relax your shoulders."),
        ("mildly_happy", "This is some good news to share!")
    ]
    
    for emotion, text in test_variants:
        try:
            audio_bytes = await tts.synthesize_speech(
                text=text,
                voice="aria",
                emotion=emotion,
                speaking_style="normal",
                speed=1.0 if "sad" in emotion else 1.2,
                pitch=1.0
            )
            print(f"✅ {emotion}: {len(audio_bytes)} bytes")
            
            # Save test file
            filename = f"test_{emotion}_voice.wav"
            with open(filename, "wb") as f:
                f.write(audio_bytes)
            print(f"   Saved as: {filename}")
            
        except Exception as e:
            print(f"❌ {emotion} synthesis failed: {e}")
    
    # Test 6: Prosody adjustments
    print("\n6. Testing Prosody Adjustments...")
    test_text = "The quick brown fox jumps over the lazy dog."
    
    try:
        audio_bytes = await tts.synthesize_speech(
            text=test_text,
            voice="marcus",
            emotion="neutral",
            speaking_style="normal",
            prosody_adjustments={
                "rate": 1.2,
                "pitch": 0.9,
                "volume": 1.1
            },
            speed=1.0,
            pitch=1.0
        )
        print(f"✅ Prosody adjustments successful: {len(audio_bytes)} bytes")
        
        # Save test file
        with open("test_prosody_voice.wav", "wb") as f:
            f.write(audio_bytes)
        print("   Saved as: test_prosody_voice.wav")
        
    except Exception as e:
        print(f"❌ Prosody adjustments failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Enhanced Zonos TTS Testing Complete!")
    print("Generated test files:")
    print("  - test_sophia_voice.wav (Female voice)")
    print("  - test_professional_voice.wav (Professional emotion)")
    print("  - test_entertainment_voice.wav (Whimsical storytelling)")
    print("  - test_very_happy_voice.wav (Emotion intensity)")
    print("  - test_slightly_sad_voice.wav (Soft emotion)")
    print("  - test_mildly_happy_voice.wav (Mild intensity)")
    print("  - test_prosody_voice.wav (Prosody adjustments)")

if __name__ == "__main__":
    asyncio.run(test_enhanced_zonos())
