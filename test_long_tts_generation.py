#!/usr/bin/env python3
"""
Test Long TTS Generation
========================

Test the updated TTS system with long text to verify full-length generation.
"""

import asyncio
import requests
import base64
import time

async def test_long_tts():
    """Test TTS with long text"""
    
    # Your long text
    long_text = """Hello, hello! This is Alex calling with Finally Payoff Debt, your cheerful AI pre-qualification specialist. I'm so glad you picked up today. I promise this will be quick, friendly, and helpful.

Here's the news: We've rolled out new income-based loan options to help people escape high-interest credit cards and personal loans. You know the ones—you pay and pay, but the balance never drops.

Now, listen... if you've got steady income, we can help you qualify for a personal loan between ten thousand and one hundred thousand dollars. That means instead of juggling multiple bills, you could roll them into one easy payment.

Picture this: One payment, one plan, one path forward... But really, one lower-interest payment feels so much lighter. That alone takes away so much stress.

Credit card rates are brutal—twenty, even thirty percent. With us, rates can start around six to seven percent.

Now that's savings! Think about the money you'd free up every month. FREEDOM from sky-high payments!

Most clients qualify for a payment between $250 and $375 a month. If you're paying multiple cards now, combining them often means you'll pay less.

Plus, funds usually hit your account in one to three business days. Excuse me—too much sparkling water. But hey, better out than in, right?

And there are no prepayment penalties. Pay it off early if you'd like. Bless me. Even my allergies are excited.

People ask, "How'd you get my info?" Simple—we work with trusted partners and lenders to connect with folks searching for solutions. Nothing shady here. Just straightforward help.

So let's recap:

Steady income? Check.

Enough debt to make consolidation worthwhile? Check.

Comfortable with $250–$375 a month? Perfect.

That means you're ready. I'm going to connect you with a live loan rep who'll walk you through details and show you your exact rate.

You may hear a quick beep during transfer—that's normal.

Before I go, let me say: carrying debt is exhausting. But this step today means savings, relief, and peace of mind.

You've been awesome. So hang tight—your rep is ready.

Let's do this! Deep breath… and here we go."""

    print("🧪 Testing Long TTS Generation")
    print("=" * 50)
    print(f"📝 Text length: {len(long_text)} characters")
    print(f"📝 Word count: {len(long_text.split())} words")
    
    # Prepare TTS request
    tts_data = {
        "text": long_text,
        "high_quality": True,
        "speed": 1.0,
        "return_audio": True
    }
    
    print(f"\n🔄 Sending to TTS service...")
    print(f"   📊 Text length: {len(long_text)} characters")
    print(f"   🎯 High quality: True")
    print(f"   ⚡ Speed: 1.0x")
    print(f"   ⏳ No timeout limit")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8012/synthesize", 
            json=tts_data, 
            timeout=None  # No timeout
        )
        
        generation_time = time.time() - start_time
        print(f"⏱️ Generation completed in {generation_time:.1f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            audio_data = result.get("audio_base64")
            metadata = result.get("metadata", {})
            
            if audio_data:
                # Save audio file
                timestamp = int(time.time())
                filename = f"long_tts_test_{timestamp}.wav"
                
                audio_bytes = base64.b64decode(audio_data)
                with open(filename, "wb") as f:
                    f.write(audio_bytes)
                
                file_size = len(audio_bytes)
                estimated_duration = metadata.get("estimated_duration_seconds", "unknown")
                
                print(f"✅ Audio saved: {filename}")
                print(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                print(f"🕐 Estimated duration: {estimated_duration} seconds")
                print(f"📈 Realtime factor: {generation_time/float(estimated_duration) if estimated_duration != 'unknown' else 'unknown'}")
                
                # Check if this is significantly longer than 5 seconds
                if isinstance(estimated_duration, (int, float)) and estimated_duration > 15:
                    print(f"🎉 SUCCESS! Generated {estimated_duration}s of audio (much longer than 5s)")
                    return True
                else:
                    print(f"⚠️ Still generating short audio: {estimated_duration}s")
                    return False
            else:
                print("❌ No audio data received")
                return False
        else:
            print(f"❌ TTS request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_long_tts())
    print(f"\nResult: {'✅ Long audio generation working!' if success else '❌ Still limited to short audio'}")
