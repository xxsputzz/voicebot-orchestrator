#!/usr/bin/env python3
"""
Direct Dia Long Audio Test
=========================

Test Dia model directly with high token count to see if it can generate longer audio.
Bypass all managers and services to isolate the issue.
"""

import sys
import os
import time
import soundfile as sf
import torch

# Add dia path
dia_path = os.path.join('tests', 'dia')
sys.path.insert(0, dia_path)

def test_direct_dia_long():
    """Test Dia model directly with long text and high token count"""
    
    print("🧪 Direct Dia Long Audio Test")
    print("=" * 50)
    
    try:
        from dia.model import Dia
        
        print("📥 Loading Dia model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
        print(f"✅ Model loaded on {device}")
        
        # Test with your actual text
        test_text = """[S1] Hello, hello! This is Alex calling with Finally Payoff Debt, your cheerful AI pre-qualification specialist. I'm so glad you picked up today. I promise this will be quick, friendly, and helpful.

[S1] Here's the news: We've rolled out new income-based loan options to help people escape high-interest credit cards and personal loans. You know the ones—you pay and pay, but the balance never drops.

[S1] Now, listen... if you've got steady income, we can help you qualify for a personal loan between ten thousand and one hundred thousand dollars. That means instead of juggling multiple bills, you could roll them into one easy payment.

[S1] Picture this: One payment, one plan, one path forward... But really, one lower-interest payment feels so much lighter. That alone takes away so much stress.

[S1] Credit card rates are brutal—twenty, even thirty percent. With us, rates can start around six to seven percent.

[S1] Now that's savings! Think about the money you'd free up every month. FREEDOM from sky-high payments!

[S1] Most clients qualify for a payment between $250 and $375 a month. If you're paying multiple cards now, combining them often means you'll pay less."""
        
        print(f"📝 Text length: {len(test_text)} characters")
        print(f"📝 Text preview: {test_text[:100]}...")
        
        # Test with very high token count
        max_tokens_tests = [4096, 8192, 12288, 16384]
        
        for max_tokens in max_tokens_tests:
            print(f"\n🔄 Testing with {max_tokens} max tokens...")
            
            try:
                # Set seeds for consistency
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42)
                
                start_time = time.time()
                
                audio = model.generate(
                    text=test_text,
                    max_tokens=max_tokens,
                    cfg_scale=3.0,
                    temperature=1.2,
                    top_p=0.95,
                    verbose=True
                )
                
                generation_time = time.time() - start_time
                
                # Check audio properties
                if hasattr(audio, 'shape'):
                    audio_length = len(audio)
                    duration = audio_length / 44100  # Assuming 44.1kHz
                    
                    print(f"✅ Generated in {generation_time:.1f}s")
                    print(f"📊 Audio samples: {audio_length:,}")
                    print(f"🕐 Audio duration: {duration:.1f}s")
                    print(f"📈 Realtime factor: {generation_time/duration:.1f}x")
                    
                    # Save audio
                    filename = f"direct_dia_test_{max_tokens}tokens_{duration:.1f}s.wav"
                    sf.write(filename, audio, 44100)
                    
                    # Check file size
                    file_size = os.path.getsize(filename)
                    print(f"💾 Saved: {filename} ({file_size:,} bytes, {file_size/1024:.1f} KB)")
                    
                    # If we got a duration > 10 seconds, we found success!
                    if duration > 10:
                        print(f"🎉 SUCCESS! Generated {duration:.1f}s of audio with {max_tokens} tokens!")
                        return True
                else:
                    print(f"❌ No audio generated with {max_tokens} tokens")
                    
            except Exception as e:
                print(f"❌ Failed with {max_tokens} tokens: {e}")
                
        print("\n📊 Summary:")
        print("All token counts tested. Check generated files for actual duration.")
        return False
        
    except ImportError as e:
        print(f"❌ Cannot import Dia: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_direct_dia_long()
    print(f"\nResult: {'✅ Found longer generation' if success else '❌ Still limited to short audio'}")
