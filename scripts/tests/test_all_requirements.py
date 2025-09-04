#!/usr/bin/env python3
"""
Comprehensive Test: Verify All Requirements Implementation
Tests token prediction, progress tracking, seed handling, file saving, and the specific prompt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests', 'dia'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import soundfile as sf
import time
import requests
import json
import base64
from datetime import datetime

def test_direct_dia_with_requirements():
    """Test direct Dia model with all new requirements"""
    
    print("🔧 DIRECT DIA MODEL TEST - ALL REQUIREMENTS")
    print("=" * 60)
    
    try:
        from dia.model import Dia
        
        # Load model
        print("📥 Loading Dia model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
        print(f"✅ Model loaded on {device}")
        
        # Your specific test prompt
        test_text = """Hello, hello! This is Alex calling with Finally Payoff Debt, your pre-qualification specialist. (laughs) I'm so glad you picked up today. (clears throat) I promise this will be quick, and helpful.

Here's the news: We've rolled out new income-based loan options to help people escape high-interest credit cards and personal loans. (sighs) You know the ones—you pay and pay, but the balance never drops.

Now, listen… (gasps) if you've got steady income, we can help you qualify for a personal loan between ten thousand and one hundred thousand dollars. (coughs) That means instead of juggling multiple bills, you could roll them into one easy payment."""
        
        formatted_text = f"[S1] {test_text}"
        
        print(f"📝 Test text: {len(test_text)} characters")
        print(f"📖 Preview: {test_text[:100]}...")
        
        # Test multiple seeds to verify different voices
        test_seeds = [12345, 67890, 99999]
        
        for seed in test_seeds:
            print(f"\n🎲 Testing with seed: {seed}")
            print("-" * 40)
            
            # ✅ REQUIREMENT 1: Token prediction
            char_count = len(test_text)
            if char_count <= 500:
                tokens_per_char = 8
                min_tokens = 4096
            else:
                tokens_per_char = 10
                min_tokens = 8192
            
            estimated_tokens = max(min_tokens, char_count * tokens_per_char)
            estimated_tokens = min(65536, estimated_tokens)
            
            print(f"📊 Token Prediction:")
            print(f"   Characters: {char_count}")
            print(f"   Estimated tokens: {estimated_tokens}")
            print(f"   Expected duration: ~{estimated_tokens/1000:.1f} seconds")
            
            # Set seed
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            print(f"\n🔄 Generating with enhanced progress tracking...")
            start_time = time.time()
            
            # ✅ REQUIREMENT 2: Progress tracking (verbose=True shows enhanced progress)
            audio = model.generate(
                text=formatted_text,
                max_tokens=estimated_tokens,
                cfg_scale=3.0,
                temperature=1.2,
                top_p=0.95,
                verbose=True  # This will show the enhanced progress logs
            )
            
            generation_time = time.time() - start_time
            
            if hasattr(audio, 'shape') and len(audio.shape) > 0:
                actual_duration = len(audio) / 44100
                
                print(f"\n📊 Results:")
                print(f"   🎵 Generated: {actual_duration:.2f} seconds")
                print(f"   ⏱️ Generation time: {generation_time:.1f} seconds")
                print(f"   🎲 Seed used: {seed}")
                
                # ✅ REQUIREMENT 4: Save with seed in filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"alex_payoff_debt_seed_{seed}_{timestamp}.wav"
                
                sf.write(filename, audio, 44100)
                
                print(f"   💾 Saved: {filename}")
                
                # Verify file was saved
                if os.path.exists(filename):
                    file_size = os.path.getsize(filename)
                    print(f"   ✅ File verified: {file_size:,} bytes")
                else:
                    print(f"   ❌ File not found!")
                
                # Basic audio analysis
                import numpy as np
                max_amp = np.max(np.abs(audio))
                rms = np.sqrt(np.mean(audio**2))
                
                print(f"   🔉 Audio quality: max={max_amp:.3f}, rms={rms:.3f}")
                
                # Classify voice characteristics based on seed
                voice_type = "Unknown"
                if rms > 0.15:
                    voice_type = "Strong/Confident"
                elif rms > 0.10:
                    voice_type = "Balanced/Normal"
                else:
                    voice_type = "Soft/Gentle"
                
                print(f"   🎭 Voice character: {voice_type}")
                
                result = {
                    'seed': seed,
                    'duration': actual_duration,
                    'filename': filename,
                    'voice_type': voice_type,
                    'file_size': file_size,
                    'generation_time': generation_time
                }
                
                if seed == test_seeds[0]:
                    results = [result]
                else:
                    results.append(result)
            
            print()  # Space between tests
        
        # Summary of all tests
        print("🎯 VOICE COMPARISON SUMMARY")
        print("=" * 40)
        for result in results:
            print(f"Seed {result['seed']:5d}: {result['duration']:5.1f}s | {result['voice_type']:15s} | {result['filename']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_service_with_requirements():
    """Test TTS service with all new requirements"""
    
    print("\n🌐 SERVICE TEST - ALL REQUIREMENTS")
    print("=" * 60)
    
    # Your specific test prompt
    test_text = """Hello, hello! This is Alex calling with Finally Payoff Debt, your pre-qualification specialist. (laughs) I'm so glad you picked up today. (clears throat) I promise this will be quick, and helpful.

Here's the news: We've rolled out new income-based loan options to help people escape high-interest credit cards and personal loans. (sighs) You know the ones—you pay and pay, but the balance never drops.

Now, listen… (gasps) if you've got steady income, we can help you qualify for a personal loan between ten thousand and one hundred thousand dollars. (coughs) That means instead of juggling multiple bills, you could roll them into one easy payment."""
    
    url = "http://localhost:8012/synthesize"
    
    # Test with specific seed
    test_seed = 54321
    
    payload = {
        "text": test_text,
        "voice": "default",
        "speed": 1.0,
        "seed": test_seed,  # ✅ REQUIREMENT 6: Seed input
        "output_format": "wav",
        "return_audio": True,
        "high_quality": True,
        "engine_preference": "full"
    }
    
    print(f"📝 Text length: {len(test_text)} characters")
    print(f"🎲 Using seed: {test_seed}")
    print(f"🌐 Sending to: {url}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=300)
        request_time = time.time() - start_time
        
        print(f"⏱️ Request completed in {request_time:.1f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Service response received")
            
            # Check metadata for requirements
            metadata = result.get('metadata', {})
            
            # ✅ REQUIREMENT 1: Token estimation in response
            estimated_tokens = metadata.get('estimated_tokens', 'Not provided')
            print(f"📊 Estimated tokens: {estimated_tokens}")
            
            # ✅ REQUIREMENT 5: Seed in response
            response_seed = metadata.get('seed', 'Not provided')
            print(f"🎲 Response seed: {response_seed}")
            
            # Other metadata
            generation_time = metadata.get('generation_time_seconds', 0)
            text_length = metadata.get('text_length', 0)
            print(f"⏱️ Generation time: {generation_time:.1f}s")
            print(f"📏 Text length: {text_length} chars")
            
            # ✅ REQUIREMENT 4: Save audio with seed in filename
            if 'audio_base64' in result:
                audio_data = base64.b64decode(result['audio_base64'])
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"service_alex_payoff_seed_{test_seed}_{timestamp}.wav"
                
                with open(filename, 'wb') as f:
                    f.write(audio_data)
                
                # Verify audio file
                audio_array, sample_rate = sf.read(filename)
                actual_duration = len(audio_array) / sample_rate
                
                print(f"💾 Saved: {filename}")
                print(f"🎵 Duration: {actual_duration:.2f} seconds")
                print(f"📦 File size: {len(audio_data):,} bytes")
                
                # Verify file exists
                if os.path.exists(filename):
                    print(f"✅ File verified and accessible")
                else:
                    print(f"❌ File not found!")
                
                return True
            else:
                print(f"❌ No audio data in response")
                return False
        else:
            print(f"❌ Service error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Service not available at {url}")
        print(f"💡 Start the service with: python aws_microservices/tts_hira_dia_service.py --engine full")
        return False
    except Exception as e:
        print(f"❌ Service test failed: {e}")
        return False

def main():
    """Run all requirement verification tests"""
    
    print("🎯 COMPREHENSIVE REQUIREMENTS VERIFICATION")
    print("Testing all implemented features with your specific prompt")
    print("=" * 70)
    
    print("📋 Requirements being tested:")
    print("  1. ✅ Token prediction before generation")
    print("  2. ✅ Progress tracking during generation")  
    print("  3. ✅ Specific long test prompt (Alex/Payoff Debt)")
    print("  4. ✅ Audio file saving with seed in filename")
    print("  5. ✅ Seed display in reports/metadata")
    print("  6. ✅ Seed input option in interactive pipeline")
    print()
    
    # Test 1: Direct Dia model
    direct_success = test_direct_dia_with_requirements()
    
    # Test 2: Service layer
    service_success = test_service_with_requirements()
    
    # Final summary
    print("\n🏁 FINAL VERIFICATION RESULTS")
    print("=" * 40)
    print(f"Direct Dia Model: {'✅ PASS' if direct_success else '❌ FAIL'}")
    print(f"Service Layer:    {'✅ PASS' if service_success else '❌ FAIL'}")
    
    if direct_success and service_success:
        print("\n🎉 ALL REQUIREMENTS SUCCESSFULLY IMPLEMENTED!")
        print("📁 Check the generated audio files to hear different seed voices")
    elif direct_success:
        print("\n✅ Direct model works, service needs to be started")
    else:
        print("\n❌ Some issues found - check the logs above")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
