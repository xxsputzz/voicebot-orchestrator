#!/usr/bin/env python3
"""
Real Whisper STT Test
====================

Test the real Whisper implementation with recorded audio files.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

async def test_real_whisper():
    """Test real Whisper STT implementation"""
    print("🎙️ REAL WHISPER STT TEST")
    print("=" * 50)
    
    try:
        # Import the real Whisper implementation
        from voicebot_orchestrator.real_whisper_stt import WhisperSTT
        
        print("✅ Real Whisper STT imported successfully")
        
        # Initialize Whisper STT
        print("🔄 Initializing Whisper STT...")
        whisper_stt = WhisperSTT(model_name="base", device="cpu")
        
        # Find recorded audio files
        audio_dir = Path("tests/audio_samples/microphone_tests")
        if not audio_dir.exists():
            print(f"❌ Audio directory not found: {audio_dir}")
            return False
        
        audio_files = list(audio_dir.glob("*.wav"))
        if not audio_files:
            print(f"❌ No audio files found in: {audio_dir}")
            return False
        
        print(f"📁 Found {len(audio_files)} audio files:")
        for file in audio_files:
            print(f"   📄 {file.name}")
        
        # Test transcription with each file
        print(f"\n🔄 Testing real Whisper transcription...")
        
        for i, audio_file in enumerate(audio_files[:3], 1):  # Test first 3 files
            print(f"\n📄 File {i}: {audio_file.name}")
            print(f"   Size: {audio_file.stat().st_size:,} bytes")
            
            try:
                # Transcribe using real Whisper
                print("   � Transcribing...")
                transcript = await whisper_stt.transcribe_file(str(audio_file))
                
                print(f"   ✅ Transcript: '{transcript}'")
                
                if transcript.strip():
                    print("   ✅ Non-empty transcription received!")
                else:
                    print("   ⚠️  Empty transcription (might be silence)")
                
            except Exception as e:
                print(f"   ❌ Transcription failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print(f"\n✅ Real Whisper STT test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure openai-whisper is installed: pip install openai-whisper")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_whisper_service():
    """Test the Whisper STT service with real audio"""
    print(f"\n🌐 WHISPER STT SERVICE TEST")
    print("=" * 50)
    
    try:
        import requests
        
        # Check if service is running
        try:
            response = requests.get("http://localhost:8002/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Whisper STT service is running")
                print(f"   Status: {health_data.get('status', 'unknown')}")
                print(f"   Ready: {health_data.get('ready', 'unknown')}")
            else:
                print(f"❌ Whisper STT service health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Whisper STT service not available: {e}")
            print("💡 Start the service with: python aws_microservices/stt_whisper_service.py")
            return False
        
        # Find a test audio file
        audio_dir = Path("tests/audio_samples/microphone_tests")
        audio_files = list(audio_dir.glob("*.wav"))
        
        if not audio_files:
            print(f"❌ No test audio files found")
            return False
        
        test_file = audio_files[0]
        print(f"📄 Using test file: {test_file.name}")
        
        # Send to STT service
        print(f"🔄 Sending audio to STT service...")
        
        with open(test_file, 'rb') as f:
            files = {'audio': f}
            response = requests.post("http://localhost:8002/transcribe", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            transcript = result.get("text", "").strip()
            confidence = result.get("confidence", 0.0)
            processing_time = result.get("processing_time_seconds", 0.0)
            
            print(f"✅ STT Service Response:")
            print(f"   Transcript: '{transcript}'")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Processing Time: {processing_time:.3f}s")
            
            if transcript:
                print(f"✅ Real transcription received!")
                return True
            else:
                print(f"⚠️  Empty transcript - might be using mock implementation")
                return False
        else:
            print(f"❌ STT service request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Service test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 COMPREHENSIVE WHISPER STT TESTING")
    print("=" * 60)
    
    # Test 1: Real Whisper implementation
    success1 = await test_real_whisper()
    
    # Test 2: Whisper STT service
    success2 = await test_whisper_service()
    
    print(f"\n🏁 TEST SUMMARY")
    print("=" * 60)
    print(f"Real Whisper Implementation: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Whisper STT Service:        {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("Your microphone recordings should now transcribe properly!")
    else:
        print(f"\n⚠️  SOME TESTS FAILED")
        print("The STT issue needs further investigation.")

if __name__ == "__main__":
    asyncio.run(main())
