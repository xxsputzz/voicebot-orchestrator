#!/usr/bin/env python3
"""
Simple Zonos TTS Test Script - No Interactive Input
Tests the Zonos TTS service with automatic inputs
"""
import asyncio
import requests
import json

async def test_zonos_tts_simple():
    """Test Zonos TTS service with predefined inputs"""
    print("🧠 Direct Zonos TTS Test (Neural Speech Synthesis)")
    print("-" * 60)
    
    zonos_url = "http://localhost:8014"
    print(f"🔍 Checking Zonos TTS service at {zonos_url}...")
    
    try:
        # Check health
        response = requests.get(f"{zonos_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Zonos TTS service not healthy: {response.status_code}")
            return False
        
        health_data = response.json()
        print(f"✅ Service status: {health_data.get('status', 'unknown')}")
        print(f"   Version: {health_data.get('version', 'unknown')}")
        
        # Get voices with details
        voices_response = requests.get(f"{zonos_url}/voices", timeout=5)
        detailed_response = requests.get(f"{zonos_url}/voices_detailed", timeout=5)
        
        if voices_response.status_code == 200:
            voices = voices_response.json()
            print(f"🎭 Available voices: {', '.join(voices[:5])}... ({len(voices)} total)")
            
            # Show detailed voice info if available
            if detailed_response.status_code == 200:
                detailed_data = detailed_response.json()
                voice_details = detailed_data.get('voices', {})
                
                print(f"\n🎤 Voice Details (first 5):")
                count = 0
                for category, voice_dict in voice_details.items():
                    for voice_name, info in voice_dict.items():
                        if voice_name in voices and count < 5:
                            gender = info.get('gender', '').title()
                            style = info.get('style', '').replace('_', ' ').title()
                            accent = info.get('accent', '').title()
                            print(f"   {voice_name:<12} | {gender:<7} | {style:<15} | {accent}")
                            count += 1
                        if count >= 5:
                            break
                    if count >= 5:
                        break
        else:
            voices = ['default']
            print("⚠️ Using default voice")
        
        # Get models
        models_response = requests.get(f"{zonos_url}/models", timeout=5)
        if models_response.status_code == 200:
            models = models_response.json()
            print(f"🤖 Available models: {', '.join(models)}")
        else:
            models = ['zonos-v1']
            print("⚠️ Using default model")
        
        # Test synthesis with predefined text
        test_text = "Hello! This is a test of the Zonos TTS neural speech synthesis."
        selected_voice = voices[0] if voices else "default"
        selected_model = models[0] if models else "zonos-v1"
        
        print(f"\n📝 Test Text: {test_text}")
        print(f"🎭 Selected voice: {selected_voice}")
        print(f"🤖 Selected model: {selected_model}")
        
        # Make synthesis request
        synthesis_data = {
            "text": test_text,
            "voice": selected_voice,
            "model": selected_model,
            "emotion": "happy",
            "speed": 1.0,
            "return_audio": True
        }
        
        print("\n🎵 Starting synthesis...")
        synthesis_response = requests.post(
            f"{zonos_url}/synthesize",
            json=synthesis_data,
            timeout=30
        )
        
        if synthesis_response.status_code == 200:
            result = synthesis_response.json()
            metadata = result.get('metadata', {})
            print(f"✅ Synthesis successful!")
            print(f"   Processing time: {metadata.get('processing_time_seconds', 0):.3f}s")
            print(f"   Audio size: {metadata.get('audio_size_bytes', 0):,} bytes")
            print(f"   Estimated duration: {metadata.get('estimated_duration_seconds', 0):.1f}s")
            return True
        else:
            print(f"❌ Synthesis failed: {synthesis_response.status_code}")
            print(f"   Error: {synthesis_response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Cannot connect to Zonos TTS service: {e}")
        print(f"   Make sure the service is running on port 8014")
        return False
    except Exception as e:
        print(f"❌ Zonos TTS test failed: {e}")
        return False

async def main():
    success = await test_zonos_tts_simple()
    result = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"\n📊 Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
