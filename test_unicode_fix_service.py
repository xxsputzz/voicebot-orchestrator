#!/usr/bin/env python3
"""
Test Unicode fix directly with the TTS service
"""
import requests
import json
import time

def test_unicode_synthesis():
    """Test TTS synthesis with emoji text that previously caused charmap errors"""
    
    # Test text with the problematic microphone emoji
    test_texts = [
        "I understand you're asking about: townhouse. Let me help you with that! 🎤",
        "Here's your response with music 🎵 and sound 🔊",
        "Banking information 🏦 with charts 📊 and trends 📈",
        "Simple text without emojis",
        "Mixed content: phone 📱 laptop 💻 and network 🌐",
        "Warning ⚠️ and checkmark ✅ and X ❌"
    ]
    
    # TTS service endpoint
    tts_url = "http://localhost:8012/synthesize"
    
    print("🧪 Testing Unicode Sanitization Fix")
    print("=" * 50)
    
    # Check if service is ready
    try:
        health_response = requests.get("http://localhost:8012/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ TTS service is running")
        else:
            print("❌ TTS service not ready")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ TTS service not accessible: {e}")
        return
    
    # Test each text
    for i, text in enumerate(test_texts, 1):
        print(f"\n🧪 Test {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Prepare request
        request_data = {
            "text": text,
            "return_audio": False,  # Don't return large audio data for testing
            "engine_preference": "auto"
        }
        
        try:
            start_time = time.time()
            
            # Send synthesis request
            response = requests.post(
                tts_url,
                json=request_data,
                timeout=300  # 5 minutes timeout for TTS generation
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                metadata = result.get("metadata", {})
                print(f"   ✅ SUCCESS - Generated in {elapsed_time:.2f}s")
                print(f"   📊 Engine: {metadata.get('engine_used', 'unknown')}")
                print(f"   📏 Text length: {metadata.get('text_length', 'unknown')}")
                print(f"   🎵 Audio size: {metadata.get('audio_size_bytes', 'unknown')} bytes")
                
                # Check if text was sanitized
                original_length = len(text)
                processed_length = metadata.get('text_length', original_length)
                if processed_length != original_length:
                    print(f"   🔧 Text was sanitized: {original_length} → {processed_length} chars")
                
            else:
                print(f"   ❌ FAILED - Status {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   💥 Error: {error_detail.get('detail', 'Unknown error')}")
                except:
                    print(f"   💥 Error: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ TIMEOUT after {elapsed_time:.2f}s")
        except requests.exceptions.RequestException as e:
            print(f"   🔥 REQUEST ERROR: {e}")
        except Exception as e:
            print(f"   💥 UNEXPECTED ERROR: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Unicode Fix Test Complete")

if __name__ == "__main__":
    test_unicode_synthesis()
