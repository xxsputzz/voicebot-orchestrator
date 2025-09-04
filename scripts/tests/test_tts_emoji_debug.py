#!/usr/bin/env python3
"""
Test to debug emoji contamination in TTS service
"""
import requests
import json
import sys
import io

def test_tts_with_debug_output():
    """Test TTS with output redirection to catch emoji contamination"""
    print("🔍 DEBUGGING TTS EMOJI CONTAMINATION")
    print("=" * 50)
    
    # Clean ASCII text
    clean_text = "I understand you're asking about: steal ball bearings. How can I help you with this?"
    
    print(f"📝 Input text: {repr(clean_text)}")
    print(f"✅ Input is clean ASCII: {all(ord(char) <= 127 for char in clean_text)}")
    
    # Test 1: Direct TTS request
    print("\n🔄 Testing direct TTS request...")
    
    # Capture stdout/stderr to see if there's contamination
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    captured_output = io.StringIO()
    
    try:
        # Temporarily redirect output
        sys.stdout = captured_output
        sys.stderr = captured_output
        
        response = requests.post(
            'http://localhost:8012/synthesize',
            json={
                'text': clean_text,
                'voice': 'af_bella',
                'return_audio': True
            },
            timeout=30
        )
        
        # Restore output
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Check captured output
        captured = captured_output.getvalue()
        print(f"📋 Captured output: {repr(captured)}")
        
        if '🎤' in captured:
            print("⚠️ Found microphone emoji in captured output!")
            for i, char in enumerate(captured):
                if ord(char) > 127:
                    print(f"   Position {i}: {repr(char)} (U+{ord(char):04X})")
        
        print(f"🌐 Response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ TTS request succeeded!")
            result = response.json()
            print(f"📊 Metadata: {result.get('metadata', {})}")
        else:
            print(f"❌ TTS request failed: {response.status_code}")
            print(f"💥 Error: {response.text}")
            
    except Exception as e:
        # Restore output
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"❌ Test failed: {e}")
    finally:
        captured_output.close()

if __name__ == "__main__":
    test_tts_with_debug_output()
