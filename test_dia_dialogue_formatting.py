#!/usr/bin/env python3
"""
Test the hypothesis that Dia needs proper dialogue formatting for long audio
"""

import requests
import json
import time
import numpy as np
import soundfile as sf
import base64

def format_text_for_dia_dialogue(text: str, max_segment_length: int = 100) -> str:
    """
    Format text with multiple [S1] speaker tags for Dia model
    This mimics natural dialogue patterns that Dia expects
    """
    if not text:
        return ""
    
    # If already has speaker tags, return as-is
    if "[S1]" in text:
        return text
    
    # Split into sentences
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in '.!?' and len(current_sentence.strip()) > 10:
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    # Add any remaining text
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    # Group sentences into segments with speaker tags
    formatted_segments = []
    current_segment = ""
    
    for sentence in sentences:
        if len(current_segment) + len(sentence) > max_segment_length and current_segment:
            formatted_segments.append(f"[S1] {current_segment.strip()}")
            current_segment = sentence
        else:
            current_segment += " " + sentence if current_segment else sentence
    
    # Add final segment
    if current_segment.strip():
        formatted_segments.append(f"[S1] {current_segment.strip()}")
    
    return "\n\n".join(formatted_segments)

def test_dialogue_formatting():
    """Test both single [S1] and multi-segment formatting"""
    
    base_text = """The ancient library held secrets beyond imagination. Dusty tomes lined endless shelves, each containing knowledge from civilizations long forgotten. As Sarah walked through the narrow aisles, her footsteps echoed in the vast silence. The smell of old parchment and leather bindings filled the air. Somewhere in these depths lay the answer she sought - a cure for the mysterious plague that had befallen her village. Time was running out, and the keeper of this knowledge was said to be both wise and dangerous."""
    
    # Test 1: Current service formatting (single [S1])
    print("🧪 TEST 1: Single [S1] formatting (current service)")
    single_formatted = f"[S1] {base_text}"
    print(f"📝 Formatted text: {single_formatted[:150]}...")
    
    result1 = test_tts_service(single_formatted, "single_s1")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Multi-segment dialogue formatting
    print("🧪 TEST 2: Multi-segment dialogue formatting")
    multi_formatted = format_text_for_dia_dialogue(base_text, max_segment_length=80)
    print(f"📝 Formatted text: {multi_formatted[:200]}...")
    print(f"📊 Number of [S1] tags: {multi_formatted.count('[S1]')}")
    
    result2 = test_tts_service(multi_formatted, "multi_s1")
    
    # Compare results
    print("\n" + "="*50)
    print("📊 COMPARISON RESULTS")
    print("="*50)
    
    if result1 and result2:
        print(f"Single [S1] - Duration: {result1['actual_duration']:.2f}s, File: {result1['filename']}")
        print(f"Multi [S1]  - Duration: {result2['actual_duration']:.2f}s, File: {result2['filename']}")
        
        improvement = result2['actual_duration'] - result1['actual_duration']
        print(f"📈 Improvement: {improvement:.2f} seconds ({improvement/result1['actual_duration']*100:.1f}%)")
    
    return result1, result2

def test_tts_service(text: str, test_name: str) -> dict:
    """Test TTS service with given text"""
    
    url = "http://localhost:8012/synthesize"
    
    payload = {
        "text": text,
        "voice": "default",
        "speed": 1.0,
        "output_format": "wav",
        "return_audio": True,
        "high_quality": True,
        "engine_preference": "full"
    }
    
    print(f"🌐 Testing with {test_name} formatting...")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        duration = time.time() - start_time
        
        print(f"⏱️ Request completed in {duration:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            
            if 'audio_base64' in result:
                # Decode and save audio
                audio_data = base64.b64decode(result['audio_base64'])
                filename = f"test_{test_name}_audio_{int(time.time())}.wav"
                
                with open(filename, 'wb') as f:
                    f.write(audio_data)
                
                # Get actual duration
                audio_array, sample_rate = sf.read(filename)
                actual_duration = len(audio_array) / sample_rate
                
                print(f"🎵 Actual duration: {actual_duration:.2f} seconds")
                print(f"💾 Saved as: {filename}")
                
                # Get metadata
                metadata = result.get('metadata', {})
                estimated_duration = metadata.get('estimated_duration_seconds', 0)
                print(f"📋 Service estimated: {estimated_duration:.2f} seconds")
                print(f"📋 Generation time: {metadata.get('generation_time_seconds', 0):.2f} seconds")
                
                return {
                    'actual_duration': actual_duration,
                    'estimated_duration': estimated_duration,
                    'filename': filename,
                    'metadata': metadata,
                    'success': True
                }
            
        else:
            print(f"❌ Service error: {response.status_code}")
            print(f"📝 Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    return {'success': False}

if __name__ == "__main__":
    print("🔍 Dia Dialogue Formatting Test")
    print("Testing hypothesis: Dia needs multiple [S1] tags for longer audio")
    print("=" * 60)
    
    test_dialogue_formatting()
    
    print("\n🏁 Test completed!")
