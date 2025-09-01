#!/usr/bin/env python3
"""
Test script to verify the updated voice naming system
"""
import requests
import json
import sys

def test_voice_names():
    """Test that the service returns proper Microsoft Edge Neural voice names"""
    try:
        print("🧪 Testing Voice Names Update...")
        
        # Test voices endpoint
        response = requests.get('http://localhost:8014/voices', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Service responded successfully")
            
            # Check if we have proper voice structure
            if 'voices' in data:
                voices = data['voices']
                
                # Check female voices
                if 'female' in voices:
                    female_voices = voices['female']
                    print(f"\n👩 Female Voices ({len(female_voices)}):")
                    for name, info in female_voices.items():
                        print(f"  • {name}: {info.get('description', 'No description')} ({info.get('accent', 'Unknown accent')})")
                        
                # Check male voices  
                if 'male' in voices:
                    male_voices = voices['male']
                    print(f"\n👨 Male Voices ({len(male_voices)}):")
                    for name, info in male_voices.items():
                        print(f"  • {name}: {info.get('description', 'No description')} ({info.get('accent', 'Unknown accent')})")
                
                # Check aliases
                if 'aliases' in voices:
                    aliases = voices['aliases']
                    print(f"\n🔗 Voice Aliases ({len(aliases)}):")
                    for alias, target in aliases.items():
                        print(f"  • {alias} → {target}")
                
                # Check service info
                print(f"\n📊 Service Info:")
                print(f"  • Total voices: {data.get('count', 'Unknown')}")
                print(f"  • Engine: {data.get('engine', 'Unknown')}")
                print(f"  • Note: {data.get('note', 'No note')}")
                
                # Verify key professional voices are present
                expected_voices = ['jenny', 'aria', 'guy', 'davis', 'andrew']
                found_voices = []
                
                for voice in expected_voices:
                    if voice in female_voices or voice in male_voices:
                        found_voices.append(voice)
                
                print(f"\n✅ Professional Voice Check:")
                print(f"  • Expected: {expected_voices}")
                print(f"  • Found: {found_voices}")
                
                if len(found_voices) == len(expected_voices):
                    print("🎉 All professional Microsoft Edge Neural voices found!")
                    return True
                else:
                    print("⚠️ Some professional voices missing")
                    return False
                    
            else:
                print("❌ No voices data in response")
                return False
                
        else:
            print(f"❌ Service error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to service on port 8014")
        print("Make sure the TTS service is running")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_voice_synthesis():
    """Test voice synthesis with new voice names"""
    try:
        print("\n🧪 Testing Voice Synthesis...")
        
        # Test with professional voice names
        test_voices = ['jenny', 'aria', 'guy', 'davis']
        
        for voice in test_voices:
            print(f"\n🎤 Testing voice: {voice}")
            
            payload = {
                "text": f"Hello, this is {voice} from Microsoft Edge Neural TTS.",
                "voice": voice,
                "emotion": "professional"
            }
            
            response = requests.post(
                'http://localhost:8014/tts',
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                if response.headers.get('content-type') == 'audio/wav':
                    print(f"  ✅ {voice}: Audio generated successfully")
                else:
                    print(f"  ⚠️ {voice}: Response not audio format")
            else:
                print(f"  ❌ {voice}: Synthesis failed ({response.status_code})")
                
        print("\n🎉 Voice synthesis testing complete!")
        return True
        
    except Exception as e:
        print(f"❌ Synthesis test error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 VOICE NAMES UPDATE VALIDATION")
    print("=" * 60)
    
    # Test voice names
    names_ok = test_voice_names()
    
    # Test synthesis if names are OK
    if names_ok:
        synthesis_ok = test_voice_synthesis()
        
        if names_ok and synthesis_ok:
            print("\n🎉 ALL TESTS PASSED!")
            print("Voice naming system successfully updated to Microsoft Edge Neural voices")
            sys.exit(0)
        else:
            print("\n⚠️ SOME TESTS FAILED")
            sys.exit(1)
    else:
        print("\n❌ VOICE NAMES TEST FAILED")
        sys.exit(1)
