#!/usr/bin/env python3
"""
Tortoise TTS Voice Diagnostics
Check what voices are actually available vs. configured
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def diagnose_voice_availability():
    """Diagnose what voices are available vs configured"""
    print("🎭 Tortoise TTS Voice Diagnostics")
    print("=" * 60)
    
    try:
        # Import the configuration
        from tortoise_tts_implementation_real import TortoiseVoiceConfig
        
        print(f"\n📋 Configured Voices ({len(TortoiseVoiceConfig.VOICE_CONFIGS)}):")
        for i, voice in enumerate(sorted(TortoiseVoiceConfig.VOICE_CONFIGS.keys()), 1):
            print(f"   {i:2}. {voice}")
        
        # Try to import Tortoise TTS and check what it actually finds
        print(f"\n🔍 Checking actual Tortoise TTS voice discovery...")
        
        try:
            from tortoise.utils.audio import load_voices
            
            # Check common voice directories
            possible_voice_dirs = [
                "voices",
                "tortoise/voices", 
                "tortoise_tts/voices",
                os.path.expanduser("~/.cache/tortoise/voices"),
                os.path.expanduser("~/AppData/Local/tortoise/voices"),
            ]
            
            print(f"\n📂 Searching voice directories:")
            found_voices = set()
            
            for voice_dir in possible_voice_dirs:
                voice_path = Path(voice_dir)
                print(f"   📁 {voice_path.absolute()}: ", end="")
                
                if voice_path.exists() and voice_path.is_dir():
                    voices_in_dir = [d.name for d in voice_path.iterdir() if d.is_dir()]
                    if voices_in_dir:
                        print(f"✅ Found {len(voices_in_dir)} voices: {', '.join(sorted(voices_in_dir))}")
                        found_voices.update(voices_in_dir)
                    else:
                        print("📁 Empty directory")
                else:
                    print("❌ Not found")
            
            # Try loading each configured voice to see which ones work
            print(f"\n🧪 Testing voice loading (this may take a moment)...")
            working_voices = []
            failed_voices = []
            
            for voice in sorted(TortoiseVoiceConfig.VOICE_CONFIGS.keys()):
                try:
                    print(f"   Testing {voice}... ", end="", flush=True)
                    voice_samples, conditioning_latents = load_voices([voice])
                    if voice_samples is not None and conditioning_latents is not None:
                        working_voices.append(voice)
                        print("✅")
                    else:
                        failed_voices.append(voice)
                        print("❌ (no data)")
                except Exception as e:
                    failed_voices.append(voice)
                    print(f"❌ ({str(e)[:30]}...)")
            
            print(f"\n📊 Results:")
            print(f"   ✅ Working voices ({len(working_voices)}): {', '.join(working_voices)}")
            print(f"   ❌ Failed voices ({len(failed_voices)}): {', '.join(failed_voices)}")
            
            # Check if there are built-in voices
            print(f"\n🔍 Checking for built-in Tortoise voices...")
            try:
                # Try some common built-in voice names
                builtin_candidates = ['angie', 'freeman', 'denise', 'pat', 'william', 'tom', 'lj']
                builtin_working = []
                
                for voice in builtin_candidates:
                    try:
                        voice_samples, conditioning_latents = load_voices([voice])
                        if voice_samples is not None:
                            builtin_working.append(voice)
                    except:
                        pass
                
                if builtin_working:
                    print(f"   ✅ Built-in voices found: {', '.join(builtin_working)}")
                else:
                    print(f"   ❌ No built-in voices found")
                    
            except Exception as e:
                print(f"   ❌ Error checking built-in voices: {e}")
            
        except ImportError as e:
            print(f"   ❌ Cannot import Tortoise TTS: {e}")
            print(f"   This suggests Tortoise TTS is not properly installed")
        
    except ImportError as e:
        print(f"❌ Cannot import voice configuration: {e}")
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")

def suggest_voice_fixes():
    """Suggest how to fix voice availability issues"""
    print(f"\n🔧 Potential Solutions:")
    print(f"   1. Download Tortoise TTS voices:")
    print(f"      - Official voices are usually downloaded automatically")
    print(f"      - Check if voices are in the correct directory")
    
    print(f"\n   2. Voice directory locations:")
    print(f"      - ./voices/")
    print(f"      - ~/.cache/tortoise/voices/")
    print(f"      - Python site-packages/tortoise/voices/")
    
    print(f"\n   3. Force voice download:")
    print(f"      - Some Tortoise versions download voices on first use")
    print(f"      - Try running a synthesis with each voice to trigger download")
    
    print(f"\n   4. Manual voice installation:")
    print(f"      - Download voices from Tortoise TTS repository")
    print(f"      - Place in appropriate voices/ directory")
    
    print(f"\n   5. Update our voice configuration:")
    print(f"      - Remove non-existent voices from VOICE_CONFIGS")
    print(f"      - Only include voices that actually work")

if __name__ == "__main__":
    diagnose_voice_availability()
    suggest_voice_fixes()
