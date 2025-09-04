"""
Download official voices from the Tortoise TTS repository.
This script downloads voice samples from the official neonbjb/tortoise-tts repository.
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
import time

# Official voice list from the repository
OFFICIAL_VOICES = {
    'angie': ['1.wav', '2.wav', '3.wav'],
    'daniel': ['1.wav', '2.wav', '3.wav'],
    'deniro': ['1.wav', '2.wav', '3.wav'],
    'emma': ['1.wav', '2.wav', '3.wav'],
    'freeman': ['1.wav', '2.wav', '3.wav'],
    'geralt': ['1.wav', '2.wav', '3.wav'],
    'halle': ['1.wav', '2.wav', '3.wav'],
    'jlaw': ['1.wav', '2.wav', '3.wav'],
    'lj': ['1.wav', '2.wav', '3.wav'],
    'myself': ['1.wav', '2.wav', '3.wav'],
    'pat': ['1.wav', '2.wav', '3.wav'],
    'snakes': ['1.wav', '2.wav', '3.wav'],
    'tom': ['1.wav', '2.wav', '3.wav'],
    'weaver': ['1.wav', '2.wav', '3.wav'],
    'william': ['1.wav', '2.wav', '3.wav'],
    'train_atkins': ['1.wav', '2.wav', '3.wav'],
    'train_dotrice': ['1.wav', '2.wav', '3.wav'],
    'train_kennard': ['1.wav', '2.wav', '3.wav'],
    'angelina': ['1.wav', '2.wav', '3.wav'],
    'craig': ['1.wav', '2.wav', '3.wav'],
    'mol': ['1.wav', '2.wav', '3.wav'],
    'rainbow': ['1.wav', '2.wav', '3.wav'],
}

BASE_URL = "https://raw.githubusercontent.com/neonbjb/tortoise-tts/main/tortoise/voices"
VOICES_DIR = Path("tortoise_voices")

def download_file(url, filepath, max_retries=3):
    """Download a file with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"  Downloading {filepath.name}... (attempt {attempt + 1})")
            
            # Add headers to avoid being blocked
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.read())
                    print(f"    ✓ Downloaded {filepath.name}")
                    return True
                else:
                    print(f"    ✗ HTTP {response.status} for {filepath.name}")
                    
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"    ✗ File not found: {filepath.name}")
                return False
            else:
                print(f"    ✗ HTTP Error {e.code}: {e.reason}")
                
        except Exception as e:
            print(f"    ✗ Error downloading {filepath.name}: {e}")
            
        if attempt < max_retries - 1:
            print(f"    Retrying in 2 seconds...")
            time.sleep(2)
    
    return False

def download_voices():
    """Download all official voices."""
    print("🎤 Downloading official Tortoise TTS voices...")
    print(f"Creating voices directory: {VOICES_DIR}")
    
    VOICES_DIR.mkdir(exist_ok=True)
    
    total_voices = len(OFFICIAL_VOICES)
    successful_voices = 0
    total_files = 0
    successful_files = 0
    
    for voice_name, files in OFFICIAL_VOICES.items():
        print(f"\n📁 Processing voice: {voice_name}")
        
        # Create voice directory
        voice_dir = VOICES_DIR / voice_name
        voice_dir.mkdir(exist_ok=True)
        
        voice_files_downloaded = 0
        
        for filename in files:
            url = f"{BASE_URL}/{voice_name}/{filename}"
            filepath = voice_dir / filename
            
            total_files += 1
            if download_file(url, filepath):
                successful_files += 1
                voice_files_downloaded += 1
            
            # Small delay between downloads to be respectful
            time.sleep(0.5)
        
        if voice_files_downloaded > 0:
            successful_voices += 1
            print(f"  ✓ {voice_name}: {voice_files_downloaded}/{len(files)} files downloaded")
        else:
            print(f"  ✗ {voice_name}: No files downloaded")
    
    print(f"\n🎯 Download Summary:")
    print(f"  Voices: {successful_voices}/{total_voices} successful")
    print(f"  Files: {successful_files}/{total_files} successful")
    print(f"  Success rate: {(successful_files/total_files)*100:.1f}%")
    
    if successful_voices > 0:
        print(f"\n✅ Voices downloaded to: {VOICES_DIR.absolute()}")
        print(f"You can now use these voices with Tortoise TTS!")
    else:
        print(f"\n❌ No voices were downloaded successfully.")
        print(f"This might be due to network issues or repository changes.")

def list_downloaded_voices():
    """List all downloaded voices and their files."""
    if not VOICES_DIR.exists():
        print(f"Voices directory {VOICES_DIR} does not exist.")
        return
    
    print(f"\n📋 Downloaded voices in {VOICES_DIR}:")
    
    for voice_dir in sorted(VOICES_DIR.iterdir()):
        if voice_dir.is_dir():
            wav_files = list(voice_dir.glob("*.wav"))
            if wav_files:
                print(f"  🎵 {voice_dir.name}: {len(wav_files)} files")
                for wav_file in sorted(wav_files):
                    size_mb = wav_file.stat().st_size / (1024 * 1024)
                    print(f"    - {wav_file.name} ({size_mb:.1f} MB)")
            else:
                print(f"  📂 {voice_dir.name}: No WAV files")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download official Tortoise TTS voices")
    parser.add_argument("--list", action="store_true", help="List downloaded voices")
    parser.add_argument("--voices", nargs="*", help="Download specific voices only")
    
    args = parser.parse_args()
    
    if args.list:
        list_downloaded_voices()
    else:
        if args.voices:
            # Filter to only specified voices
            filtered_voices = {k: v for k, v in OFFICIAL_VOICES.items() if k in args.voices}
            if filtered_voices:
                OFFICIAL_VOICES.clear()
                OFFICIAL_VOICES.update(filtered_voices)
                print(f"Downloading only specified voices: {', '.join(args.voices)}")
            else:
                print(f"None of the specified voices were found in the official list.")
                print(f"Available voices: {', '.join(OFFICIAL_VOICES.keys())}")
                exit(1)
        
        download_voices()
        print("\nTo list downloaded voices, run: python download_official_voices.py --list")
