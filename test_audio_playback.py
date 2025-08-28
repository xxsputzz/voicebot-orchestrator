"""
Audio playback utility for TTS testing - Optional enhancement
"""
import sys
import tempfile
import os
import asyncio
import time
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebot_orchestrator.tts import KokoroTTS


def play_tts_audio(text: str, save_file: bool = False) -> None:
    """
    Generate and optionally play TTS audio.
    
    Args:
        text: Text to synthesize
        save_file: Whether to save audio file
    """
    try:
        tts = KokoroTTS()
        
        print(f"🔊 Generating speech for: '{text}'")
        
        # Generate audio
        audio_data = asyncio.run(tts.synthesize_speech(text))
        
        print(f"✅ Generated {len(audio_data)} bytes of audio data")
        
        if save_file:
            # Save to file
            output_file = f"tts_output_{int(time.time())}.wav"
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            print(f"💾 Saved audio to: {output_file}")
            
            # Try to play with system default player (Windows)
            if os.name == 'nt':  # Windows
                try:
                    os.system(f'start "" "{output_file}"')
                    print("🎵 Playing audio with default player...")
                except Exception as e:
                    print(f"⚠️ Could not auto-play: {e}")
            else:
                print("💡 Open the saved WAV file with your audio player to hear the result")
        else:
            print("💡 Audio generated but not saved. Use --save flag to save and play.")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """CLI for audio playback testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TTS Audio Playback Test")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--save", action="store_true", help="Save audio file and try to play")
    
    args = parser.parse_args()
    
    play_tts_audio(args.text, args.save)


if __name__ == "__main__":
    main()
