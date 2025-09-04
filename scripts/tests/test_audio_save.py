#!/usr/bin/env python3
"""
Demo: Tortoise TTS Auto-Save Functionality
Shows how to control audio file saving
"""

from tortoise_tts_implementation_real import TortoiseTTSEngine, TortoiseTTS
import os

def main():
    print("🎤 Tortoise TTS Auto-Save Demo")
    print("=" * 50)
    
    # Initialize the engine
    print("🔄 Initializing Tortoise TTS Engine...")
    engine = TortoiseTTSEngine()
    
    # Test 1: Auto-save enabled (default)
    print("\n📝 Test 1: Auto-save enabled (default)")
    audio1, sr1 = engine.generate_speech(
        "This audio will be automatically saved!",
        voice="angie",
        save_audio=True  # This is the default
    )
    print(f"✅ Generated {audio1.shape[0]/sr1:.2f} seconds of audio")
    
    # Test 2: Auto-save disabled
    print("\n📝 Test 2: Auto-save disabled")
    audio2, sr2 = engine.generate_speech(
        "This audio will NOT be saved automatically.",
        voice="tom", 
        save_audio=False
    )
    print(f"✅ Generated {audio2.shape[0]/sr2:.2f} seconds of audio (not saved)")
    
    # Test 3: Manual save with custom filename
    print("\n📝 Test 3: Manual save with custom filename")
    audio3, sr3 = engine.generate_speech(
        "I will save this manually with a custom name.",
        voice="william",
        save_audio=False  # Don't auto-save
    )
    
    # Save manually with custom filename
    custom_path = engine.save_audio_to_file(
        audio3, sr3, 
        filename="my_custom_audio.wav",
        text="Custom filename demo",
        voice="william"
    )
    print(f"✅ Manually saved to: {custom_path}")
    
    # Test 4: Service-level synthesis with auto-save
    print("\n📝 Test 4: Service-level auto-save")
    tts_service = TortoiseTTS()
    base64_audio, metadata = tts_service.synthesize_to_base64(
        "Service-level synthesis with auto-save enabled!",
        voice="grace",
        save_audio=True
    )
    print(f"✅ Service synthesis complete: {metadata['duration']:.2f}s audio")
    
    # Show saved files
    print("\n📁 Audio files in audio_output directory:")
    audio_files = os.listdir("audio_output")
    for i, file in enumerate(audio_files, 1):
        file_path = os.path.join("audio_output", file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"  {i}. {file} ({file_size:.1f} KB)")
    
    print(f"\n🎉 Demo complete! Generated {len(audio_files)} audio files.")
    print("💡 Tip: Set save_audio=False to disable auto-saving")
    print("💾 All audio files are saved in the 'audio_output' folder")

if __name__ == "__main__":
    main()
