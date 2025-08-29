#!/usr/bin/env python3
"""
Generate audio samples for all kokoro female voices.
Creates audio clips in tests/audio_samples directory to avoid cluttering main root folder.
Logs generation times for each voice model.
"""
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from voicebot_orchestrator.tts import KokoroTTS

def get_audio_output_path(filename):
    """Get the path for audio output files in tests/audio_samples directory."""
    # Create audio_samples directory if it doesn't exist
    audio_dir = os.path.join(os.path.dirname(__file__), "audio_samples")
    os.makedirs(audio_dir, exist_ok=True)
    return os.path.join(audio_dir, filename)

def get_log_file_path():
    """Get the path for the timing log file."""
    return get_audio_output_path("voice_generation_timing_log.txt")

def log_timing(voice_name, generation_time, text_length, success):
    """Log the timing information to a text file."""
    log_file = get_log_file_path()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create or append to log file
    with open(log_file, 'a', encoding='utf-8') as f:
        if os.path.getsize(log_file) == 0:  # If file is empty, add header
            f.write("Kokoro Female Voice Generation Timing Log\n")
            f.write("=" * 60 + "\n")
            f.write("Timestamp\t\tVoice\t\tTime(s)\tChars\tStatus\n")
            f.write("-" * 60 + "\n")
        
        status = "SUCCESS" if success else "FAILED"
        f.write(f"{timestamp}\t{voice_name:12}\t{generation_time:.3f}\t{text_length}\t{status}\n")

async def generate_voice_sample(voice_name, text, description):
    """Generate a voice sample for a specific voice with timing measurement."""
    print(f"🎙️ Generating {voice_name} voice sample")
    print(f"📝 Text: {text}")
    print(f"🎭 Voice: {voice_name} ({description})")
    
    start_time = time.time()
    success = False
    
    try:
        # Initialize TTS with the specific voice
        tts = KokoroTTS(voice=voice_name)
        
        # Generate audio
        print("🔊 Generating audio...")
        audio_start = time.time()
        audio_bytes = await tts.synthesize_speech(text)
        generation_time = time.time() - audio_start
        
        print(f"✅ Success! Generated {len(audio_bytes)} bytes of audio in {generation_time:.3f} seconds")
        
        # Save to file with clear name
        filename = f"{voice_name}_banking_sample.wav"
        file_path = get_audio_output_path(filename)
        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
        
        print(f"💾 Saved to: {file_path}")
        print(f"⏱️ Generation time: {generation_time:.3f} seconds")
        success = True
        
        # Log the timing
        log_timing(voice_name, generation_time, len(text), success)
        
        print("=" * 60)
        return True
        
    except Exception as e:
        generation_time = time.time() - start_time
        print(f"❌ Error generating {voice_name}: {e}")
        print(f"⏱️ Failed after: {generation_time:.3f} seconds")
        
        # Log the failed attempt
        log_timing(voice_name, generation_time, len(text), success)
        
        print("=" * 60)
        return False

async def generate_all_female_voices():
    """Generate audio samples for all requested kokoro female voices."""
    print("🎵 Kokoro Female Voice Generator")
    print("=" * 60)
    print("📁 Output directory: tests/audio_samples/")
    print("📊 Timing log: voice_generation_timing_log.txt")
    print("=" * 60)
    
    # Clear/initialize the log file for this session
    log_file = get_log_file_path()
    if os.path.exists(log_file):
        # Add session separator to existing log
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n--- New Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    
    # Define the female voices and their descriptions
    female_voices = {
        "af_alloy": "Female American English - Alloy",
        "af_aoede": "Female American English - Aoede", 
        "af_bella": "Female American English - Bella",
        "af_kore": "Female American English - Kore",
        "af_nicole": "Female American English - Nicole",
        "af_nova": "Female American English - Nova",
        "af_sarah": "Female American English - Sarah"
    }
    
    # Banking-related test text for each voice
    test_texts = {
        "af_alloy": "Hello! I'm Alloy from First National Bank. I can help you check your account balance, transfer funds, or answer questions about our banking services.",
        "af_aoede": "Welcome to First National Bank! I'm Aoede, your virtual banking assistant. How may I assist you with your financial needs today?",
        "af_bella": "Hello! Welcome to First National Bank. I'm Bella, your AI banking assistant. How can I help you with your account today?",
        "af_kore": "Good day! This is Kore from First National Bank. I'm here to help you with account inquiries, loan applications, and payment processing.",
        "af_nicole": "Hi there! I'm Nicole, your personal banking assistant at First National Bank. Let me help you manage your finances efficiently.",
        "af_nova": "Welcome! I'm Nova from First National Bank's customer service team. I can assist you with transactions, account management, and financial planning.",
        "af_sarah": "Hello and welcome to First National Bank! I'm Sarah, ready to help you with all your banking needs, from simple transfers to investment advice."
    }
    
    successful_generations = 0
    total_voices = len(female_voices)
    total_start_time = time.time()
    
    # Generate samples for each voice
    for voice_name, description in female_voices.items():
        text = test_texts.get(voice_name, "Hello! This is a test of the voice synthesis system.")
        success = await generate_voice_sample(voice_name, text, description)
        if success:
            successful_generations += 1
        
        # Small delay between generations to avoid overwhelming the system
        await asyncio.sleep(0.5)
    
    total_time = time.time() - total_start_time
    
    # Summary
    print("🎯 Generation Summary")
    print("=" * 60)
    print(f"✅ Successfully generated: {successful_generations}/{total_voices} voices")
    print(f"⏱️ Total generation time: {total_time:.3f} seconds")
    print(f"📁 Audio files saved in: {get_audio_output_path('')}")
    print(f"📊 Timing log saved to: {get_log_file_path()}")
    
    # Add summary to log file
    with open(get_log_file_path(), 'a', encoding='utf-8') as f:
        f.write(f"\nSession Summary: {successful_generations}/{total_voices} successful, Total time: {total_time:.3f}s\n")
        f.write("=" * 60 + "\n\n")
    
    if successful_generations == total_voices:
        print("🎉 All female voice samples generated successfully!")
    elif successful_generations > 0:
        print("⚠️ Some voice samples were generated, but there were issues with others.")
    else:
        print("❌ Failed to generate any voice samples. Check your Kokoro TTS setup.")
    
    print("=" * 60)
    print("📋 Generated files:")
    audio_dir = get_audio_output_path("")
    try:
        for file in sorted(os.listdir(audio_dir)):
            if file.endswith('.wav') and any(voice in file for voice in female_voices.keys()):
                file_path = os.path.join(audio_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"   🎵 {file} ({file_size:,} bytes)")
        
        # Show log file
        if os.path.exists(get_log_file_path()):
            log_size = os.path.getsize(get_log_file_path())
            print(f"   📊 voice_generation_timing_log.txt ({log_size:,} bytes)")
    except Exception as e:
        print(f"   ❌ Error listing files: {e}")

if __name__ == "__main__":
    print("🚀 Starting Kokoro Female Voice Generation...")
    asyncio.run(generate_all_female_voices())
