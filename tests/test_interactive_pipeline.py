#!/usr/bin/env python3
"""
Interactive Pipeline Tester
============================

Interactive test menu for testing specific service combinations:
1. Select which services to test (STT, LLM, TTS)
2. Choose individual pipeline components or full pipeline
3. Test only what you want to test
4. Detailed error reporting for debugging

Usage:
    python tests/test_interactive_pipeline.py
"""

import asyncio
import sys
import time
import json
import base64
import requests
import pyaudio
import wave
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import traceback
from contextlib import contextmanager

from contextlib import contextmanager

def chunk_text_intelligently(text: str, max_chunk_size: int = 2800) -> List[str]:
    """
    Split text into chunks that respect sentence boundaries and stay under max_chunk_size.
    Leaves some buffer under the 3000 character limit for safety.
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    remaining_text = text
    
    while remaining_text:
        if len(remaining_text) <= max_chunk_size:
            chunks.append(remaining_text.strip())
            break
        
        # Find the best split point (sentence boundary)
        chunk = remaining_text[:max_chunk_size]
        
        # Look for sentence endings (., !, ?) followed by space or newline
        best_split = -1
        for i in range(len(chunk) - 1, max(0, len(chunk) - 200), -1):
            if chunk[i] in '.!?' and i + 1 < len(chunk) and chunk[i + 1] in ' \n\t':
                best_split = i + 1
                break
        
        # If no sentence boundary found, look for other good split points
        if best_split == -1:
            for i in range(len(chunk) - 1, max(0, len(chunk) - 100), -1):
                if chunk[i] in ',;:' and i + 1 < len(chunk) and chunk[i + 1] in ' \n\t':
                    best_split = i + 1
                    break
        
        # If still no good split point, look for word boundaries
        if best_split == -1:
            for i in range(len(chunk) - 1, max(0, len(chunk) - 50), -1):
                if chunk[i] == ' ':
                    best_split = i + 1
                    break
        
        # If no word boundary, just split at max size (shouldn't happen with normal text)
        if best_split == -1:
            best_split = max_chunk_size
        
        chunks.append(remaining_text[:best_split].strip())
        remaining_text = remaining_text[best_split:].strip()
    
    return [chunk for chunk in chunks if chunk]  # Remove empty chunks

@contextmanager
def suppress_http_logs():
    """Temporarily suppress HTTP and service logs during health checks"""
    # Suppress uvicorn, httpx, and other HTTP-related logs
    loggers_to_suppress = [
        'uvicorn.access',
        'uvicorn.error', 
        'httpx',
        'requests.packages.urllib3.connectionpool',
        'urllib3.connectionpool'
    ]
    
    original_levels = {}
    try:
        for logger_name in loggers_to_suppress:
            logger = logging.getLogger(logger_name)
            original_levels[logger_name] = logger.level
            logger.setLevel(logging.WARNING)  # Only show warnings and errors
        yield
    finally:
        # Restore original logging levels
        for logger_name, original_level in original_levels.items():
            logging.getLogger(logger_name).setLevel(original_level)

class InteractivePipelineTester:
    """Interactive tester for specific service combinations"""
    
    def __init__(self):
        """Initialize the interactive tester"""
        self.base_dir = Path(__file__).parent
        self.audio_dir = self.base_dir / "audio_samples" / "interactive_pipeline"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # All possible service endpoints
        self.all_services = {
            "orchestrator": {"url": "http://localhost:8000", "type": "orchestrator"},
            "whisper_stt": {"url": "http://localhost:8003", "type": "stt"},
            "kokoro_tts": {"url": "http://localhost:8011", "type": "tts"},
            "hira_dia_tts": {"url": "http://localhost:8012", "type": "tts"}, 
            "dia_4bit_tts": {"url": "http://localhost:8013", "type": "tts"},
            "zonos_tts": {"url": "http://localhost:8014", "type": "tts"},
            "mistral_llm": {"url": "http://localhost:8021", "type": "llm"},
            "gpt_llm": {"url": "http://localhost:8022", "type": "llm"}
        }
        
        # Track which services are available
        self.available_services = {}
        
        # Audio recording settings
        self.recording_settings = {
            'format': pyaudio.paInt16,
            'channels': 1,
            'rate': 16000,  # 16kHz for speech
            'chunk': 1024,
            'record_seconds': 5,  # Default recording duration
        }
        
    def record_microphone_audio(self, duration: int = 5) -> str:
        """Record audio from microphone and save to file"""
        print(f"🎤 Recording audio for {duration} seconds...")
        print("   Speak now...")
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        try:
            # Open microphone stream
            stream = audio.open(
                format=self.recording_settings['format'],
                channels=self.recording_settings['channels'],
                rate=self.recording_settings['rate'],
                input=True,
                frames_per_buffer=self.recording_settings['chunk']
            )
            
            frames = []
            
            # Record audio
            for i in range(0, int(self.recording_settings['rate'] / self.recording_settings['chunk'] * duration)):
                data = stream.read(self.recording_settings['chunk'])
                frames.append(data)
            
            print("✅ Recording completed!")
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recorded_audio_{timestamp}.wav"
            filepath = self.audio_dir / filename
            
            # Write WAV file
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(self.recording_settings['channels'])
                wf.setsampwidth(audio.get_sample_size(self.recording_settings['format']))
                wf.setframerate(self.recording_settings['rate'])
                wf.writeframes(b''.join(frames))
            
            print(f"🎵 Audio saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            return None
        finally:
            audio.terminate()
    
    def chunk_text_intelligently(self, text: str, max_chars: int = 2900) -> List[str]:
        """Split text into chunks that respect sentence boundaries"""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first (looking for . ! ? followed by space or newline)
        import re
        sentences = re.split(r'([.!?]+(?:\s|\n|$))', text)
        
        # Recombine sentences with their punctuation
        sentence_list = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence_list.append(sentences[i] + sentences[i + 1])
            else:
                sentence_list.append(sentences[i])
        
        for sentence in sentence_list:
            # If adding this sentence would exceed the limit
            if len(current_chunk) + len(sentence) > max_chars:
                if current_chunk:  # Save current chunk if it has content
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence too long - split by words
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 <= max_chars:
                            temp_chunk += (" " if temp_chunk else "") + word
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word
                    if temp_chunk:
                        current_chunk = temp_chunk
            else:
                current_chunk += (" " if current_chunk else "") + sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
        
    def check_service_health(self, service_name: str, endpoint: str) -> bool:
        """Check if a service is running and healthy"""
        try:
            # Try health endpoint first - disable requests logging to reduce noise
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            health_response = requests.get(f"{endpoint}/health", timeout=3)
            if health_response.status_code == 200:
                return True
        except:
            pass
        
        try:
            # Try root endpoint as fallback
            root_response = requests.get(endpoint, timeout=3)
            if root_response.status_code in [200, 404]:  # 404 is OK for some services
                return True
        except:
            pass
        
        return False
    
    def detect_available_services(self) -> Dict:
        """Detect which services are currently available"""
        print("🔍 Checking service availability...")
        print("-" * 40)
        
        available = {}
        
        with suppress_http_logs():  # Suppress INFO logs during health checks
            for service_name, config in self.all_services.items():
                endpoint = config["url"]
                if self.check_service_health(service_name, endpoint):
                    available[service_name] = config
                    print(f"  ✅ {service_name:<20} - {endpoint}")
                else:
                    print(f"  ❌ {service_name:<20} - Not available")
        
        self.available_services = available
        
        print(f"\n📊 Available: {len(available)} services")
        return available
    
    def select_services_by_type(self, service_type: str) -> Optional[str]:
        """Let user select a service of specific type"""
        # If no services detected at all, try to find them in the config
        if not self.available_services:
            print(f"💡 Attempting to find available {service_type.upper()} services...")
            potential_services = {name: config for name, config in self.all_services.items() 
                                if config["type"] == service_type}
            
            # Quick check if any are available
            for service_name, config in potential_services.items():
                if self.check_service_health(service_name, config["url"]):
                    self.available_services[service_name] = config
                    print(f"✅ Found and selected {service_type.upper()}: {service_name}")
                    return service_name
            
            print(f"❌ No {service_type.upper()} services are running")
            return None
        
        available_of_type = {name: config for name, config in self.available_services.items() 
                           if config["type"] == service_type}
        
        if not available_of_type:
            print(f"❌ No {service_type.upper()} services available")
            return None
        
        if len(available_of_type) == 1:
            service_name = list(available_of_type.keys())[0]
            print(f"✅ Auto-selected {service_type.upper()}: {service_name}")
            return service_name
        
        print(f"\n📋 Available {service_type.upper()} services:")
        services_list = list(available_of_type.keys())
        for i, service_name in enumerate(services_list, 1):
            endpoint = available_of_type[service_name]["url"]
            print(f"  {i}. {service_name} - {endpoint}")
        
        while True:
            try:
                choice = input(f"\nSelect {service_type.upper()} service (1-{len(services_list)}): ").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(services_list):
                        selected = services_list[idx]
                        print(f"✅ Selected {service_type.upper()}: {selected}")
                        return selected
                print(f"❌ Please enter a number between 1 and {len(services_list)}")
            except KeyboardInterrupt:
                return None
    
    async def test_stt_to_llm(self, stt_service: str, llm_service: str) -> bool:
        """Test STT → LLM pipeline component with real microphone recording"""
        print(f"\n🎙️ Testing {stt_service} → {llm_service}")
        print("-" * 50)
        
        try:
            # Option to use microphone or text input
            use_mic = input("Use microphone recording? (y/n, default=y): ").strip().lower()
            if use_mic in ['', 'y', 'yes']:
                # Record from microphone
                duration = input("Recording duration in seconds (default=5): ").strip()
                try:
                    duration = int(duration) if duration else 5
                    duration = max(1, min(duration, 30))  # Limit between 1-30 seconds
                except ValueError:
                    duration = 5
                
                print(f"\n🎤 Get ready to speak in 3 seconds...")
                time.sleep(3)
                
                audio_file = self.record_microphone_audio(duration)
                if not audio_file:
                    print("❌ Recording failed, falling back to text input")
                    test_text = input("Enter test text: ").strip() or "Hello, how are you today?"
                    print(f"📝 Using text input: '{test_text}'")
                else:
                    # Try STT service first, then direct Whisper as backup
                    print(f"🔄 Attempting transcription with recorded audio...")
                    
                    try:
                        # Send audio to STT service
                        stt_endpoint = self.available_services[stt_service]["url"]
                        print(f"🔄 Trying STT service: {stt_endpoint}")
                        
                        with open(audio_file, 'rb') as f:
                            files = {'audio': f}
                            stt_response = requests.post(f"{stt_endpoint}/transcribe", files=files, timeout=30)
                        
                        if stt_response.status_code == 200:
                            stt_result = stt_response.json()
                            test_text = stt_result.get("text", "").strip()
                            confidence = stt_result.get("confidence", 0.0)
                            
                            if test_text:
                                print(f"✅ STT Service Success: '{test_text}' (confidence: {confidence:.2f})")
                            else:
                                print(f"⚠️  STT service returned empty transcript")
                                test_text = ""
                        else:
                            print(f"❌ STT service failed: {stt_response.status_code}")
                            test_text = ""
                    
                    except Exception as e:
                        print(f"❌ STT service error: {e}")
                        test_text = ""
                    
                    # If STT service failed or returned empty, try direct Whisper
                    if not test_text:
                        print("🔄 Trying direct Whisper transcription...")
                        try:
                            import sys
                            import os
                            
                            # Add the project root to Python path
                            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            if project_root not in sys.path:
                                sys.path.insert(0, project_root)
                            
                            from voicebot_orchestrator.real_whisper_stt import WhisperSTT
                            
                            whisper = WhisperSTT()
                            test_text = await whisper.transcribe_file(audio_file)
                            test_text = test_text.strip() if test_text else ""
                            
                            if test_text:
                                print(f"✅ Direct Whisper Success: '{test_text}'")
                            else:
                                print("⚠️  Direct Whisper also returned empty")
                                
                        except Exception as e:
                            print(f"❌ Direct Whisper failed: {e}")
                            # Try alternative approach with simple whisper import
                            try:
                                import whisper
                                print("🔄 Trying fallback Whisper approach...")
                                
                                model = whisper.load_model("base")
                                result = model.transcribe(audio_file)
                                test_text = result["text"].strip()
                                
                                if test_text:
                                    print(f"✅ Fallback Whisper Success: '{test_text}'")
                                else:
                                    print("⚠️  Fallback Whisper also returned empty")
                                    
                            except Exception as e2:
                                print(f"❌ Fallback Whisper also failed: {e2}")
                    
                    # Final fallback if both methods failed
                    if not test_text:
                        print("⚠️  All transcription methods failed - using fallback")
                        test_text = "Hello, how are you today?"
            else:
                # Text input mode
                test_text = input("Enter test text (or press Enter for default): ").strip()
                if not test_text:
                    test_text = "Hello, how are you today?"
                print(f"📝 Input text: '{test_text}'")
            
            # Send to LLM
            llm_endpoint = self.available_services[llm_service]["url"]
            llm_data = {
                "text": test_text,
                "temperature": 0.7,
                "max_tokens": 150
            }
            
            print(f"🔄 Sending to LLM: {llm_endpoint}")
            
            llm_response = requests.post(f"{llm_endpoint}/generate", json=llm_data, timeout=30)
            
            if llm_response.status_code == 200:
                llm_result = llm_response.json()
                llm_text = llm_result.get("response", "").strip()
                print(f"✅ LLM Response: '{llm_text[:200]}...'")
                return True
            else:
                print(f"❌ LLM request failed: {llm_response.status_code}")
                print(f"   Response: {llm_response.text}")
                return False
                
        except Exception as e:
            print(f"❌ STT→LLM test failed: {e}")
            traceback.print_exc()
            return False
    
    async def test_llm_to_tts(self, llm_service: str, tts_service: str) -> bool:
        """Test LLM → TTS pipeline component"""
        print(f"\n🧠 Testing {llm_service} → {tts_service}")
        print("-" * 50)
        
        try:
            # Get input text for LLM
            input_text = input("Enter input for LLM (or press Enter for default): ").strip()
            if not input_text:
                input_text = "Tell me a short joke"
            
            print(f"📝 LLM Input: '{input_text}'")
            
            # Send to LLM first
            llm_endpoint = self.available_services[llm_service]["url"]
            llm_data = {
                "text": input_text,
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            print(f"🔄 Sending to LLM: {llm_endpoint}")
            
            llm_response = requests.post(f"{llm_endpoint}/generate", json=llm_data, timeout=30)
            if llm_response.status_code != 200:
                print(f"❌ LLM request failed: {llm_response.status_code}")
                print(f"   Response: {llm_response.text}")
                return False
                
            llm_result = llm_response.json() 
            llm_text = llm_result.get("response", "").strip()
            print(f"✅ LLM Output: '{llm_text[:100]}...'")
            
            # Send LLM output to TTS
            tts_endpoint = self.available_services[tts_service]["url"]
            tts_data = {
                "text": llm_text,
                "voice": "af_bella",  # Use a valid Kokoro voice instead of "default"
                "speed": 1.0,
                "return_audio": True
            }
            
            print(f"🔄 Sending to TTS: {tts_endpoint}")
            
            tts_response = requests.post(f"{tts_endpoint}/synthesize", json=tts_data, timeout=120)  # Increased for Nari Dia
            if tts_response.status_code == 200:
                tts_result = tts_response.json()
                audio_data = tts_result.get("audio_base64")
                
                if audio_data:
                    # Save audio file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"llm_to_tts_{llm_service}_{tts_service}_{timestamp}.wav"
                    filepath = self.audio_dir / filename
                    
                    audio_bytes = base64.b64decode(audio_data)
                    with open(filepath, "wb") as f:
                        f.write(audio_bytes)
                    
                    print(f"✅ Audio saved: {filename}")
                    return True
                else:
                    print(f"❌ No audio data received from TTS")
                    return False
            else:
                print(f"❌ TTS request failed: {tts_response.status_code}")
                print(f"   Response: {tts_response.text}")
                return False
                
        except Exception as e:
            print(f"❌ LLM→TTS test failed: {e}")
            traceback.print_exc()
            return False
    
    async def test_typed_text_to_tts(self, tts_service: str) -> bool:
        """Test direct text input to TTS with unlimited timeout and custom options"""
        print(f"\n🎤 Testing Typed Text → {tts_service}")
        print("-" * 50)
        
        try:
            # Multi-line text input with full preview and editing capability
            print("📝 Enter your text for TTS synthesis:")
            print("   💡 Instructions:")
            print("      - Paste or type your text (unlimited length)")
            print("      - Press Enter after each line")
            print("      - Type 'DONE' on a new line when finished")
            print("      - Type 'CANCEL' to cancel")
            print("      - Leave empty and type 'DONE' for default text")
            print()
            
            lines = []
            line_count = 1
            
            print("📖 Enter your text (type 'DONE' when finished):")
            
            while True:
                try:
                    line = input(f"Line {line_count:2d}: ")
                    
                    if line.strip().upper() == "DONE":
                        break
                    elif line.strip().upper() == "CANCEL":
                        print("❌ Cancelled by user")
                        return False
                    else:
                        lines.append(line)
                        line_count += 1
                        
                except KeyboardInterrupt:
                    print("\n❌ Input cancelled.")
                    return False
                except EOFError:
                    break
            
            # Handle the text input
            if not lines:
                text_input = "Hello! This is a test of the text-to-speech system with unlimited character support. You can paste or type as much text as you want, and the system will generate high-quality speech from it. How does this sound?"
                print(f"\n📝 Using default text")
            else:
                text_input = "\n".join(lines)  # Preserve line breaks
                print(f"\n📝 Text entered successfully!")
            
            # Show complete text for review
            char_count = len(text_input)
            word_count = len(text_input.split())
            line_count = len(text_input.split('\n'))
            
            print(f"\n� Text Statistics:")
            print(f"   📏 Length: {char_count:,} characters")
            print(f"   📝 Words: {word_count:,}")
            print(f"   📄 Lines: {line_count:,}")
            
            print(f"\n📖 Complete Text Preview:")
            print("-" * 60)
            print(text_input)
            print("-" * 60)
            
            # Confirm before processing
            while True:
                confirm = input(f"\n✅ Process this text? (y/n/edit): ").strip().lower()
                if confirm in ['y', 'yes']:
                    break
                elif confirm in ['n', 'no']:
                    print("❌ Cancelled by user")
                    return False
                elif confirm in ['e', 'edit']:
                    print("📝 Edit mode not implemented yet. Please restart text entry.")
                    return False
                else:
                    print("❌ Please enter 'y' (yes), 'n' (no), or 'edit'")
            
            if len(text_input) > 10000:  # Warn for very long text
                print(f"\n⚠️  Large text detected ({len(text_input):,} characters)")
                print(f"   This may take several minutes to generate")
                continue_choice = input("   Continue? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    print("❌ Cancelled by user")
                    return False
            
            # Optional seed setting
            seed_input = input("\n🎲 Enter random seed (leave empty for random): ").strip()
            if seed_input:
                try:
                    seed = int(seed_input)
                    print(f"   🎲 Using seed: {seed}")
                except ValueError:
                    import random
                    seed = random.randint(1, 999999)
                    print(f"   🎲 Invalid seed, using random: {seed}")
            else:
                import random
                seed = random.randint(1, 999999)
                print(f"   🎲 Using random seed: {seed}")
            
            # Optional quality setting
            quality_choice = input("\nHigh quality mode? (y/n, default=y): ").strip().lower()
            high_quality = quality_choice in ['', 'y', 'yes']
            
            # Optional speed setting  
            speed_input = input("Speech speed (0.5-2.0, default=1.0): ").strip()
            try:
                speed = float(speed_input) if speed_input else 1.0
                speed = max(0.5, min(speed, 2.0))  # Clamp between 0.5 and 2.0
            except ValueError:
                speed = 1.0
            
            print(f"\n🔄 Sending to TTS: {self.available_services[tts_service]['url']}")
            print(f"   📊 Text length: {len(text_input)} characters")
            print(f"   🎯 High quality: {high_quality}")
            print(f"   ⚡ Speed: {speed}x")
            print(f"   🎲 Seed: {seed}")
            print(f"   ⏳ No timeout limit - will wait for completion")
            
            # Prepare TTS request
            tts_endpoint = self.available_services[tts_service]["url"]
            print(f"\n🔄 Sending to TTS: {tts_endpoint}")
            print(f"   📊 Text length: {len(text_input)} characters")
            print(f"   🎯 High quality: {high_quality}")
            print(f"   ⚡ Speed: {speed}x")
            print(f"   ⏳ No timeout limit - will wait for completion")
            
            # Check if text needs chunking for Dia TTS
            if len(text_input) > 3000:
                print(f"\n📄 Text is {len(text_input)} characters - chunking for Dia TTS (max 3000 chars)")
                chunks = self.chunk_text_intelligently(text_input, max_chars=2900)
                print(f"   ✂️  Split into {len(chunks)} chunks")
                
                audio_files = []
                total_generation_time = 0
                
                for i, chunk in enumerate(chunks, 1):
                    print(f"\n🎵 Generating chunk {i}/{len(chunks)} ({len(chunk)} chars)...")
                    
                    chunk_data = {
                        "text": chunk,
                        "high_quality": high_quality,
                        "speed": speed,
                        "return_audio": True
                    }
                    
                    start_time = time.time()
                    chunk_response = requests.post(f"{tts_endpoint}/synthesize", json=chunk_data, timeout=None)
                    chunk_time = time.time() - start_time
                    total_generation_time += chunk_time
                    
                    print(f"   ⏱️ Chunk {i} completed in {chunk_time:.1f} seconds")
                    
                    if chunk_response.status_code == 200:
                        chunk_result = chunk_response.json()
                        chunk_audio = chunk_result.get("audio_base64")
                        
                        if chunk_audio:
                            # Save individual chunk
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            safe_text = "".join(c for c in text_input[:30] if c.isalnum() or c in ' -_').strip()
                            safe_text = "_".join(safe_text.split())
                            chunk_filename = f"chunk_{i:02d}_of_{len(chunks):02d}_{safe_text}_{tts_service}_{timestamp}.wav"
                            chunk_filepath = self.audio_dir / chunk_filename
                            
                            chunk_audio_bytes = base64.b64decode(chunk_audio)
                            with open(chunk_filepath, "wb") as f:
                                f.write(chunk_audio_bytes)
                            
                            audio_files.append(chunk_filepath)
                            print(f"   ✅ Chunk {i} saved: {chunk_filename}")
                        else:
                            print(f"   ❌ No audio data for chunk {i}")
                            return False
                    else:
                        print(f"   ❌ Chunk {i} failed: {chunk_response.status_code}")
                        print(f"   Response: {chunk_response.text}")
                        return False
                
                print(f"\n🎉 All chunks generated in {total_generation_time:.1f} seconds total")
                print(f"📁 Generated {len(audio_files)} audio files:")
                
                total_size = 0
                for audio_file in audio_files:
                    file_size = audio_file.stat().st_size
                    total_size += file_size
                    print(f"   📦 {audio_file.name} ({file_size:,} bytes)")
                
                print(f"\n📊 Total audio size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
                print(f"💡 You can combine these files with audio editing software if needed")
                
                return True
            
            else:
                # Single chunk processing (original logic)
                tts_data = {
                    "text": text_input,
                    "high_quality": high_quality,
                    "speed": speed,
                    "seed": seed,
                    "return_audio": True
                }
                
                # Send TTS request with no timeout limit
                start_time = time.time()
                print(f"\n🎵 Generating speech... (this may take a while for longer text)")
                
                tts_response = requests.post(f"{tts_endpoint}/synthesize", json=tts_data, timeout=None)  # No timeout!
                
                generation_time = time.time() - start_time
                print(f"⏱️ Generation completed in {generation_time:.1f} seconds")
                
                if tts_response.status_code == 200:
                    tts_result = tts_response.json()
                    audio_data = tts_result.get("audio_base64")
                    duration = tts_result.get("duration", "unknown")
                    
                    if audio_data:
                        # Save audio file with descriptive name
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # Create safe filename from first few words
                        safe_text = "".join(c for c in text_input[:30] if c.isalnum() or c in ' -_').strip()
                        safe_text = "_".join(safe_text.split())
                        filename = f"custom_text_{safe_text}_seed_{seed}_{tts_service}_{timestamp}.wav"
                        filepath = self.audio_dir / filename
                        
                        audio_bytes = base64.b64decode(audio_data)
                        with open(filepath, "wb") as f:
                            f.write(audio_bytes)
                        
                        print(f"✅ Audio saved: {filename}")
                        print(f"📊 Audio duration: {duration} seconds")
                        print(f"📁 Saved to: {filepath}")
                        
                        # Optional: Play audio info
                        file_size = len(audio_bytes)
                        print(f"📦 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                        
                        return True
                    else:
                        print(f"❌ No audio data received from TTS")
                        return False
                else:
                    print(f"❌ TTS request failed: {tts_response.status_code}")
                    print(f"   Response: {tts_response.text}")
                    return False
                
        except Exception as e:
            print(f"❌ Typed Text→TTS test failed: {e}")
            traceback.print_exc()
            return False
    
    async def test_full_pipeline(self, stt_service: str, llm_service: str, tts_service: str) -> bool:
        """Test complete STT → LLM → TTS pipeline with real microphone recording"""
        print(f"\n🎯 Testing Full Pipeline: {stt_service} → {llm_service} → {tts_service}")
        print("-" * 70)
        
        try:
            # Option to use microphone or text input
            use_mic = input("Use microphone for STT input? (y/n, default=y): ").strip().lower()
            
            if use_mic in ['', 'y', 'yes']:
                # Record from microphone
                duration = input("Recording duration in seconds (default=5): ").strip()
                try:
                    duration = int(duration) if duration else 5
                    duration = max(1, min(duration, 30))  # Limit between 1-30 seconds
                except ValueError:
                    duration = 5
                
                print(f"\n🎤 Get ready to speak in 3 seconds...")
                time.sleep(3)
                
                audio_file = self.record_microphone_audio(duration)
                if not audio_file:
                    print("❌ Recording failed, falling back to text input")
                    input_text = input("Enter test input: ").strip() or "What's the weather like today?"
                    print(f"📝 Using text input: '{input_text}'")
                else:
                    # Send audio to STT service
                    stt_endpoint = self.available_services[stt_service]["url"]
                    
                    print(f"🔄 Step 1 - STT Processing: {stt_endpoint}")
                    
                    with open(audio_file, 'rb') as f:
                        files = {'audio': f}
                        stt_response = requests.post(f"{stt_endpoint}/transcribe", files=files, timeout=30)
                    
                    if stt_response.status_code == 200:
                        stt_result = stt_response.json()
                        input_text = stt_result.get("text", "").strip()  # Changed from "transcript" to "text"
                        confidence = stt_result.get("confidence", 0.0)
                        print(f"✅ STT Transcript: '{input_text}' (confidence: {confidence:.2f})")
                        
                        if not input_text:
                            print("❌ Empty transcript from STT service!")
                            print("🔄 Attempting direct Whisper transcription...")
                            
                            # Try direct Whisper transcription as fallback
                            try:
                                import sys
                                import os
                                
                                # Add the project root to Python path
                                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                                if project_root not in sys.path:
                                    sys.path.insert(0, project_root)
                                
                                from voicebot_orchestrator.real_whisper_stt import WhisperSTT
                                
                                whisper = WhisperSTT()
                                input_text = await whisper.transcribe_file(audio_file)
                                input_text = input_text.strip() if input_text else ""
                                
                                if input_text:
                                    print(f"✅ Direct Whisper Result: '{input_text}'")
                                else:
                                    print("⚠️  Direct Whisper also returned empty")
                                    
                            except Exception as e:
                                print(f"❌ Direct Whisper failed: {e}")
                                # Try alternative approach with simple whisper import
                                try:
                                    import whisper
                                    print("🔄 Trying fallback Whisper approach...")
                                    
                                    model = whisper.load_model("base")
                                    result = model.transcribe(audio_file)
                                    input_text = result["text"].strip()
                                    
                                    if input_text:
                                        print(f"✅ Fallback Whisper Success: '{input_text}'")
                                    else:
                                        print("⚠️  Even fallback Whisper returned empty - using minimal fallback")
                                        input_text = "What's the weather like today?"
                                        
                                except Exception as e2:
                                    print(f"❌ Fallback Whisper also failed: {e2}")
                                    print("⚠️  Using fallback text")
                                    input_text = "What's the weather like today?"
                    else:
                        print(f"❌ STT step failed: {stt_response.status_code}")
                        print(f"   Response: {stt_response.text}")
                        print("   Using fallback text")
                        input_text = "What's the weather like today?"
            else:
                # Text input mode
                input_text = input("Enter test input (or press Enter for default): ").strip()
                if not input_text:
                    input_text = "What's the weather like today?"
                print(f"🎙️ STT Input (text mode): '{input_text}'")
            
            # Step 2: LLM Processing
            llm_endpoint = self.available_services[llm_service]["url"]
            llm_data = {
                "text": input_text,
                "temperature": 0.7,
                "max_tokens": 150
            }
            
            print(f"🔄 Step 2 - LLM Processing: {llm_endpoint}")
            
            llm_response = requests.post(f"{llm_endpoint}/generate", json=llm_data, timeout=30)
            if llm_response.status_code != 200:
                print(f"❌ LLM step failed: {llm_response.status_code}")
                print(f"   Response: {llm_response.text}")
                return False
                
            llm_result = llm_response.json()
            llm_text = llm_result.get("response", "").strip()
            print(f"✅ LLM Output: '{llm_text[:100]}...'")
            
            # Step 2: TTS Synthesis
            tts_endpoint = self.available_services[tts_service]["url"]
            tts_data = {
                "text": llm_text,
                "voice": "af_bella",  # Use a valid Kokoro voice instead of "default"
                "speed": 1.0,
                "return_audio": True
            }
            
            print(f"🔄 Step 2 - TTS Synthesis: {tts_endpoint}")
            
            tts_response = requests.post(f"{tts_endpoint}/synthesize", json=tts_data, timeout=120)  # Increased for Nari Dia
            if tts_response.status_code == 200:
                tts_result = tts_response.json()
                audio_data = tts_result.get("audio_base64")
                
                if audio_data:
                    # Save audio file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_name = f"{stt_service}_{llm_service}_{tts_service}"
                    filename = f"full_pipeline_{safe_name}_{timestamp}.wav"
                    filepath = self.audio_dir / filename
                    
                    audio_bytes = base64.b64decode(audio_data)
                    with open(filepath, "wb") as f:
                        f.write(audio_bytes)
                    
                    print(f"✅ Complete Pipeline Success! Audio saved: {filename}")
                    return True
                else:
                    print(f"❌ No audio generated")
                    return False
            else:
                print(f"❌ TTS step failed: {tts_response.status_code}")
                print(f"   Response: {tts_response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Full pipeline test failed: {e}")
            traceback.print_exc()
            return False
    
    async def test_direct_dia_tts(self) -> bool:
        """Test direct Dia TTS model with EOS analysis and optimizations"""
        print("\n🎯 Direct Dia TTS Test with EOS Analysis")
        print("-" * 50)
        
        try:
            import sys
            import os
            import torch
            import soundfile as sf
            import time
            import numpy as np
            import random
            from datetime import datetime
            import gc
            
            # Check if we're in the right directory for Dia
            current_dir = os.getcwd()
            tests_dia_path = os.path.join(current_dir, "tests", "dia")
            
            if not os.path.exists(tests_dia_path):
                print(f"❌ Dia model directory not found: {tests_dia_path}")
                print(f"   Current directory: {current_dir}")
                print(f"   Please run from the main Orkestra directory")
                return False
            
            # Apply GPU optimizations
            print("🚀 Applying GPU optimizations...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.cuda.set_per_process_memory_fraction(0.7)
                print("   ✅ GPU optimizations applied")
            else:
                print("   ⚠️ CUDA not available, using CPU")
            
            # Load Dia model
            print("📥 Loading Dia model...")
            original_dir = os.getcwd()
            os.chdir(tests_dia_path)
            
            sys.path.insert(0, tests_dia_path)
            from dia.model import Dia
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model with memory management
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
            print(f"✅ Model loaded on {device}")
            
            # Show GPU status
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
                gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"📊 GPU Memory: {gpu_allocated:.1f}GB allocated, {gpu_reserved:.1f}GB reserved of {gpu_memory:.1f}GB total")
            
            # Get user input for text
            print("\n📝 TEXT INPUT")
            print("=" * 30)
            print("Enter your text (type 'DONE' on a new line to finish):")
            print("Or press Enter for default Alex/Payoff Debt text")
            
            user_lines = []
            first_line = input("> ").strip()
            
            if first_line == "":
                # Use default text
                test_text = """Hello, hello! This is Alex calling with Finally Payoff Debt, your pre-qualification specialist. (laughs) I'm so glad you picked up today. (clears throat) I promise this will be quick, and helpful.
Here's the news: We've rolled out new income-based loan options to help people escape high-interest credit cards and personal loans. (sighs) You know the ones—you pay and pay, but the balance never drops.
Now, listen… (gasps) if you've got steady income, we can help you qualify for a personal loan between ten thousand and one hundred thousand dollars. (coughs) That means instead of juggling multiple bills, you could roll them into one easy payment."""
                print("📖 Using default Alex/Payoff Debt text")
            elif first_line.upper() == "DONE":
                print("❌ No text entered!")
                return False
            else:
                user_lines.append(first_line)
                
                while True:
                    line = input("> ").strip()
                    if line.upper() == "DONE":
                        break
                    user_lines.append(line)
                
                test_text = "\n".join(user_lines)
                print(f"📖 Custom text entered ({len(user_lines)} lines)")
            
            # Format and show text
            formatted_text = f"[S1] {test_text.replace('→', '->')}"  # Unicode fix + Dia format
            print(f"\n📊 Text Analysis:")
            print(f"   Characters: {len(test_text)}")
            print(f"   Lines: {test_text.count(chr(10)) + 1}")
            print(f"   Words (approx): {len(test_text.split())}")
            
            # Get user input for seed
            print("\n🎲 SEED SELECTION")
            print("=" * 30)
            seed_input = input("Enter seed number (or press Enter for random): ").strip()
            
            if seed_input == "":
                user_seed = random.randint(1000, 99999)
                print(f"🎲 Random seed generated: {user_seed}")
            else:
                try:
                    user_seed = int(seed_input)
                    print(f"🎲 Using seed: {user_seed}")
                except ValueError:
                    print("❌ Invalid seed, using random")
                    user_seed = random.randint(1000, 99999)
                    print(f"🎲 Random seed generated: {user_seed}")
            
            # Get token count preference with realistic time estimates
            print("\n🔧 TOKEN CONFIGURATION")
            print("=" * 30)
            print("Available options (with realistic time estimates):")
            print("1. 🚀 Quick test (2048 tokens ~ 10-12 minutes)")
            print("2. ⚖️ Medium test (8192 tokens ~ 40-48 minutes)")
            print("3. 🎯 Long test (16384 tokens ~ 80-96 minutes)")
            print("4. 🎨 Custom amount")
            print("5. 💡 Speed test (1024 tokens ~ 5-6 minutes)")
            
            choice = input("Choose option (1-5, default=1 for speed): ").strip()
            
            if choice == "1" or choice == "":
                token_count = 2048
                est_minutes = 11
                print("🚀 Quick test selected")
            elif choice == "2":
                token_count = 8192
                est_minutes = 44
                print("⚖️ Medium test selected")
                confirm = input(f"⚠️ This will take ~{est_minutes} minutes. Continue? (y/N): ").strip().lower()
                if confirm != 'y':
                    token_count = 2048
                    est_minutes = 11
                    print("🚀 Switched to quick test")
            elif choice == "3":
                token_count = 16384
                est_minutes = 88
                print("🎯 Long test selected")
                confirm = input(f"⚠️ This will take ~{est_minutes} minutes. Continue? (y/N): ").strip().lower()
                if confirm != 'y':
                    token_count = 2048
                    est_minutes = 11
                    print("🚀 Switched to quick test")
            elif choice == "4":
                custom_tokens = input("Enter custom token count (512-65536): ").strip()
                try:
                    token_count = int(custom_tokens)
                    if token_count < 512:
                        print("⚠️ Minimum 512 tokens")
                        token_count = 512
                    elif token_count > 65536:
                        print("⚠️ Maximum 65536 tokens")
                        token_count = 65536
                    
                    # Estimate time: ~295 seconds per 1000 tokens (based on recent tests)
                    est_minutes = (token_count / 1000) * 295 / 60
                    
                    if est_minutes > 30:
                        confirm = input(f"⚠️ Custom test will take ~{est_minutes:.0f} minutes. Continue? (y/N): ").strip().lower()
                        if confirm != 'y':
                            token_count = 2048
                            est_minutes = 11
                            print("🚀 Switched to quick test")
                    
                    print(f"🎨 Custom test: {token_count} tokens (~{est_minutes:.0f} minutes)")
                except ValueError:
                    print("❌ Invalid token count, using quick test")
                    token_count = 2048
                    est_minutes = 11
            elif choice == "5":
                token_count = 1024
                est_minutes = 5
                print("💡 Speed test selected")
            else:
                token_count = 2048
                est_minutes = 11
                print("🚀 Default quick test selected")
            
            print(f"\n⏱️ ESTIMATED TIME: ~{est_minutes} minutes")
            
            # Final confirmation for longer tests
            if est_minutes > 15:
                confirm = input(f"\n⚠️ This will take approximately {est_minutes} minutes. Continue? (y/N): ").strip().lower()
                if confirm != 'y':
                    print("❌ Test cancelled by user")
                    return False
            
            # Set seed and run generation
            print(f"\n🔄 STARTING GENERATION")
            print("=" * 50)
            
            torch.manual_seed(user_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(user_seed)
            
            print(f"📊 Configuration:")
            print(f"   Seed: {user_seed}")
            print(f"   Tokens: {token_count}")
            print(f"   Expected time: ~{est_minutes} minutes")
            
            print(f"\n🔄 Generating audio with GPU optimizations...")
            start_time = time.time()
            
            try:
                with torch.no_grad():  # Save memory
                    audio = model.generate(
                        text=formatted_text,
                        max_tokens=token_count,
                        cfg_scale=3.0,
                        temperature=1.2,
                        top_p=0.95,
                        verbose=True  # Shows EOS analysis
                    )
                
                generation_time = time.time() - start_time
                
                # Clear cache after generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Analyze results
                if hasattr(audio, 'shape') and len(audio.shape) > 0:
                    actual_duration = len(audio) / 44100
                    samples = len(audio)
                    
                    # Format times
                    gen_hours = int(generation_time // 3600)
                    gen_minutes = int((generation_time % 3600) // 60)
                    gen_seconds = int(generation_time % 60)
                    gen_time_str = f"{gen_hours:02d}:{gen_minutes:02d}:{gen_seconds:02d}"
                    
                    print(f"\n📊 RESULTS:")
                    print(f"   🎵 Generated duration: {actual_duration:.2f} seconds")
                    print(f"   🔢 Audio samples: {samples:,}")
                    print(f"   ⏱️ Generation time: {gen_time_str}")
                    print(f"   🎯 Tokens used: {token_count}")
                    print(f"   📈 Performance: {token_count/generation_time:.1f} tokens/sec")
                    print(f"   ⚡ Efficiency: {actual_duration/generation_time:.3f} audio_sec/gen_sec")
                    
                    # Save file in the audio directory
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"direct_dia_seed_{user_seed}_tokens_{token_count}_{timestamp}.wav"
                    
                    # Change back to original directory for saving
                    os.chdir(original_dir)
                    filepath = self.audio_dir / filename
                    sf.write(filepath, audio, 44100)
                    print(f"   💾 Saved: {filename}")
                    
                    # Quality analysis
                    max_amplitude = np.max(np.abs(audio))
                    rms = np.sqrt(np.mean(audio**2))
                    
                    if rms > 0.15:
                        voice_type = "Strong/Confident"
                    elif rms > 0.10:
                        voice_type = "Normal/Balanced"
                    else:
                        voice_type = "Soft/Quiet"
                    
                    print(f"   🎭 Voice character: {voice_type}")
                    print(f"   🔉 Audio quality: Max={max_amplitude:.3f}, RMS={rms:.3f}")
                    
                    # Performance comparison with estimate
                    time_ratio = generation_time / (est_minutes * 60)
                    
                    if time_ratio < 0.8:
                        print(f"   🚀 Faster than expected! ({time_ratio:.2f}x estimated)")
                    elif time_ratio > 1.2:
                        print(f"   🐌 Slower than expected ({time_ratio:.2f}x estimated)")
                    else:
                        print(f"   ✅ Performance as expected ({time_ratio:.2f}x estimated)")
                    
                    # EOS Analysis summary
                    expected_duration = token_count / 1000
                    efficiency_ratio = actual_duration / expected_duration if expected_duration > 0 else 0
                    
                    print(f"\n🔚 EOS ANALYSIS SUMMARY:")
                    print(f"   📊 Token efficiency: {token_count} tokens → {actual_duration:.1f}s audio")
                    print(f"   📈 Audio/token ratio: {efficiency_ratio:.2f}x expected")
                    
                    if efficiency_ratio < 0.5:
                        print(f"   📉 Early EOS detected - audio shorter than expected")
                    elif efficiency_ratio > 1.5:
                        print(f"   📈 Extended generation - audio longer than expected")
                    else:
                        print(f"   ✅ Normal EOS behavior - audio length as expected")
                    
                    print(f"\n✅ Direct Dia TTS test completed successfully!")
                    print(f"   🎵 Audio file: {filename}")
                    print(f"   📊 Duration: {actual_duration:.2f}s")
                    print(f"   🎭 Voice: {voice_type}")
                    
                    return True
                    
                else:
                    print(f"   ❌ No audio generated")
                    return False
            
            except KeyboardInterrupt:
                print(f"\n⚠️ Generation interrupted by user")
                return False
            except Exception as e:
                print(f"   ❌ Generation failed: {e}")
                import traceback
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"❌ Direct Dia TTS test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Restore original directory
            try:
                os.chdir(original_dir)
            except:
                pass
    
    async def test_direct_zonos_tts(self) -> bool:
        """Test direct Zonos TTS service with neural speech synthesis"""
        print("\n🧠 Direct Zonos TTS Test (Neural Speech Synthesis)")
        print("-" * 60)
        
        try:
            import json
            import requests
            import time
            import numpy as np
            import random
            from datetime import datetime
            
            # Check if Zonos TTS service is running
            zonos_url = "http://localhost:8014"
            print(f"🔍 Checking Zonos TTS service at {zonos_url}...")
            
            try:
                response = requests.get(f"{zonos_url}/health", timeout=5)
                if response.status_code != 200:
                    print(f"❌ Zonos TTS service not healthy: {response.status_code}")
                    return False
                
                health_data = response.json()
                print(f"✅ Service status: {health_data.get('status', 'unknown')}")
                print(f"   Version: {health_data.get('version', 'unknown')}")
                
            except requests.RequestException as e:
                print(f"❌ Cannot connect to Zonos TTS service: {e}")
                print(f"   Make sure the service is running on port 8014")
                return False
            
            # Get available voices and models
            try:
                voices_response = requests.get(f"{zonos_url}/voices", timeout=5)
                models_response = requests.get(f"{zonos_url}/models", timeout=5)
                
                if voices_response.status_code == 200:
                    response_data = voices_response.json()
                    # Handle both list format (new) and object format (legacy)
                    if isinstance(response_data, list):
                        voices = response_data
                    else:
                        voices = response_data.get('voices', [])
                    print(f"🎭 Available voices: {', '.join(voices)}")
                else:
                    voices = ['default']
                    print("⚠️ Using default voice")
                
                if models_response.status_code == 200:
                    response_data = models_response.json()
                    # Handle both list format (new) and object format (legacy)
                    if isinstance(response_data, list):
                        models = response_data
                    else:
                        models = response_data.get('models', [])
                    print(f"🤖 Available models: {', '.join(models)}")
                else:
                    models = ['zonos-v1']
                    print("⚠️ Using default model")
                    
            except Exception as e:
                print(f"⚠️ Could not get voice/model info: {e}")
                voices = ['default']
                models = ['zonos-v1']
            
            # Get user input for text
            print("\n📝 TEXT INPUT")
            print("=" * 30)
            print("Enter your text (type 'DONE' on a new line to finish):")
            print("Or press Enter for default neural TTS demo text")
            
            user_lines = []
            first_line = input("> ").strip()
            
            if first_line == "":
                # Use default text showcasing neural capabilities
                test_text = """Welcome to Zonos Neural TTS! This is a demonstration of advanced neural speech synthesis technology. 
I can express emotions, adjust speaking styles, and generate natural-sounding speech with realistic intonation patterns.
Listen to how I can be excited, thoughtful, or conversational depending on the context. The neural networks behind my voice 
create smooth transitions, natural breathing patterns, and expressive delivery that brings text to life."""
                print("📖 Using default neural TTS demo text")
            elif first_line.upper() == "DONE":
                print("❌ No text entered!")
                return False
            else:
                user_lines.append(first_line)
                
                while True:
                    line = input("> ").strip()
                    if line.upper() == "DONE":
                        break
                    user_lines.append(line)
                
                test_text = "\n".join(user_lines)
                print(f"📖 Custom text entered ({len(user_lines)} lines)")
            
            # Show text analysis
            print(f"\n📊 Text Analysis:")
            print(f"   Characters: {len(test_text)}")
            print(f"   Lines: {test_text.count(chr(10)) + 1}")
            print(f"   Words (approx): {len(test_text.split())}")
            
            # Get user preferences
            print("\n🎛️ VOICE & MODEL SELECTION")
            print("=" * 35)
            
            # Voice selection
            if len(voices) > 1:
                print("Available voices:")
                
                # Try to get detailed voice information
                try:
                    detailed_response = requests.get(f"{zonos_url}/voices_detailed", timeout=5)
                    if detailed_response.status_code == 200:
                        detailed_data = detailed_response.json()
                        voice_details = detailed_data.get('voices', {})
                        
                        # Create a mapping of voice names to details
                        voice_info_map = {}
                        for category, voice_dict in voice_details.items():
                            for voice_name, info in voice_dict.items():
                                voice_info_map[voice_name] = {
                                    'gender': info.get('gender', '').title(),
                                    'style': info.get('style', '').replace('_', ' ').title(),
                                    'accent': info.get('accent', '').title(),
                                    'age': info.get('age', '').replace('_', ' ').title()
                                }
                        
                        # Display voices with details
                        for i, voice in enumerate(voices, 1):
                            if voice in voice_info_map:
                                info = voice_info_map[voice]
                                gender = info['gender'] or 'Unknown'
                                style = info['style'] or 'Standard'
                                accent = info['accent'] or 'General'
                                print(f"  {i:2}. {voice:<12} | {gender:<7} | {style:<15} | {accent} accent")
                            else:
                                print(f"  {i:2}. {voice}")
                    else:
                        # Fallback to simple list
                        for i, voice in enumerate(voices, 1):
                            print(f"  {i}. {voice}")
                        
                except Exception as e:
                    # Fallback to simple list
                    for i, voice in enumerate(voices, 1):
                        print(f"  {i}. {voice}")
                
                voice_choice = input(f"Select voice (1-{len(voices)}, default=1): ").strip()
                try:
                    if voice_choice:
                        voice_idx = int(voice_choice) - 1
                        if 0 <= voice_idx < len(voices):
                            selected_voice = voices[voice_idx]
                        else:
                            print(f"⚠️ Invalid choice {voice_choice}, using default")
                            selected_voice = voices[0]
                    else:
                        selected_voice = voices[0]
                except ValueError:
                    print(f"⚠️ Invalid input '{voice_choice}', using default")
                    selected_voice = voices[0]
            else:
                selected_voice = voices[0]
            
            print(f"🎭 Selected voice: {selected_voice}")
            
            # Model selection
            if len(models) > 1:
                print("\nAvailable models:")
                for i, model in enumerate(models, 1):
                    print(f"  {i}. {model}")
                
                model_choice = input(f"Select model (1-{len(models)}, default=1): ").strip()
                try:
                    model_idx = int(model_choice) - 1 if model_choice else 0
                    if 0 <= model_idx < len(models):
                        selected_model = models[model_idx]
                    else:
                        selected_model = models[0]
                except ValueError:
                    selected_model = models[0]
            else:
                selected_model = models[0]
            
            print(f"🤖 Selected model: {selected_model}")
            
            # Emotion selection
            print("\n😊 EMOTION CONTROL")
            print("=" * 25)
            emotions = ['neutral', 'happy', 'excited', 'calm', 'thoughtful', 'conversational']
            print("Available emotions:")
            for i, emotion in enumerate(emotions, 1):
                print(f"  {i}. {emotion}")
            
            emotion_choice = input(f"Select emotion (1-{len(emotions)}, default=1): ").strip()
            try:
                emotion_idx = int(emotion_choice) - 1 if emotion_choice else 0
                if 0 <= emotion_idx < len(emotions):
                    selected_emotion = emotions[emotion_idx]
                else:
                    selected_emotion = 'neutral'
            except ValueError:
                selected_emotion = 'neutral'
            
            print(f"😊 Selected emotion: {selected_emotion}")
            
            # Seed selection
            print("\n🎲 SEED SELECTION")
            print("=" * 25)
            seed_input = input("Enter seed number (or press Enter for random): ").strip()
            
            if seed_input == "":
                user_seed = random.randint(1000, 99999)
                print(f"🎲 Random seed generated: {user_seed}")
            else:
                try:
                    user_seed = int(seed_input)
                    print(f"🎲 Using seed: {user_seed}")
                except ValueError:
                    print("❌ Invalid seed, using random")
                    user_seed = random.randint(1000, 99999)
                    print(f"🎲 Random seed generated: {user_seed}")
            
            # Prepare synthesis request
            synthesis_request = {
                "text": test_text,
                "voice": selected_voice,
                "model": selected_model,
                "emotion": selected_emotion,
                "seed": user_seed,
                "return_audio": True,
                "output_format": "wav"
            }
            
            print(f"\n🔄 STARTING SYNTHESIS")
            print("=" * 35)
            print(f"📊 Configuration:")
            print(f"   Voice: {selected_voice}")
            print(f"   Model: {selected_model}")
            print(f"   Emotion: {selected_emotion}")
            print(f"   Seed: {user_seed}")
            print(f"   Text length: {len(test_text)} characters")
            
            print(f"\n🧠 Synthesizing with neural networks...")
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{zonos_url}/synthesize",
                    json=synthesis_request,
                    timeout=60  # Longer timeout for synthesis
                )
                
                synthesis_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if 'audio_base64' in result and result['audio_base64']:
                        # Decode base64 audio
                        import base64
                        audio_bytes = base64.b64decode(result['audio_base64'])
                        
                        # Save audio file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"zonos_neural_seed_{user_seed}_{selected_voice}_{selected_emotion}_{timestamp}.wav"
                        filepath = self.audio_dir / filename
                        
                        with open(filepath, 'wb') as f:
                            f.write(audio_bytes)
                        
                        # Get audio info from response
                        audio_duration = result.get('duration', 0)
                        sample_rate = result.get('sample_rate', 22050)
                        
                        # Format synthesis time
                        syn_minutes = int(synthesis_time // 60)
                        syn_seconds = int(synthesis_time % 60)
                        syn_time_str = f"{syn_minutes:02d}:{syn_seconds:02d}"
                        
                        print(f"\n📊 SYNTHESIS RESULTS:")
                        print(f"   🎵 Generated duration: {audio_duration:.2f} seconds")
                        print(f"   🔊 Sample rate: {sample_rate:,} Hz")
                        print(f"   ⏱️ Synthesis time: {syn_time_str}")
                        print(f"   🚀 Speed ratio: {audio_duration/synthesis_time:.2f}x real-time")
                        print(f"   💾 Saved: {filename}")
                        
                        # Quality analysis if available in response
                        if 'quality_metrics' in result:
                            metrics = result['quality_metrics']
                            print(f"\n🔍 QUALITY METRICS:")
                            for metric, value in metrics.items():
                                print(f"   {metric}: {value}")
                        
                        # Neural analysis
                        if 'neural_info' in result:
                            neural_info = result['neural_info']
                            print(f"\n🧠 NEURAL SYNTHESIS INFO:")
                            print(f"   Model layers: {neural_info.get('layers', 'N/A')}")
                            print(f"   Processing steps: {neural_info.get('steps', 'N/A')}")
                            print(f"   Emotion strength: {neural_info.get('emotion_strength', 'N/A')}")
                        
                        print(f"\n✅ Zonos Neural TTS test completed successfully!")
                        print(f"   🎵 Audio file: {filename}")
                        print(f"   📊 Duration: {audio_duration:.2f}s")
                        print(f"   🎭 Voice/Emotion: {selected_voice}/{selected_emotion}")
                        print(f"   🧠 Model: {selected_model}")
                        
                        return True
                        
                    else:
                        print(f"   ❌ No audio data in response")
                        return False
                        
                else:
                    print(f"   ❌ Synthesis failed: {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                    except:
                        print(f"   Error: {response.text}")
                    return False
                    
            except requests.Timeout:
                print(f"   ❌ Synthesis timed out (>60 seconds)")
                return False
            except Exception as e:
                print(f"   ❌ Synthesis failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Zonos TTS test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_direct_tortoise_tts(self) -> bool:
        """Test direct Tortoise TTS service with ultra high-quality voice synthesis"""
        print("\n🐢 Direct Tortoise TTS Test (Ultra High-Quality Voice Synthesis)")
        print("-" * 70)
        
        try:
            import json
            import requests
            import time
            import base64
            import soundfile as sf
            import io
            import numpy as np
            from datetime import datetime
            
            # Check if Tortoise TTS service is running
            tortoise_url = "http://localhost:8015"
            print(f"🔍 Checking Tortoise TTS service at {tortoise_url}...")
            
            try:
                response = requests.get(f"{tortoise_url}/health", timeout=5)
                if response.status_code != 200:
                    print(f"❌ Tortoise TTS service not healthy: {response.status_code}")
                    return False
                
                health_data = response.json()
                print(f"✅ Service status: {health_data.get('status', 'unknown')}")
                engine_info = health_data.get('engine', 'Enhanced Placeholder')
                print(f"   Engine: {engine_info}")
                
            except requests.RequestException as e:
                print(f"❌ Cannot connect to Tortoise TTS service: {e}")
                print(f"   Make sure the service is running on port 8015")
                return False
            
            # Get available voices and presets
            try:
                voices_response = requests.get(f"{tortoise_url}/voices", timeout=5)
                presets_response = requests.get(f"{tortoise_url}/presets", timeout=5)
                
                if voices_response.status_code == 200:
                    voices = voices_response.json()  # Direct list response
                    if not isinstance(voices, list):
                        voices = ['angie', 'denise', 'freeman']  # Fallback
                    print(f"🎭 Available voices: {', '.join(voices[:10])}{'...' if len(voices) > 10 else ''}")
                    print(f"   Total voices: {len(voices)}")
                else:
                    voices = ['angie', 'denise', 'freeman']
                    print("⚠️ Using default voices")
                
                if presets_response.status_code == 200:
                    response_data = presets_response.json()
                    presets = response_data.get('presets', ['fast', 'standard', 'high_quality'])
                    print(f"🔧 Available quality presets: {', '.join(presets)}")
                else:
                    presets = ['fast', 'standard', 'high_quality']
                    print("⚠️ Using default presets")
                    
            except Exception as e:
                print(f"⚠️ Could not get voice/preset info: {e}")
                voices = ['angie', 'denise', 'freeman', 'pat', 'william']
                presets = ['fast', 'standard', 'high_quality']
            
            # Voice selection - simplified two-column display
            print("\n🎭 VOICE SELECTION")
            print("=" * 60)
            
            # Display voices in two columns, simple format
            print(f"{'COLUMN 1':<28} {'COLUMN 2':<28}")
            print("-" * 56)
            
            all_display_voices = voices
            mid_point = (len(voices) + 1) // 2
            
            for i in range(mid_point):
                # Left column
                if i < len(voices):
                    left_voice = voices[i]
                    left_text = f"{i+1:2d}. {left_voice}"
                else:
                    left_text = ""
                
                # Right column
                right_index = i + mid_point
                if right_index < len(voices):
                    right_voice = voices[right_index]
                    right_text = f"{right_index+1:2d}. {right_voice}"
                else:
                    right_text = ""
                
                print(f"{left_text:<28} {right_text:<28}")
            
            print(f"\n   0. Use default voice (angie)")
            print(f"\nTotal voices: {len(voices)}")
            
            voice_choice = input(f"\nSelect voice (0-{len(voices)}): ").strip()
            
            if voice_choice == "0" or voice_choice == "":
                selected_voice = "angie"
                print(f"🎭 Using default voice: {selected_voice}")
            elif voice_choice.isdigit() and 1 <= int(voice_choice) <= len(voices):
                selected_voice = voices[int(voice_choice) - 1]
                print(f"🎭 Selected voice: {selected_voice}")
            else:
                print("❌ Invalid choice, using default voice: angie")
                selected_voice = "angie"
            
            # Quality preset selection
            print("\n🔧 QUALITY PRESET")
            print("=" * 30)
            for i, preset in enumerate(presets, 1):
                descriptions = {
                    'fast': 'Quick generation (~3s), good quality',
                    'standard': 'Balanced quality/speed (~5s)', 
                    'high_quality': 'Best quality (~8s), slower'
                }
                desc = descriptions.get(preset, 'Custom preset')
                print(f"  {i}. {preset} - {desc}")
            
            preset_choice = input(f"\nSelect quality preset (1-{len(presets)}) or Enter for standard: ").strip()
            
            if preset_choice == "":
                selected_preset = "standard"
                print(f"🔧 Using default preset: {selected_preset}")
            elif preset_choice.isdigit() and 1 <= int(preset_choice) <= len(presets):
                selected_preset = presets[int(preset_choice) - 1]
                print(f"🔧 Selected preset: {selected_preset}")
            else:
                print("❌ Invalid choice, using standard preset")
                selected_preset = "standard"
            
            # Get user input for text
            print("\n📝 TEXT INPUT")
            print("=" * 30)
            print("Enter your text (type 'DONE' on a new line to finish):")
            print("Or leave empty and type 'DONE' for default ultra-quality demo text")
            print()
            
            user_lines = []
            print("Start typing (type 'DONE' when finished):")
            
            while True:
                line = input("> ").strip()
                if line.upper() == "DONE":
                    break
                elif line == "" and len(user_lines) == 0:
                    # If first line is empty and user types DONE next, they want default text
                    continue
                else:
                    user_lines.append(line)
            
            if len(user_lines) == 0:
                # Use default text showcasing Tortoise capabilities
                test_text = """Welcome to Tortoise TTS, the pinnacle of neural voice synthesis technology. 
This system represents the cutting edge of artificial intelligence voice generation, capable of producing speech 
that is virtually indistinguishable from human speakers. With advanced prosodic modeling and emotional expression, 
I can convey subtle nuances in tone, rhythm, and emphasis that bring text to life in remarkable ways."""
                print("📖 Using default ultra-quality demo text")
            else:
                test_text = "\n".join(user_lines)
                print(f"📖 Custom text entered ({len(user_lines)} lines)")
            
            # Show synthesis parameters
            print(f"\n📊 Synthesis Parameters:")
            print(f"   Voice: {selected_voice}")
            print(f"   Quality: {selected_preset}")
            print(f"   Text length: {len(test_text)} characters")
            print(f"   Estimated words: {len(test_text.split())}")
            
            # Confirmation before synthesis
            print(f"\n❓ Proceed with synthesis?")
            confirmation = input("   Type 'yes' or 'y' to continue, anything else to cancel: ").strip().lower()
            
            if confirmation not in ['yes', 'y']:
                print("❌ Synthesis cancelled by user")
                return False
            
            # Prepare synthesis request
            synthesis_data = {
                "text": test_text,
                "voice": selected_voice,
                "preset": selected_preset,
                "return_audio": True
            }
            
            print(f"\n🚀 Starting synthesis with Tortoise TTS...")
            print("   This may take longer than other TTS engines due to ultra-high quality processing...")
            
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{tortoise_url}/synthesize",
                    json=synthesis_data,
                    timeout=120  # Longer timeout for Tortoise
                )
                
                synthesis_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print(f"✅ Synthesis completed in {synthesis_time:.2f} seconds")
                    print(f"   Quality: {result.get('quality', selected_preset)}")
                    print(f"   Voice used: {result.get('voice', selected_voice)}")
                    
                    # Handle audio data
                    if result.get("audio_base64"):
                        print("🎵 Audio generated successfully!")
                        
                        # Decode and save audio
                        try:
                            audio_data = base64.b64decode(result["audio_base64"])
                            
                            # Generate filename with timestamp and voice
                            timestamp = datetime.now().strftime("%H%M%S")
                            filename = f"tortoise_test_{selected_voice}_{selected_preset}_{timestamp}.wav"
                            
                            with open(filename, "wb") as f:
                                f.write(audio_data)
                            
                            print(f"💾 Audio saved: {filename}")
                            
                            # Audio analysis
                            try:
                                # Read audio for analysis
                                audio_io = io.BytesIO(audio_data)
                                audio_array, sample_rate = sf.read(audio_io)
                                
                                duration = len(audio_array) / sample_rate
                                rms = np.sqrt(np.mean(audio_array**2))
                                
                                print(f"📊 Audio Analysis:")
                                print(f"   Duration: {duration:.2f} seconds")
                                print(f"   Sample Rate: {sample_rate} Hz")
                                print(f"   RMS Level: {rms:.4f}")
                                print(f"   Real-time factor: {synthesis_time/duration:.2f}x")
                                
                            except Exception as e:
                                print(f"⚠️ Audio analysis failed: {e}")
                            
                        except Exception as e:
                            print(f"❌ Failed to save audio: {e}")
                    
                    else:
                        print("⚠️ No audio data in response")
                        
                    # Show performance metrics
                    print(f"\n⚡ Performance Metrics:")
                    chars_per_sec = len(test_text) / synthesis_time
                    print(f"   Characters/second: {chars_per_sec:.1f}")
                    print(f"   Words/minute: {(len(test_text.split()) * 60) / synthesis_time:.1f}")
                    
                    return True
                
                else:
                    print(f"❌ Synthesis failed: HTTP {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                    except:
                        print(f"   Response: {response.text[:200]}")
                    return False
                    
            except requests.exceptions.Timeout:
                synthesis_time = time.time() - start_time
                print(f"❌ Synthesis timed out after {synthesis_time:.1f} seconds")
                print("   Tortoise TTS requires more time for ultra-high quality synthesis")
                return False
            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Tortoise TTS test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_interactive_tests(self):
        """Run interactive test menu without automatic service checks"""
        print("🎯 Interactive Pipeline Tester")
        
        while True:
            print("\n" + "="*60)
            print("📋 TEST MENU")
            print("="*60)
            
            print("\n🧪 PIPELINE TESTS:")
            print("  1. Test STT → LLM (select specific services)")
            print("  2. Test LLM → TTS (select specific services)")
            print("  3. Test Full Pipeline STT → LLM → TTS")
            print("  4. Test Typed Text → TTS (custom text input)")
            
            print("\n🔍 SERVICE TESTS:")
            print("  5. Check service availability")
            print("  6. Test Direct Dia TTS (with EOS analysis)")
            print("  7. Test Direct Zonos TTS (neural speech synthesis)")
            print("  8. Test Direct Tortoise TTS (ultra high-quality voice synthesis)")
            
            print("\n  0. Exit")
            
            try:
                choice = input("\nSelect test type (0-8): ").strip()
                
                if choice == "0":
                    print("👋 Goodbye!")
                    break
                
                elif choice == "1":
                    print("\n🎙️ STT → LLM Test")
                    # Check services first for this test
                    self.detect_available_services()
                    stt_service = self.select_services_by_type("stt")
                    if not stt_service:
                        continue
                    llm_service = self.select_services_by_type("llm")
                    if not llm_service:
                        continue
                    
                    success = await self.test_stt_to_llm(stt_service, llm_service)
                    result = "✅ SUCCESS" if success else "❌ FAILED"
                    print(f"\n📊 Result: {result}")
                
                elif choice == "2":
                    print("\n🧠 LLM → TTS Test")
                    # Check services first for this test
                    self.detect_available_services()
                    llm_service = self.select_services_by_type("llm")
                    if not llm_service:
                        continue
                    tts_service = self.select_services_by_type("tts")
                    if not tts_service:
                        continue
                    
                    success = await self.test_llm_to_tts(llm_service, tts_service)
                    result = "✅ SUCCESS" if success else "❌ FAILED"
                    print(f"\n📊 Result: {result}")
                
                elif choice == "3":
                    print("\n🎯 Full Pipeline Test")
                    # Check services first for this test
                    self.detect_available_services()
                    stt_service = self.select_services_by_type("stt")
                    if not stt_service:
                        continue
                    llm_service = self.select_services_by_type("llm")
                    if not llm_service:
                        continue
                    tts_service = self.select_services_by_type("tts")
                    if not tts_service:
                        continue
                    
                    success = await self.test_full_pipeline(stt_service, llm_service, tts_service)
                    result = "✅ SUCCESS" if success else "❌ FAILED"
                    print(f"\n📊 Result: {result}")
                
                elif choice == "4":
                    print("\n🎤 Typed Text → TTS Test")
                    # Direct to TTS service selection without any service checks
                    tts_service = self.select_services_by_type("tts")
                    if not tts_service:
                        print("❌ No TTS services available. Use option 5 to check service status.")
                        continue
                    
                    success = await self.test_typed_text_to_tts(tts_service)
                    result = "✅ SUCCESS" if success else "❌ FAILED"
                    print(f"\n📊 Result: {result}")
                
                elif choice == "5":
                    self.detect_available_services()
                    print("\n📋 Available Services:")
                    if not self.available_services:
                        print("  ❌ No services available")
                    else:
                        for service_name, config in self.available_services.items():
                            service_type = config["type"].upper()
                            url = config["url"]
                            print(f"  ✅ {service_name:<20} ({service_type:<3}) - {url}")
                
                elif choice == "6":
                    print("\n🎯 Direct Dia TTS Test with EOS Analysis")
                    success = await self.test_direct_dia_tts()
                    result = "✅ SUCCESS" if success else "❌ FAILED"
                    print(f"\n📊 Result: {result}")
                
                elif choice == "7":
                    success = await self.test_direct_zonos_tts()
                    result = "✅ SUCCESS" if success else "❌ FAILED"
                    print(f"\n📊 Result: {result}")
                
                elif choice == "8":
                    success = await self.test_direct_tortoise_tts()
                    result = "✅ SUCCESS" if success else "❌ FAILED"
                    print(f"\n📊 Result: {result}")
                
                else:
                    print("❌ Invalid choice. Please enter 0-8.")
                
                if choice in ["1", "2", "3", "4", "6", "7", "8"]:
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                traceback.print_exc()
                input("\nPress Enter to continue...")

async def main():
    """Main function"""
    tester = InteractivePipelineTester()
    await tester.run_interactive_tests()

if __name__ == "__main__":
    asyncio.run(main())
