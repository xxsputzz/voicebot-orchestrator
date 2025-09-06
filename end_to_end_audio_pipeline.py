#!/usr/bin/env python3
"""
End-to-End Audio Pipeline Demo
Demonstrates real audio→text→response→audio streaming using converted WebSocket services

This connects the three converted services:
1. WebSocket Whisper STT (from aws_microservices/stt_whisper_service.py)
2. WebSocket GPT LLM (from aws_microservices/llm_gpt_service.py)  
3. WebSocket Tortoise TTS (from aws_microservices/tts_tortoise_service.py)

Through the WebSocket orchestrator for real-time streaming communication.
"""

import asyncio
import json
import logging
import time
import base64
import tempfile
import os
import sys
from typing import Dict, Any, Optional
import websockets
import uuid
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class EndToEndAudioPipeline:
    """End-to-end audio pipeline using converted WebSocket services"""
    
    def __init__(self):
        self.client_websocket = None
        self.orchestrator_url = "ws://localhost:9000"  # Client port
        self.session_id = str(uuid.uuid4())
        self.pipeline_active = False
        
        # Pipeline state
        self.stt_result = None
        self.llm_response = None
        self.tts_audio = None
        
        # Performance tracking
        self.pipeline_start_time = None
        self.stt_time = None
        self.llm_time = None
        self.tts_time = None
        
    def safe_print(self, text: str):
        """Safe print function that handles Unicode characters for Windows console."""
        try:
            print(text)
        except UnicodeEncodeError:
            safe_text = text.encode('ascii', 'replace').decode('ascii')
            print(safe_text)
    
    async def create_sample_audio(self) -> str:
        """Create sample audio file for testing"""
        try:
            # Create a simple WAV file with a sample phrase
            import wave
            import struct
            import math
            
            # Audio parameters
            sample_rate = 22050
            duration = 3.0  # 3 seconds
            frequency = 440.0  # A4 note
            amplitude = 0.3
            
            # Generate sine wave samples
            samples = []
            for i in range(int(sample_rate * duration)):
                t = i / sample_rate
                sample = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
                samples.append(sample)
            
            # Create temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            
            with wave.open(temp_file.name, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Write samples
                for sample in samples:
                    wav_file.writeframes(struct.pack('<h', sample))
            
            temp_file.close()
            
            # Read as base64
            with open(temp_file.name, 'rb') as f:
                audio_data = f.read()
            
            # Cleanup
            os.unlink(temp_file.name)
            
            # Return base64 encoded audio
            return base64.b64encode(audio_data).decode('utf-8')
            
        except Exception as e:
            self.safe_print(f"❌ Error creating sample audio: {e}")
            return None
    
    async def connect_to_orchestrator(self) -> bool:
        """Connect to WebSocket orchestrator"""
        try:
            self.safe_print(f"📡 Connecting to orchestrator: {self.orchestrator_url}")
            
            self.client_websocket = await websockets.connect(
                self.orchestrator_url,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.safe_print("✅ Connected to orchestrator")
            return True
            
        except Exception as e:
            self.safe_print(f"❌ Failed to connect to orchestrator: {e}")
            return False
    
    async def handle_message(self, message_data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        try:
            message_type = message_data.get("type")
            data = message_data.get("data", {})
            session_id = message_data.get("session_id")
            
            if session_id != self.session_id:
                return  # Not our session
            
            if message_type == "transcription_result":
                await self.handle_stt_result(data)
            elif message_type == "transcription_final":
                await self.handle_stt_result(data)
            elif message_type == "llm_stream_complete":
                await self.handle_llm_complete(data)
            elif message_type == "llm_response":
                await self.handle_llm_complete(data)
            elif message_type == "tts_stream_complete":
                await self.handle_tts_complete(data)
            elif message_type == "tts_audio_complete":
                await self.handle_tts_complete(data)
            elif message_type == "audio_chunk":
                await self.handle_audio_chunk(data)
            elif message_type == "error":
                self.safe_print(f"❌ Pipeline error: {data.get('error', 'Unknown error')}")
            else:
                logging.debug(f"Received message type: {message_type}")
                
        except Exception as e:
            self.safe_print(f"❌ Error handling message: {e}")
    
    async def handle_stt_result(self, data: Dict[str, Any]):
        """Handle STT transcription result"""
        self.stt_result = data.get("text", "").strip()
        self.stt_time = time.time()
        processing_time = data.get("processing_time_seconds", 0)
        
        self.safe_print(f"🎙️  STT Result ({processing_time:.2f}s): '{self.stt_result}'")
        
        if self.stt_result:
            # Send to LLM for processing
            await self.send_to_llm(self.stt_result)
    
    async def handle_llm_complete(self, data: Dict[str, Any]):
        """Handle LLM response completion"""
        self.llm_response = data.get("text", "").strip() or data.get("full_text", "").strip()
        self.llm_time = time.time()
        processing_time = data.get("processing_time_seconds", 0)
        
        self.safe_print(f"🤖 LLM Response ({processing_time:.2f}s): '{self.llm_response}'")
        
        if self.llm_response:
            # Send to TTS for synthesis
            await self.send_to_tts(self.llm_response)
    
    async def handle_tts_complete(self, data: Dict[str, Any]):
        """Handle TTS synthesis completion"""
        self.tts_time = time.time()
        processing_time = data.get("processing_time_seconds", 0)
        audio_length = data.get("audio_length_bytes", 0)
        
        self.safe_print(f"🔊 TTS Complete ({processing_time:.2f}s): {audio_length} bytes audio generated")
        
        # Pipeline complete
        await self.complete_pipeline()
    
    async def handle_audio_chunk(self, data: Dict[str, Any]):
        """Handle audio chunk from TTS streaming"""
        chunk_index = data.get("chunk_index", 0)
        total_chunks = data.get("total_chunks", 1)
        is_final = data.get("is_final", False)
        
        if is_final:
            self.tts_time = time.time()
            self.safe_print(f"🔊 TTS Streaming Complete: {total_chunks} audio chunks received")
            await self.complete_pipeline()
    
    async def send_to_stt(self, audio_base64: str):
        """Send audio to STT service"""
        try:
            message = {
                "type": "audio_file",
                "session_id": self.session_id,
                "target_service": "stt_whisper_ws",
                "data": {
                    "audio_data": audio_base64,
                    "language": "auto",
                    "task": "transcribe"
                }
            }
            
            await self.client_websocket.send(json.dumps(message))
            self.safe_print("📤 Sent audio to STT service")
            
        except Exception as e:
            self.safe_print(f"❌ Error sending to STT: {e}")
    
    async def send_to_llm(self, text: str):
        """Send text to LLM service"""
        try:
            message = {
                "type": "text_input",
                "session_id": self.session_id,
                "target_service": "llm_gpt_ws",
                "data": {
                    "text": text,
                    "stream_tokens": True,
                    "use_cache": True,
                    "domain_context": "conversational_ai",
                    "max_tokens": 256
                }
            }
            
            await self.client_websocket.send(json.dumps(message))
            self.safe_print("📤 Sent text to LLM service")
            
        except Exception as e:
            self.safe_print(f"❌ Error sending to LLM: {e}")
    
    async def send_to_tts(self, text: str):
        """Send text to TTS service"""
        try:
            message = {
                "type": "text_input",
                "session_id": self.session_id,
                "target_service": "tts_tortoise_ws",
                "data": {
                    "text": text,
                    "voice": "rainbow",
                    "stream_audio": True,
                    "output_format": "base64"
                }
            }
            
            await self.client_websocket.send(json.dumps(message))
            self.safe_print("📤 Sent text to TTS service")
            
        except Exception as e:
            self.safe_print(f"❌ Error sending to TTS: {e}")
    
    async def complete_pipeline(self):
        """Complete the pipeline and show results"""
        if not all([self.stt_result, self.llm_response, self.tts_time]):
            return  # Pipeline not complete yet
        
        total_time = self.tts_time - self.pipeline_start_time
        
        self.safe_print("\n" + "="*70)
        self.safe_print("🎯 END-TO-END PIPELINE COMPLETE!")
        self.safe_print("="*70)
        self.safe_print(f"📊 Total Processing Time: {total_time:.2f}s")
        self.safe_print()
        self.safe_print("📋 Pipeline Results:")
        self.safe_print(f"   🎙️  STT: '{self.stt_result}'")
        self.safe_print(f"   🤖 LLM: '{self.llm_response}'")
        self.safe_print(f"   🔊 TTS: Audio generated successfully")
        self.safe_print()
        
        # Detailed timing breakdown
        if self.stt_time:
            stt_duration = self.stt_time - self.pipeline_start_time
            self.safe_print(f"⏱️  STT Processing: {stt_duration:.2f}s")
        
        if self.llm_time and self.stt_time:
            llm_duration = self.llm_time - self.stt_time
            self.safe_print(f"⏱️  LLM Processing: {llm_duration:.2f}s")
        
        if self.tts_time and self.llm_time:
            tts_duration = self.tts_time - self.llm_time
            self.safe_print(f"⏱️  TTS Processing: {tts_duration:.2f}s")
        
        self.safe_print("="*70)
        
        self.pipeline_active = False
    
    async def run_pipeline(self, audio_base64: str = None):
        """Run the complete audio pipeline"""
        self.safe_print("🚀 Starting End-to-End Audio Pipeline...")
        self.safe_print("🔄 Testing converted WebSocket services integration")
        self.safe_print("-" * 70)
        
        # Connect to orchestrator
        if not await self.connect_to_orchestrator():
            return False
        
        # Create sample audio if not provided
        if not audio_base64:
            self.safe_print("🎵 Creating sample audio for testing...")
            audio_base64 = await self.create_sample_audio()
            if not audio_base64:
                self.safe_print("❌ Failed to create sample audio")
                return False
            self.safe_print(f"✅ Sample audio created: {len(audio_base64)} bytes (base64)")
        
        # Reset pipeline state
        self.stt_result = None
        self.llm_response = None
        self.tts_audio = None
        self.pipeline_start_time = time.time()
        self.pipeline_active = True
        
        # Start the pipeline by sending audio to STT
        self.safe_print("\n🎬 Starting Audio Pipeline:")
        await self.send_to_stt(audio_base64)
        
        # Listen for responses
        try:
            timeout = 60  # 60 second timeout
            start_time = time.time()
            
            async for message in self.client_websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(data)
                    
                    # Check if pipeline is complete
                    if not self.pipeline_active:
                        break
                    
                    # Check timeout
                    if time.time() - start_time > timeout:
                        self.safe_print(f"⏰ Pipeline timeout after {timeout}s")
                        break
                        
                except json.JSONDecodeError:
                    self.safe_print(f"❌ Invalid JSON received: {message}")
                except Exception as e:
                    self.safe_print(f"❌ Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.safe_print("📡 WebSocket connection closed")
        except Exception as e:
            self.safe_print(f"❌ Pipeline error: {e}")
        finally:
            if self.client_websocket:
                await self.client_websocket.close()
        
        return self.pipeline_active is False  # Success if pipeline completed
    
    async def test_individual_services(self):
        """Test each service individually"""
        self.safe_print("🧪 Testing Individual Services...")
        self.safe_print("-" * 50)
        
        # Connect to orchestrator
        if not await self.connect_to_orchestrator():
            return False
        
        try:
            # Test STT service
            self.safe_print("1️⃣ Testing STT Service...")
            audio_base64 = await self.create_sample_audio()
            if audio_base64:
                await self.send_to_stt(audio_base64)
                await asyncio.sleep(3)  # Wait for response
                self.safe_print(f"   STT Test: {'✅ PASS' if self.stt_result else '❌ FAIL'}")
            
            # Test LLM service
            self.safe_print("2️⃣ Testing LLM Service...")
            await self.send_to_llm("Hello, can you hear me?")
            await asyncio.sleep(3)  # Wait for response
            self.safe_print(f"   LLM Test: {'✅ PASS' if self.llm_response else '❌ FAIL'}")
            
            # Test TTS service
            self.safe_print("3️⃣ Testing TTS Service...")
            await self.send_to_tts("This is a test of the text to speech system.")
            await asyncio.sleep(5)  # Wait for response
            self.safe_print(f"   TTS Test: {'✅ PASS' if self.tts_time else '❌ FAIL'}")
            
            self.safe_print("-" * 50)
            self.safe_print("🧪 Individual service testing complete")
            
        except Exception as e:
            self.safe_print(f"❌ Service testing error: {e}")
        finally:
            if self.client_websocket:
                await self.client_websocket.close()
        
        return True

async def check_orchestrator_health() -> bool:
    """Check if orchestrator is running"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8080/health", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "healthy"
    except Exception:
        pass
    
    return False

def safe_print(text):
    """Safe print function that handles Unicode characters for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)

async def main():
    """Main entry point"""
    safe_print("🔊 End-to-End Audio Pipeline Demo")
    safe_print("🔄 Using converted WebSocket services")
    safe_print("="*70)
    
    # Check if orchestrator is running
    if not await check_orchestrator_health():
        safe_print("❌ WebSocket orchestrator is not running!")
        safe_print("💡 Please start the orchestrator first:")
        safe_print("   python launch_converted_services.py")
        safe_print("   Select option 1 to start all services")
        return
    
    safe_print("✅ Orchestrator is running")
    
    pipeline = EndToEndAudioPipeline()
    
    # Ask user what to do
    safe_print("\nSelect test mode:")
    safe_print("1. Full End-to-End Pipeline Test")
    safe_print("2. Individual Service Tests")
    safe_print("3. Both")
    
    try:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice in ["2", "3"]:
            safe_print("\n" + "="*70)
            await pipeline.test_individual_services()
        
        if choice in ["1", "3"]:
            safe_print("\n" + "="*70)
            success = await pipeline.run_pipeline()
            
            if success:
                safe_print("\n🎉 End-to-End Pipeline Test: ✅ SUCCESS")
            else:
                safe_print("\n❌ End-to-End Pipeline Test: FAILED")
        
    except KeyboardInterrupt:
        safe_print("\n👋 Test interrupted by user")
    except Exception as e:
        safe_print(f"\n❌ Test error: {e}")
    
    safe_print("\n✅ Audio Pipeline Demo completed")

if __name__ == "__main__":
    # Add required imports
    import wave
    import struct
    import math
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        safe_print("\n👋 Demo interrupted")
    except Exception as e:
        safe_print(f"❌ Demo error: {e}")
        sys.exit(1)
