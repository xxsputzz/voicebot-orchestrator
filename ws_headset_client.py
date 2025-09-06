#!/usr/bin/env python3
"""
Simple WebSocket Headset Client for Testing

A basic client that simulates a headset connecting to the orchestrator
for testing the streaming pipeline.
"""

import asyncio
import json
import logging
import math
import time
import uuid
import wave
import struct
from typing impo                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, input, "\n📱 Select option (0-3): "
                    )Optional, Dict
import websockets
import threading
import queue

# Try to import audio libraries
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

class HeadsetClient:
    """Simple headset client for testing WebSocket streaming"""
    
    def __init__(self, orchestrator_host: str = "localhost", orchestrator_port: int = 9000):
        self.orchestrator_host = orchestrator_host
        self.orchestrator_port = orchestrator_port
        self.session_id = str(uuid.uuid4())
        self.websocket = None
        self.connected = False
        self.audio_queue = queue.Queue()
        self.logger = logging.getLogger(__name__)
        
        # Audio configuration
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.format = pyaudio.paInt16 if AUDIO_AVAILABLE else None
        self.audio = None
        self.audio_input_stream = None
        self.audio_output_stream = None
        
        # Test mode (for when audio hardware isn't available)
        self.test_mode = not AUDIO_AVAILABLE
        
    async def connect(self):
        """Connect to the WebSocket orchestrator"""
        uri = f"ws://{self.orchestrator_host}:{self.orchestrator_port}/client"
        
        try:
            print(f"🔄 Connecting to {uri}...")
            self.websocket = await websockets.connect(uri)
            self.connected = True
            print("✅ Connected successfully!")
            
            # Send initial session start message
            await self.send_session_start()
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.logger.error(f"Failed to connect to orchestrator: {e}")
            raise
    
    async def send_session_start(self):
        """Send session start message"""
        message = {
            "type": "session_start",
            "session_id": self.session_id,
            "timestamp": time.time(),
            "data": {
                "client_type": "headset",
                "audio_format": {
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "bit_depth": 16
                }
            },
            "metadata": {
                "client_version": "1.0.0",
                "test_mode": self.test_mode
            }
        }
        
        await self.websocket.send(json.dumps(message))
        self.logger.info(f"Started session: {self.session_id}")
    
    def init_audio(self):
        """Initialize audio streams"""
        if not AUDIO_AVAILABLE:
            self.logger.warning("PyAudio not available, running in test mode")
            return False
        
        try:
            self.audio = pyaudio.PyAudio()
            
            # Input stream for microphone
            self.audio_input_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_input_callback
            )
            
            # Output stream for speakers
            self.audio_output_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.logger.info("Audio streams initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio: {e}")
            return False
    
    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        """Audio input callback for microphone data"""
        # Add audio data to queue for processing
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    async def start_audio_streaming(self):
        """Start streaming audio to orchestrator"""
        if not self.audio_input_stream:
            # Test mode - send fake audio data
            await self.send_test_audio()
            return
        
        # Start audio input stream
        self.audio_input_stream.start_stream()
        
        # Process audio data from queue
        while self.connected:
            try:
                # Get audio data with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                await self.send_audio_chunk(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing audio: {e}")
                break
    
    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to orchestrator"""
        if not self.connected or not self.websocket:
            return
        
        message = {
            "type": "audio_chunk",
            "session_id": self.session_id,
            "timestamp": time.time(),
            "data": {
                "audio_data": audio_data.hex(),  # Convert bytes to hex string
                "format": {
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "bit_depth": 16
                }
            },
            "metadata": {
                "chunk_size": len(audio_data)
            }
        }
        
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            self.logger.error(f"Failed to send audio chunk: {e}")
            self.connected = False
    
    async def send_test_audio(self):
        """Send test audio data for testing without hardware"""
        self.logger.info("Sending test audio data...")
        
        # Generate simple test audio (sine wave)
        duration = 3.0  # 3 seconds
        frequency = 440  # A4 note
        
        for i in range(int(duration * self.sample_rate / self.chunk_size)):
            # Generate sine wave chunk
            chunk_data = []
            for j in range(self.chunk_size):
                sample_idx = i * self.chunk_size + j
                sample = int(16000 * 
                           math.sin(2 * math.pi * frequency * sample_idx / self.sample_rate))
                chunk_data.append(sample)
            
            # Convert to bytes
            audio_bytes = struct.pack('<%dh' % len(chunk_data), *chunk_data)
            
            await self.send_audio_chunk(audio_bytes)
            await asyncio.sleep(self.chunk_size / self.sample_rate)  # Real-time simulation
        
        # Send text input as alternative
        await asyncio.sleep(1)
        await self.send_text_input("Hello, this is a test message from the headset client.")
    
    async def send_text_input(self, text: str):
        """Send text input to orchestrator"""
        if not self.connected or not self.websocket:
            print("❌ Not connected to orchestrator")
            return
        
        message = {
            "type": "text_input",
            "session_id": self.session_id,
            "timestamp": time.time(),
            "data": {
                "text": text
            },
            "metadata": {
                "input_method": "keyboard"
            }
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            print(f"✅ Sent: {text}")
            print("🔄 Waiting for response...")
        except Exception as e:
            self.logger.error(f"Failed to send text input: {e}")
            print(f"❌ Failed to send message: {e}")
    
    async def listen_for_responses(self):
        """Listen for responses from orchestrator"""
        try:
            async for message in self.websocket:
                await self.handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Connection to orchestrator closed")
            self.connected = False
        except Exception as e:
            self.logger.error(f"Error listening for responses: {e}")
            self.connected = False
    
    async def handle_message(self, message: str):
        """Handle message from orchestrator"""
        try:
            msg_data = json.loads(message)
            msg_type = msg_data.get('type')
            data = msg_data.get('data', {})
            
            if msg_type == 'transcript_partial':
                transcript = data.get('text', '')
                self.logger.info(f"🎤 Partial: {transcript}")
                
            elif msg_type == 'transcript_final':
                transcript = data.get('text', '')
                self.logger.info(f"🎤 Final: {transcript}")
                
            elif msg_type == 'response_token':
                token = data.get('token', '')
                print(f"🤖 {token}", end='', flush=True)
                
            elif msg_type == 'response_final':
                response = data.get('text', '')
                print(f"\n🤖 Complete: {response}")
                
            elif msg_type == 'audio_output':
                await self.handle_audio_output(data)
                
            elif msg_type == 'error':
                error_msg = data.get('message', 'Unknown error')
                self.logger.error(f"❌ Error: {error_msg}")
                
            elif msg_type == 'session_started':
                self.logger.info("✅ Session started successfully")
                
            else:
                self.logger.debug(f"Received message type: {msg_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def handle_audio_output(self, data: Dict):
        """Handle audio output from TTS"""
        if not self.audio_output_stream:
            self.logger.info("🔊 Received audio output (no speakers available)")
            return
        
        try:
            # Get audio data
            audio_hex = data.get('audio_data', '')
            if audio_hex:
                audio_bytes = bytes.fromhex(audio_hex)
                
                # Play audio
                self.audio_output_stream.write(audio_bytes)
                self.logger.info("🔊 Played audio output")
                
        except Exception as e:
            self.logger.error(f"Error playing audio output: {e}")
    
    async def run_interactive_session(self):
        """Run an interactive session"""
        print("\n🎧 Headset Client Interactive Session")
        print("=" * 50)
        self.show_menu()
        
        # Start listening for responses
        listen_task = asyncio.create_task(self.listen_for_responses())
        
        try:
            while self.connected:
                try:
                    # Get user input (in a separate thread to avoid blocking)
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, input, "\n� Select option (1-4): "
                    )
                    
                    choice = user_input.strip()
                    
                    if choice == '0':
                        print("👋 Exiting session...")
                        break
                    
                    elif choice == '1':
                        # Send text message
                        text_input = await asyncio.get_event_loop().run_in_executor(
                            None, input, "💬 Enter your message: "
                        )
                        if text_input.strip():
                            await self.send_text_input(text_input.strip())
                    
                    elif choice == '2':
                        print("🎤 Starting audio streaming...")
                        await self.start_audio_streaming()
                    
                    elif choice == '3':
                        self.show_menu()
                    
                    else:
                        print(f"❌ Invalid option '{choice}'. Please choose 0-3.")
                        self.show_menu()
                        
                except EOFError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in interactive session: {e}")
                    
        finally:
            listen_task.cancel()
    
    def show_menu(self):
        """Display the interactive menu"""
        print("\n📋 Available Options:")
        print("  1. Send text message")
        print("  2. Start audio streaming")
        print("  3. Show this menu")
        print("  0. Exit session")
        print("-" * 30)
    
    async def disconnect(self):
        """Disconnect from orchestrator"""
        if self.connected:
            # Send session end message
            message = {
                "type": "session_end",
                "session_id": self.session_id,
                "timestamp": time.time(),
                "data": {"reason": "client_disconnect"}
            }
            
            try:
                await self.websocket.send(json.dumps(message))
            except:
                pass
        
        # Close audio streams
        if self.audio_input_stream:
            self.audio_input_stream.stop_stream()
            self.audio_input_stream.close()
        
        if self.audio_output_stream:
            self.audio_output_stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        # Close websocket
        if self.websocket:
            await self.websocket.close()
        
        self.connected = False
        self.logger.info("Disconnected from orchestrator")

async def main():
    """Main function for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    client = HeadsetClient()
    
    try:
        print("🔄 Connecting to orchestrator...")
        await client.connect()
        
        # Initialize audio (optional)
        if client.init_audio():
            print("🎤 Audio initialized successfully")
        else:
            print("🎤 Running in test mode (no audio hardware)")
        
        # Run interactive session
        await client.run_interactive_session()
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.disconnect()
        print("👋 Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())
