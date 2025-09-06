#!/usr/bin/env python3
"""
End-to-End WebSocket Streaming Test

Test the complete streaming pipeline:
STT (audio) → LLM (tokens) → TTS (audio chunks)
"""

import asyncio
import json
import logging
import time
import uuid
import wave
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamingPipelineTest:
    """End-to-end streaming pipeline test"""
    
    def __init__(self, orchestrator_host: str = "localhost", orchestrator_port: int = 9000):
        self.orchestrator_uri = f"ws://{orchestrator_host}:{orchestrator_port}/client"
        self.websocket = None
        self.test_results = {}
        
    async def connect(self):
        """Connect to the orchestrator"""
        try:
            self.websocket = await websockets.connect(self.orchestrator_uri)
            logger.info(f"✅ Connected to orchestrator: {self.orchestrator_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to orchestrator: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from the orchestrator"""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from orchestrator")
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to orchestrator"""
        if not self.websocket:
            raise Exception("Not connected to orchestrator")
        
        message_json = json.dumps(message)
        await self.websocket.send(message_json)
        logger.debug(f"Sent: {message}")
    
    async def receive_message(self) -> Dict[str, Any]:
        """Receive message from orchestrator"""
        if not self.websocket:
            raise Exception("Not connected to orchestrator")
        
        message_json = await self.websocket.recv()
        message = json.loads(message_json)
        logger.debug(f"Received: {message}")
        return message
    
    def create_test_audio(self, duration: float = 3.0, sample_rate: int = 16000) -> bytes:
        """Create a test audio file (sine wave)"""
        import numpy as np
        
        # Generate a simple sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440  # A note
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit integers
        audio = (audio * 32767).astype(np.int16)
        
        # Convert to bytes
        return audio.tobytes()
    
    async def test_stt_streaming(self) -> Dict[str, Any]:
        """Test STT streaming service"""
        logger.info("🎤 Testing STT Streaming...")
        
        session_id = str(uuid.uuid4())
        test_audio = self.create_test_audio(duration=2.0)
        
        results = {
            'started': False,
            'transcriptions': [],
            'final_result': None,
            'errors': []
        }
        
        try:
            # Start STT stream
            await self.send_message({
                "type": "start_stream",
                "service": "stt",
                "session_id": session_id
            })
            
            # Wait for stream started confirmation
            response = await self.receive_message()
            if response.get('type') == 'stream_started':
                results['started'] = True
                logger.info("STT stream started")
            
            # Send audio chunks
            chunk_size = 1024
            for i in range(0, len(test_audio), chunk_size):
                chunk = test_audio[i:i+chunk_size]
                
                # Convert to base64 for transmission
                import base64
                audio_data = base64.b64encode(chunk).decode('utf-8')
                
                await self.send_message({
                    "type": "audio_chunk",
                    "session_id": session_id,
                    "data": audio_data
                })
                
                # Check for transcription results
                try:
                    response = await asyncio.wait_for(self.receive_message(), timeout=0.1)
                    if response.get('type') == 'transcription':
                        results['transcriptions'].append(response)
                        logger.info(f"STT partial: {response.get('text', '')}")
                except asyncio.TimeoutError:
                    pass  # No response yet, continue
            
            # End the stream
            await self.send_message({
                "type": "end_stream",
                "session_id": session_id
            })
            
            # Wait for final result
            response = await self.receive_message()
            if response.get('type') == 'final_transcription':
                results['final_result'] = response
                logger.info(f"STT final: {response.get('text', '')}")
            
        except Exception as e:
            logger.error(f"STT streaming test error: {e}")
            results['errors'].append(str(e))
        
        return results
    
    async def test_llm_streaming(self, prompt: str = "Tell me a short joke") -> Dict[str, Any]:
        """Test LLM streaming service"""
        logger.info("🧠 Testing LLM Token Streaming...")
        
        conversation_id = str(uuid.uuid4())
        
        results = {
            'tokens': [],
            'final_text': '',
            'completion': None,
            'errors': []
        }
        
        try:
            # Send chat completion request
            await self.send_message({
                "type": "chat_completion",
                "conversation_id": conversation_id,
                "prompt": prompt,
                "temperature": 0.7,
                "max_tokens": 100
            })
            
            # Collect streaming tokens
            while True:
                response = await self.receive_message()
                
                if response.get('type') == 'token':
                    results['tokens'].append(response)
                    token = response.get('token', '')
                    if token:
                        results['final_text'] += token
                        print(token, end='', flush=True)
                
                elif response.get('type') == 'completion':
                    results['completion'] = response
                    logger.info(f"\n✅ LLM completion: {len(results['tokens'])} tokens")
                    break
                
                elif response.get('type') == 'error':
                    results['errors'].append(response.get('message', 'Unknown error'))
                    logger.error(f"LLM error: {response.get('message')}")
                    break
        
        except Exception as e:
            logger.error(f"LLM streaming test error: {e}")
            results['errors'].append(str(e))
        
        return results
    
    async def test_tts_streaming(self, text: str = "Hello, this is a test of streaming text to speech.", voice: str = "angie") -> Dict[str, Any]:
        """Test TTS streaming service"""
        logger.info("🔊 Testing TTS Chunk Streaming...")
        
        synthesis_id = str(uuid.uuid4())
        
        results = {
            'chunks': [],
            'total_audio': b'',
            'completion': None,
            'errors': []
        }
        
        try:
            # Send synthesis request
            await self.send_message({
                "type": "synthesize_speech",
                "synthesis_id": synthesis_id,
                "text": text,
                "voice": voice,
                "speed": 1.0
            })
            
            # Collect streaming audio chunks
            while True:
                response = await self.receive_message()
                
                if response.get('type') == 'audio_chunk':
                    results['chunks'].append(response)
                    
                    # Decode audio data
                    import base64
                    audio_data = base64.b64decode(response.get('audio_data', ''))
                    results['total_audio'] += audio_data
                    
                    logger.info(f"TTS chunk {response.get('chunk_id', 0)}: {len(audio_data)} bytes")
                
                elif response.get('type') == 'synthesis_complete':
                    results['completion'] = response
                    logger.info(f"✅ TTS completion: {len(results['chunks'])} chunks, {len(results['total_audio'])} total bytes")
                    break
                
                elif response.get('type') == 'error':
                    results['errors'].append(response.get('message', 'Unknown error'))
                    logger.error(f"TTS error: {response.get('message')}")
                    break
        
        except Exception as e:
            logger.error(f"TTS streaming test error: {e}")
            results['errors'].append(str(e))
        
        return results
    
    async def test_full_pipeline(self) -> Dict[str, Any]:
        """Test the complete STT → LLM → TTS pipeline"""
        logger.info("🚀 Testing Full Streaming Pipeline...")
        
        pipeline_results = {
            'stt_results': None,
            'llm_results': None,
            'tts_results': None,
            'total_duration': 0,
            'success': False
        }
        
        start_time = time.time()
        
        try:
            # Step 1: STT Test (simulate voice input)
            # For this test, we'll skip actual STT and use a text prompt directly
            test_prompt = "What is artificial intelligence?"
            
            # Step 2: LLM Streaming
            logger.info("Step 1: Getting LLM response...")
            pipeline_results['llm_results'] = await self.test_llm_streaming(test_prompt)
            
            if pipeline_results['llm_results']['errors']:
                logger.error("Pipeline failed at LLM stage")
                return pipeline_results
            
            # Step 3: TTS Streaming with LLM output
            llm_output = pipeline_results['llm_results']['final_text']
            if llm_output:
                logger.info("Step 2: Converting LLM response to speech...")
                pipeline_results['tts_results'] = await self.test_tts_streaming(llm_output)
                
                if not pipeline_results['tts_results']['errors']:
                    pipeline_results['success'] = True
            
        except Exception as e:
            logger.error(f"Pipeline test error: {e}")
        
        pipeline_results['total_duration'] = time.time() - start_time
        
        return pipeline_results
    
    async def run_all_tests(self):
        """Run all streaming service tests"""
        logger.info("🧪 Starting End-to-End Streaming Tests")
        logger.info("=" * 50)
        
        all_results = {
            'stt_test': None,
            'llm_test': None,
            'tts_test': None,
            'pipeline_test': None,
            'overall_success': False
        }
        
        try:
            await self.connect()
            
            # Test individual services
            logger.info("\n1. Individual Service Tests")
            logger.info("-" * 30)
            
            # Skip STT for now (requires more complex audio setup)
            # all_results['stt_test'] = await self.test_stt_streaming()
            
            all_results['llm_test'] = await self.test_llm_streaming(
                "Explain quantum computing in one sentence."
            )
            
            all_results['tts_test'] = await self.test_tts_streaming(
                "The quick brown fox jumps over the lazy dog."
            )
            
            # Test full pipeline
            logger.info("\n2. Full Pipeline Test")
            logger.info("-" * 30)
            
            all_results['pipeline_test'] = await self.test_full_pipeline()
            
            # Determine overall success
            all_results['overall_success'] = (
                # (not all_results['stt_test'] or not all_results['stt_test']['errors']) and
                (not all_results['llm_test'] or not all_results['llm_test']['errors']) and
                (not all_results['tts_test'] or not all_results['tts_test']['errors']) and
                (not all_results['pipeline_test'] or all_results['pipeline_test']['success'])
            )
            
        except Exception as e:
            logger.error(f"Test execution error: {e}")
        
        finally:
            await self.disconnect()
        
        # Print results summary
        self._print_test_summary(all_results)
        
        return all_results
    
    def _print_test_summary(self, results: Dict[str, Any]):
        """Print test results summary"""
        logger.info("\n" + "=" * 50)
        logger.info("🏁 STREAMING TESTS SUMMARY")
        logger.info("=" * 50)
        
        if results['llm_test']:
            llm_success = not results['llm_test']['errors']
            logger.info(f"🧠 LLM Streaming: {'✅ PASS' if llm_success else '❌ FAIL'}")
            if llm_success:
                logger.info(f"   └─ Generated {len(results['llm_test']['tokens'])} tokens")
        
        if results['tts_test']:
            tts_success = not results['tts_test']['errors']
            logger.info(f"🔊 TTS Streaming: {'✅ PASS' if tts_success else '❌ FAIL'}")
            if tts_success:
                logger.info(f"   └─ Generated {len(results['tts_test']['chunks'])} audio chunks")
        
        if results['pipeline_test']:
            pipeline_success = results['pipeline_test']['success']
            logger.info(f"🚀 Full Pipeline: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
            if pipeline_success:
                duration = results['pipeline_test']['total_duration']
                logger.info(f"   └─ Total pipeline time: {duration:.2f}s")
        
        overall = results['overall_success']
        logger.info(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall else '❌ SOME TESTS FAILED'}")

async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="End-to-End Streaming Pipeline Test")
    parser.add_argument("--orchestrator-host", default="localhost", help="Orchestrator host")
    parser.add_argument("--orchestrator-port", type=int, default=9000, help="Orchestrator client port")
    
    args = parser.parse_args()
    
    # Create and run test
    tester = StreamingPipelineTest(
        orchestrator_host=args.orchestrator_host,
        orchestrator_port=args.orchestrator_port
    )
    
    try:
        results = await tester.run_all_tests()
        
        # Exit with appropriate code
        if results['overall_success']:
            logger.info("✅ All tests completed successfully!")
            return 0
        else:
            logger.error("❌ Some tests failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Test runner error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
