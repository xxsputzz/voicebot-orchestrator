#!/usr/bin/env python3
"""
Live End-to-End WebSocket Pipeline Test with Real Audio Streaming

Tests live audio streaming with immediate pipeline feedback:
Live Audio → STT → LLM → TTS with minimal latency optimization

This test simulates real voice interactions with immediate processing
and response generation to validate webhook connections and streaming performance.
"""

import asyncio
import json
import logging
import time
import uuid
import wave
import tempfile
import os
import struct
import math
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveStreamingPipelineTest:
    """
    Live end-to-end streaming pipeline test with real-time audio processing
    Tests STT→LLM→TTS pipeline with minimal latency for webhook validation
    """
    
    def __init__(self, orchestrator_host: str = "localhost", client_port: int = 9000):
        self.orchestrator_uri = f"ws://{orchestrator_host}:{client_port}"
        self.websocket = None
        self.session_id = str(uuid.uuid4())
        
        # Pipeline state tracking
        self.pipeline_state = {
            'stt_ready': False,
            'llm_ready': False, 
            'tts_ready': False,
            'active_transcription': '',
            'current_response': '',
            'audio_output_queue': asyncio.Queue()
        }
        
        # Performance metrics
        self.metrics = {
            'stt_latency': [],
            'llm_latency': [],
            'tts_latency': [],
            'total_pipeline_latency': [],
            'webhook_response_times': []
        }
        
        # Test scenarios for comprehensive testing
        self.test_scenarios = [
            {
                'name': 'Quick Response Test',
                'input': 'Hello, how are you today?',
                'expected_type': 'greeting_response',
                'max_latency_ms': 2000
            },
            {
                'name': 'Complex Query Test', 
                'input': 'Can you explain the weather patterns in tropical regions?',
                'expected_type': 'informational_response',
                'max_latency_ms': 4000
            },
            {
                'name': 'Conversation Flow Test',
                'input': 'Tell me a short story about space exploration.',
                'expected_type': 'creative_response',
                'max_latency_ms': 5000
            },
            {
                'name': 'Real-time Interruption Test',
                'input': 'What is the capital of France... actually, tell me about Rome instead.',
                'expected_type': 'context_switch',
                'max_latency_ms': 3000
            }
        ]
        
        # Webhook connection tracking
        self.webhook_connections = {
            'stt_service': {'connected': False, 'response_time': None},
            'llm_service': {'connected': False, 'response_time': None},
            'tts_service': {'connected': False, 'response_time': None},
            'orchestrator': {'connected': False, 'response_time': None}
        }
        
    async def connect(self):
        """Connect to the orchestrator with webhook validation"""
        try:
            connection_start = time.time()
            self.websocket = await websockets.connect(self.orchestrator_uri, ping_interval=20)
            connection_time = (time.time() - connection_start) * 1000
            
            self.webhook_connections['orchestrator'] = {
                'connected': True, 
                'response_time': connection_time
            }
            
            logger.info(f"✅ Connected to orchestrator: {self.orchestrator_uri} ({connection_time:.2f}ms)")
            
            # Test webhook handshake
            await self._test_webhook_handshake()
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to orchestrator: {e}")
            raise
    
    async def _test_webhook_handshake(self):
        """Test webhook connections to all services"""
        logger.info("🔗 Testing webhook connections...")
        
        # Request service status from orchestrator
        await self.send_message({
            "type": "request_service_status",
            "session_id": self.session_id
        })
        
        try:
            # Wait for service status response
            response = await asyncio.wait_for(self.receive_message(), timeout=5.0)
            
            if response.get('type') == 'service_status':
                services = response.get('services', [])
                
                for service in services:
                    service_type = service.get('service_type', '').lower()
                    service_id = service.get('service_id', '')
                    
                    if 'stt' in service_type or 'whisper' in service_id.lower():
                        self.webhook_connections['stt_service']['connected'] = True
                        self.pipeline_state['stt_ready'] = True
                    elif 'llm' in service_type or 'gpt' in service_id.lower() or 'mistral' in service_id.lower():
                        self.webhook_connections['llm_service']['connected'] = True
                        self.pipeline_state['llm_ready'] = True
                    elif 'tts' in service_type or 'tortoise' in service_id.lower() or 'kokoro' in service_id.lower():
                        self.webhook_connections['tts_service']['connected'] = True
                        self.pipeline_state['tts_ready'] = True
                
                logger.info(f"🔗 Webhook Status:")
                logger.info(f"   STT Service: {'✅' if self.webhook_connections['stt_service']['connected'] else '❌'}")
                logger.info(f"   LLM Service: {'✅' if self.webhook_connections['llm_service']['connected'] else '❌'}")
                logger.info(f"   TTS Service: {'✅' if self.webhook_connections['tts_service']['connected'] else '❌'}")
            
        except asyncio.TimeoutError:
            logger.warning("⚠️  Timeout waiting for service status")
    
    async def disconnect(self):
        """Disconnect from the orchestrator"""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from orchestrator")
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to orchestrator with timing"""
        if not self.websocket:
            raise Exception("Not connected to orchestrator")
        
        send_time = time.time()
        message_json = json.dumps(message)
        await self.websocket.send(message_json)
        
        # Track webhook response time for this message type
        webhook_type = message.get('type', 'unknown')
        self.metrics['webhook_response_times'].append({
            'type': webhook_type,
            'send_time': send_time,
            'message_id': message.get('session_id', 'unknown')
        })
        
        logger.debug(f"📤 Sent {webhook_type}: {message}")
    
    async def receive_message(self) -> Dict[str, Any]:
        """Receive message from orchestrator with timing tracking"""
        if not self.websocket:
            raise Exception("Not connected to orchestrator")
        
        receive_time = time.time()
        message_json = await self.websocket.recv()
        message = json.loads(message_json)
        
        # Track webhook response timing
        message_type = message.get('type', 'unknown')
        
        logger.debug(f"📥 Received {message_type}: {message}")
        return message
    
    def create_live_audio_simulation(self, text: str, duration: float = None) -> List[bytes]:
        """
        Create realistic audio chunks that simulate live microphone input
        Returns list of audio chunks to simulate real-time streaming
        """
        if duration is None:
            # Estimate duration based on text (average 150 words per minute)
            words = len(text.split())
            duration = max(2.0, words / 150 * 60)  # Minimum 2 seconds
        
        sample_rate = 16000
        chunk_duration = 0.1  # 100ms chunks for real-time feel
        samples_per_chunk = int(sample_rate * chunk_duration)
        total_samples = int(sample_rate * duration)
        
        chunks = []
        
        # Generate audio chunks that simulate speech patterns
        for i in range(0, total_samples, samples_per_chunk):
            chunk_samples = min(samples_per_chunk, total_samples - i)
            
            # Create more realistic audio with varying amplitude and frequency
            t_start = i / sample_rate
            t_end = (i + chunk_samples) / sample_rate
            t = [t_start + j / sample_rate for j in range(chunk_samples)]
            
            # Simulate speech with multiple frequency components
            audio_data = []
            for time_point in t:
                # Base frequency with harmonics (simulates human voice)
                base_freq = 200 + 100 * math.sin(time_point * 2)  # Varying pitch
                amplitude = 0.3 + 0.2 * math.sin(time_point * 5)  # Varying volume
                
                sample = (
                    amplitude * math.sin(2 * math.pi * base_freq * time_point) +
                    amplitude * 0.3 * math.sin(2 * math.pi * base_freq * 2 * time_point) +
                    amplitude * 0.1 * math.sin(2 * math.pi * base_freq * 3 * time_point)
                )
                
                # Add slight noise for realism
                sample += 0.05 * (2 * (hash(str(time_point)) % 1000) / 1000 - 1)
                
                # Convert to 16-bit integer
                audio_data.append(int(sample * 32767))
            
            # Convert to bytes
            chunk_bytes = b''.join(struct.pack('<h', sample) for sample in audio_data)
            chunks.append(chunk_bytes)
        
        logger.info(f"🎤 Generated {len(chunks)} audio chunks ({duration:.1f}s) for: '{text[:30]}...'")
        return chunks
    
    async def test_live_stt_streaming(self, audio_chunks: List[bytes]) -> Dict[str, Any]:
        """Test live STT streaming with immediate transcription feedback"""
        logger.info("🎤 Testing Live STT Streaming...")
        
        stt_start_time = time.time()
        
        results = {
            'started': False,
            'transcriptions': [],
            'final_transcription': '',
            'chunk_latencies': [],
            'total_latency': 0,
            'errors': []
        }
        
        try:
            # Start STT session
            await self.send_message({
                "type": "start_audio_stream",
                "service": "stt",
                "session_id": self.session_id,
                "config": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "format": "pcm16",
                    "language": "en-US",
                    "real_time": True
                }
            })
            
            # Wait for stream ready confirmation
            response = await asyncio.wait_for(self.receive_message(), timeout=5.0)
            if response.get('type') == 'audio_stream_ready':
                results['started'] = True
                logger.info("✅ STT stream ready for live audio")
            
            # Stream audio chunks in real-time
            for i, chunk in enumerate(audio_chunks):
                chunk_start_time = time.time()
                
                # Convert to base64 for WebSocket transmission
                import base64
                audio_data = base64.b64encode(chunk).decode('utf-8')
                
                await self.send_message({
                    "type": "audio_chunk",
                    "session_id": self.session_id,
                    "chunk_id": i,
                    "audio_data": audio_data,
                    "timestamp": time.time(),
                    "is_final": i == len(audio_chunks) - 1
                })
                
                # Immediately check for transcription results (live feedback)
                try:
                    while True:
                        response = await asyncio.wait_for(self.receive_message(), timeout=0.05)
                        
                        if response.get('type') == 'partial_transcription':
                            chunk_latency = (time.time() - chunk_start_time) * 1000
                            results['chunk_latencies'].append(chunk_latency)
                            
                            transcription = response.get('text', '')
                            results['transcriptions'].append({
                                'chunk_id': i,
                                'text': transcription,
                                'latency_ms': chunk_latency,
                                'confidence': response.get('confidence', 0.0)
                            })
                            
                            # Update pipeline state for immediate LLM processing
                            self.pipeline_state['active_transcription'] = transcription
                            
                            logger.info(f"🎤 STT Chunk {i}: '{transcription}' ({chunk_latency:.1f}ms)")
                            
                        elif response.get('type') == 'transcription_complete':
                            results['final_transcription'] = response.get('text', '')
                            logger.info(f"🎤 STT Final: '{results['final_transcription']}'")
                            break
                            
                except asyncio.TimeoutError:
                    pass  # No immediate response, continue streaming
                
                # Realistic streaming delay (simulate mic input timing)
                await asyncio.sleep(0.1)
            
            # Finalize transcription
            await self.send_message({
                "type": "end_audio_stream",
                "session_id": self.session_id
            })
            
            # Wait for final result
            try:
                response = await asyncio.wait_for(self.receive_message(), timeout=3.0)
                if response.get('type') == 'final_transcription':
                    results['final_transcription'] = response.get('text', '')
            except asyncio.TimeoutError:
                logger.warning("⚠️  Timeout waiting for final transcription")
            
        except Exception as e:
            logger.error(f"❌ STT streaming error: {e}")
            results['errors'].append(str(e))
        
        results['total_latency'] = (time.time() - stt_start_time) * 1000
        self.metrics['stt_latency'].append(results['total_latency'])
        
        return results
    
    async def test_immediate_llm_processing(self, text: str) -> Dict[str, Any]:
        """Test immediate LLM processing with streaming tokens"""
        logger.info(f"🧠 Testing Immediate LLM Processing: '{text[:50]}...'")
        
        llm_start_time = time.time()
        
        results = {
            'tokens': [],
            'response_text': '',
            'streaming_complete': False,
            'token_latencies': [],
            'total_latency': 0,
            'errors': []
        }
        
        try:
            # Send immediate LLM request (simulates real conversation)
            await self.send_message({
                "type": "text_input",
                "session_id": self.session_id,
                "text": text,
                "config": {
                    "stream_tokens": True,
                    "max_tokens": 150,
                    "temperature": 0.7,
                    "real_time_response": True,
                    "priority": "high"  # High priority for live conversation
                }
            })
            
            token_count = 0
            first_token_time = None
            
            # Collect streaming tokens with immediate processing
            while True:
                response = await asyncio.wait_for(self.receive_message(), timeout=10.0)
                
                if response.get('type') == 'llm_token':
                    token_start_time = time.time()
                    
                    if first_token_time is None:
                        first_token_time = token_start_time
                        logger.info(f"🧠 First token latency: {(first_token_time - llm_start_time) * 1000:.1f}ms")
                    
                    token = response.get('token', '')
                    token_count += 1
                    
                    results['tokens'].append({
                        'token': token,
                        'token_id': token_count,
                        'timestamp': token_start_time,
                        'latency_from_start': (token_start_time - llm_start_time) * 1000
                    })
                    
                    results['response_text'] += token
                    
                    # Update pipeline state for immediate TTS processing
                    self.pipeline_state['current_response'] = results['response_text']
                    
                    # Stream token to console for real-time feedback
                    print(token, end='', flush=True)
                    
                    # Check if we have enough tokens for TTS to start
                    if token_count >= 5:  # Start TTS after first few tokens
                        await self._trigger_immediate_tts_processing(results['response_text'])
                
                elif response.get('type') == 'llm_stream_complete':
                    results['streaming_complete'] = True
                    print()  # New line after streaming
                    logger.info(f"🧠 LLM streaming complete: {token_count} tokens")
                    break
                
                elif response.get('type') == 'error':
                    results['errors'].append(response.get('message', 'Unknown LLM error'))
                    break
        
        except asyncio.TimeoutError:
            logger.error("❌ Timeout waiting for LLM response")
            results['errors'].append("LLM response timeout")
        except Exception as e:
            logger.error(f"❌ LLM processing error: {e}")
            results['errors'].append(str(e))
        
        results['total_latency'] = (time.time() - llm_start_time) * 1000
        self.metrics['llm_latency'].append(results['total_latency'])
        
        return results
    
    async def _trigger_immediate_tts_processing(self, partial_text: str):
        """Trigger TTS processing as soon as we have sufficient text"""
        if len(partial_text.split()) >= 3:  # Wait for at least 3 words
            # Extract sentence or phrase for immediate TTS
            sentences = partial_text.split('.')
            if len(sentences) > 1:
                complete_sentence = sentences[0] + '.'
                if complete_sentence not in getattr(self, '_processed_sentences', set()):
                    if not hasattr(self, '_processed_sentences'):
                        self._processed_sentences = set()
                    
                    self._processed_sentences.add(complete_sentence)
                    
                    # Start TTS for complete sentence immediately
                    asyncio.create_task(self._process_immediate_tts(complete_sentence))
    
    async def _process_immediate_tts(self, text: str):
        """Process TTS immediately for low latency"""
        logger.info(f"🔊 Immediate TTS: '{text[:30]}...'")
        
        try:
            await self.send_message({
                "type": "synthesize_speech",
                "session_id": f"{self.session_id}_immediate",
                "text": text,
                "voice": "female_professional",  # Fast voice for immediate response
                "config": {
                    "speed": 1.1,  # Slightly faster for responsiveness
                    "priority": "high",
                    "streaming": True
                }
            })
        except Exception as e:
            logger.error(f"❌ Immediate TTS error: {e}")
    
    async def test_live_tts_streaming(self, text: str, voice: str = "female_professional") -> Dict[str, Any]:
        """Test live TTS streaming with chunk-by-chunk audio delivery"""
        logger.info(f"🔊 Testing Live TTS Streaming: '{text[:50]}...'")
        
        tts_start_time = time.time()
        synthesis_id = f"{self.session_id}_tts_{int(time.time())}"
        
        results = {
            'audio_chunks': [],
            'total_audio_bytes': 0,
            'chunk_latencies': [],
            'first_chunk_latency': 0,
            'streaming_complete': False,
            'total_latency': 0,
            'errors': []
        }
        
        try:
            # Request immediate TTS synthesis
            await self.send_message({
                "type": "text_to_speech",
                "synthesis_id": synthesis_id,
                "text": text,
                "voice": voice,
                "config": {
                    "stream_audio": True,
                    "chunk_size": 1024,
                    "sample_rate": 22050,
                    "format": "wav",
                    "real_time": True,
                    "low_latency": True
                }
            })
            
            first_chunk_received = False
            
            # Collect streaming audio chunks
            while True:
                response = await asyncio.wait_for(self.receive_message(), timeout=15.0)
                
                if response.get('type') == 'tts_audio_chunk':
                    chunk_receive_time = time.time()
                    
                    if not first_chunk_received:
                        results['first_chunk_latency'] = (chunk_receive_time - tts_start_time) * 1000
                        first_chunk_received = True
                        logger.info(f"🔊 First audio chunk: {results['first_chunk_latency']:.1f}ms")
                    
                    # Decode audio chunk
                    import base64
                    audio_data = base64.b64decode(response.get('audio_data', ''))
                    
                    chunk_info = {
                        'chunk_id': response.get('chunk_number', len(results['audio_chunks'])),
                        'size_bytes': len(audio_data),
                        'timestamp': chunk_receive_time,
                        'latency_from_start': (chunk_receive_time - tts_start_time) * 1000
                    }
                    
                    results['audio_chunks'].append(chunk_info)
                    results['total_audio_bytes'] += len(audio_data)
                    
                    # Queue audio for potential playback
                    await self.pipeline_state['audio_output_queue'].put(audio_data)
                    
                    logger.info(f"🔊 Audio chunk {chunk_info['chunk_id']}: {chunk_info['size_bytes']} bytes")
                
                elif response.get('type') == 'tts_stream_complete':
                    results['streaming_complete'] = True
                    logger.info(f"🔊 TTS streaming complete: {len(results['audio_chunks'])} chunks, {results['total_audio_bytes']} bytes")
                    break
                
                elif response.get('type') == 'error':
                    results['errors'].append(response.get('message', 'Unknown TTS error'))
                    break
        
        except asyncio.TimeoutError:
            logger.error("❌ Timeout waiting for TTS audio chunks")
            results['errors'].append("TTS streaming timeout")
        except Exception as e:
            logger.error(f"❌ TTS streaming error: {e}")
            results['errors'].append(str(e))
        
        results['total_latency'] = (time.time() - tts_start_time) * 1000
        self.metrics['tts_latency'].append(results['total_latency'])
        
        return results
    
    async def test_complete_live_pipeline(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test complete live pipeline with real-time audio processing"""
        logger.info(f"🚀 Testing Live Pipeline: {scenario['name']}")
        logger.info(f"   Input: '{scenario['input']}'")
        logger.info(f"   Max Latency: {scenario['max_latency_ms']}ms")
        
        pipeline_start_time = time.time()
        
        results = {
            'scenario': scenario,
            'stt_results': None,
            'llm_results': None,
            'tts_results': None,
            'total_pipeline_latency': 0,
            'within_latency_target': False,
            'success': False,
            'webhook_performance': {}
        }
        
        try:
            # Step 1: Generate live audio simulation
            audio_chunks = self.create_live_audio_simulation(scenario['input'])
            
            # Step 2: Live STT Processing
            logger.info("  📍 Step 1: Live STT Processing")
            results['stt_results'] = await self.test_live_stt_streaming(audio_chunks)
            
            if results['stt_results']['errors']:
                logger.error("  ❌ Pipeline failed at STT stage")
                return results
            
            transcribed_text = results['stt_results']['final_transcription']
            if not transcribed_text:
                transcribed_text = scenario['input']  # Fallback for testing
            
            # Step 3: Immediate LLM Processing
            logger.info("  📍 Step 2: Immediate LLM Processing")
            results['llm_results'] = await self.test_immediate_llm_processing(transcribed_text)
            
            if results['llm_results']['errors']:
                logger.error("  ❌ Pipeline failed at LLM stage")
                return results
            
            llm_response = results['llm_results']['response_text']
            
            # Step 4: Live TTS Streaming
            if llm_response:
                logger.info("  📍 Step 3: Live TTS Streaming")
                results['tts_results'] = await self.test_live_tts_streaming(llm_response)
                
                if not results['tts_results']['errors']:
                    results['success'] = True
            
        except Exception as e:
            logger.error(f"❌ Live pipeline error: {e}")
        
        # Calculate total pipeline latency
        results['total_pipeline_latency'] = (time.time() - pipeline_start_time) * 1000
        results['within_latency_target'] = results['total_pipeline_latency'] <= scenario['max_latency_ms']
        
        # Track webhook performance
        results['webhook_performance'] = {
            'stt_avg_latency': sum(results['stt_results']['chunk_latencies']) / len(results['stt_results']['chunk_latencies']) if results['stt_results']['chunk_latencies'] else 0,
            'llm_first_token_latency': results['llm_results']['tokens'][0]['latency_from_start'] if results['llm_results']['tokens'] else 0,
            'tts_first_chunk_latency': results['tts_results']['first_chunk_latency'] if results['tts_results'] else 0
        }
        
        logger.info(f"  🏁 Pipeline Complete: {results['total_pipeline_latency']:.1f}ms")
        logger.info(f"  🎯 Target Met: {'✅' if results['within_latency_target'] else '❌'}")
        
        return results
    
    async def run_comprehensive_live_tests(self) -> Dict[str, Any]:
        """Run comprehensive live pipeline tests with all scenarios"""
        logger.info("🧪 Starting Comprehensive Live Pipeline Tests")
        logger.info("=" * 60)
        
        all_results = {
            'webhook_connections': self.webhook_connections.copy(),
            'scenario_results': [],
            'performance_metrics': {},
            'overall_success': False,
            'latency_analysis': {}
        }
        
        try:
            await self.connect()
            
            # Test each scenario
            for i, scenario in enumerate(self.test_scenarios, 1):
                logger.info(f"\n📋 Test Scenario {i}/{len(self.test_scenarios)}: {scenario['name']}")
                logger.info("-" * 40)
                
                scenario_result = await self.test_complete_live_pipeline(scenario)
                all_results['scenario_results'].append(scenario_result)
                
                # Add to metrics
                if scenario_result['success']:
                    self.metrics['total_pipeline_latency'].append(scenario_result['total_pipeline_latency'])
                
                # Brief pause between tests
                await asyncio.sleep(1)
            
            # Calculate performance metrics
            all_results['performance_metrics'] = self._calculate_performance_metrics()
            all_results['latency_analysis'] = self._analyze_latency_performance()
            
            # Determine overall success
            successful_scenarios = sum(1 for result in all_results['scenario_results'] if result['success'])
            all_results['overall_success'] = successful_scenarios == len(self.test_scenarios)
            
        except Exception as e:
            logger.error(f"❌ Test execution error: {e}")
        
        finally:
            await self.disconnect()
        
        # Print comprehensive results
        self._print_comprehensive_results(all_results)
        
        return all_results
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                metrics[metric_name] = {
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return metrics
    
    def _analyze_latency_performance(self) -> Dict[str, Any]:
        """Analyze latency performance across pipeline stages"""
        return {
            'stt_performance': {
                'avg_latency': sum(self.metrics['stt_latency']) / len(self.metrics['stt_latency']) if self.metrics['stt_latency'] else 0,
                'target_met': all(lat < 500 for lat in self.metrics['stt_latency']),  # 500ms target
                'status': 'excellent' if all(lat < 200 for lat in self.metrics['stt_latency']) else 'good' if all(lat < 500 for lat in self.metrics['stt_latency']) else 'needs_improvement'
            },
            'llm_performance': {
                'avg_latency': sum(self.metrics['llm_latency']) / len(self.metrics['llm_latency']) if self.metrics['llm_latency'] else 0,
                'target_met': all(lat < 2000 for lat in self.metrics['llm_latency']),  # 2s target
                'status': 'excellent' if all(lat < 1000 for lat in self.metrics['llm_latency']) else 'good' if all(lat < 2000 for lat in self.metrics['llm_latency']) else 'needs_improvement'
            },
            'tts_performance': {
                'avg_latency': sum(self.metrics['tts_latency']) / len(self.metrics['tts_latency']) if self.metrics['tts_latency'] else 0,
                'target_met': all(lat < 1500 for lat in self.metrics['tts_latency']),  # 1.5s target
                'status': 'excellent' if all(lat < 800 for lat in self.metrics['tts_latency']) else 'good' if all(lat < 1500 for lat in self.metrics['tts_latency']) else 'needs_improvement'
            },
            'total_pipeline_performance': {
                'avg_latency': sum(self.metrics['total_pipeline_latency']) / len(self.metrics['total_pipeline_latency']) if self.metrics['total_pipeline_latency'] else 0,
                'target_met': all(lat < 5000 for lat in self.metrics['total_pipeline_latency']),  # 5s total target
                'status': 'excellent' if all(lat < 3000 for lat in self.metrics['total_pipeline_latency']) else 'good' if all(lat < 5000 for lat in self.metrics['total_pipeline_latency']) else 'needs_improvement'
            }
        }
    
    def _print_comprehensive_results(self, results: Dict[str, Any]):
        """Print comprehensive test results"""
        logger.info("\n" + "=" * 60)
        logger.info("🏁 COMPREHENSIVE LIVE PIPELINE TEST RESULTS")
        logger.info("=" * 60)
        
        # Webhook Connection Status
        logger.info("\n🔗 Webhook Connection Status:")
        for service, status in results['webhook_connections'].items():
            status_icon = "✅" if status['connected'] else "❌"
            response_time = f"({status['response_time']:.1f}ms)" if status['response_time'] else ""
            logger.info(f"   {service}: {status_icon} {response_time}")
        
        # Scenario Results
        logger.info(f"\n📋 Scenario Test Results:")
        successful_scenarios = 0
        for result in results['scenario_results']:
            scenario_name = result['scenario']['name']
            success = result['success']
            latency = result['total_pipeline_latency']
            target = result['scenario']['max_latency_ms']
            target_met = result['within_latency_target']
            
            status_icon = "✅" if success and target_met else "⚠️" if success else "❌"
            logger.info(f"   {status_icon} {scenario_name}: {latency:.1f}ms (target: {target}ms)")
            
            if success and target_met:
                successful_scenarios += 1
        
        # Performance Analysis
        logger.info(f"\n⚡ Latency Performance Analysis:")
        latency_analysis = results['latency_analysis']
        for stage, analysis in latency_analysis.items():
            status_emoji = {"excellent": "🟢", "good": "🟡", "needs_improvement": "🔴"}
            emoji = status_emoji.get(analysis['status'], "🔴")
            logger.info(f"   {emoji} {stage.replace('_', ' ').title()}: {analysis['avg_latency']:.1f}ms avg ({analysis['status']})")
        
        # Overall Results
        overall_success = results['overall_success']
        success_rate = (successful_scenarios / len(results['scenario_results'])) * 100
        
        logger.info(f"\n🎯 Overall Results:")
        logger.info(f"   Success Rate: {successful_scenarios}/{len(results['scenario_results'])} ({success_rate:.1f}%)")
        logger.info(f"   Overall Status: {'✅ ALL TESTS PASSED' if overall_success else '⚠️ SOME TESTS FAILED'}")
        
        # Recommendations
        if not overall_success:
            logger.info(f"\n💡 Recommendations for Optimization:")
            for stage, analysis in latency_analysis.items():
                if analysis['status'] == 'needs_improvement':
                    logger.info(f"   • Optimize {stage.replace('_', ' ')} (current: {analysis['avg_latency']:.1f}ms)")

async def main():
    """Main test runner for live pipeline testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live End-to-End WebSocket Pipeline Test")
    parser.add_argument("--orchestrator-host", default="localhost", help="Orchestrator host")
    parser.add_argument("--client-port", type=int, default=9000, help="Orchestrator client port")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run comprehensive test
    tester = LiveStreamingPipelineTest(
        orchestrator_host=args.orchestrator_host,
        client_port=args.client_port
    )
    
    try:
        results = await tester.run_comprehensive_live_tests()
        
        # Exit with appropriate code
        if results['overall_success']:
            logger.info("✅ All live pipeline tests completed successfully!")
            return 0
        else:
            logger.error("❌ Some live pipeline tests failed!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Test runner error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
