"""
Streaming TTS Engine
Advanced chunk-based synthesis for long texts with real-time audio streaming
"""
import asyncio
import re
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from voicebot_orchestrator.zonos_tts import ZonosTTS
import io
import time
import logging

@dataclass
class StreamingConfig:
    """Configuration for streaming synthesis"""
    chunk_size: int = 200  # Characters per chunk
    overlap_words: int = 2  # Words to overlap between chunks
    max_concurrent: int = 3  # Maximum concurrent synthesis tasks
    buffer_size: int = 5  # Number of chunks to buffer ahead
    pause_detection: bool = True  # Insert pauses at sentence boundaries
    smart_chunking: bool = True  # Use intelligent text segmentation

@dataclass
class SynthesisChunk:
    """Individual synthesis chunk with metadata"""
    id: int
    text: str
    start_pos: int
    end_pos: int
    audio_bytes: Optional[bytes] = None
    duration_ms: Optional[int] = None
    status: str = "pending"  # pending, processing, completed, error
    error: Optional[str] = None

class TextChunker:
    """Intelligent text chunking for optimal synthesis"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        self.clause_breaks = re.compile(r'[,;:]\s+')
        
    def chunk_text(self, text: str) -> List[SynthesisChunk]:
        """
        Intelligently chunk text for optimal synthesis
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of SynthesisChunk objects
        """
        if not self.config.smart_chunking:
            return self._simple_chunk(text)
        
        chunks = []
        current_pos = 0
        chunk_id = 0
        
        # Split into paragraphs first
        paragraphs = self.paragraph_breaks.split(text)
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            # Process each paragraph
            paragraph_chunks = self._chunk_paragraph(paragraph, current_pos, chunk_id)
            chunks.extend(paragraph_chunks)
            
            # Update positions
            current_pos += len(paragraph) + 2  # +2 for paragraph break
            chunk_id = len(chunks)
        
        return chunks
    
    def _chunk_paragraph(self, paragraph: str, start_pos: int, start_id: int) -> List[SynthesisChunk]:
        """Chunk a single paragraph intelligently"""
        chunks = []
        current_pos = 0
        chunk_id = start_id
        
        # Split by sentences first
        sentences = self.sentence_endings.split(paragraph)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk + sentence) > self.config.chunk_size and current_chunk:
                # Create chunk from current content
                chunk = SynthesisChunk(
                    id=chunk_id,
                    text=current_chunk.strip(),
                    start_pos=start_pos + current_pos - len(current_chunk),
                    end_pos=start_pos + current_pos
                )
                chunks.append(chunk)
                chunk_id += 1
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
            
            current_pos += len(sentence) + 1
        
        # Add remaining content as final chunk
        if current_chunk.strip():
            chunk = SynthesisChunk(
                id=chunk_id,
                text=current_chunk.strip(),
                start_pos=start_pos + current_pos - len(current_chunk),
                end_pos=start_pos + current_pos
            )
            chunks.append(chunk)
        
        return chunks
    
    def _simple_chunk(self, text: str) -> List[SynthesisChunk]:
        """Simple character-based chunking fallback"""
        chunks = []
        chunk_size = self.config.chunk_size
        
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            chunk = SynthesisChunk(
                id=len(chunks),
                text=chunk_text,
                start_pos=i,
                end_pos=min(i + chunk_size, len(text))
            )
            chunks.append(chunk)
        
        return chunks

class StreamingTTSEngine:
    """
    Advanced streaming TTS engine for long-form content
    """
    
    def __init__(self, tts_engine: ZonosTTS, config: Optional[StreamingConfig] = None):
        self.tts_engine = tts_engine
        self.config = config or StreamingConfig()
        self.chunker = TextChunker(self.config)
        
        # Processing state
        self.active_streams: Dict[str, Dict] = {}
        self.synthesis_semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Statistics
        self.stats = {
            "total_chunks_processed": 0,
            "total_audio_generated": 0,
            "average_chunk_time": 0.0,
            "cache_hits": 0
        }
    
    async def stream_synthesis(
        self,
        text: str,
        stream_id: str,
        voice: str = "default",
        emotion: str = "neutral",
        speaking_style: str = "normal",
        speed: float = 1.0,
        pitch: float = 1.0,
        **synthesis_params
    ) -> AsyncGenerator[Tuple[SynthesisChunk, bytes], None]:
        """
        Stream synthesis for long text with real-time audio generation
        
        Args:
            text: Long text to synthesize
            stream_id: Unique identifier for this stream
            voice: Voice to use
            emotion: Emotion style
            speaking_style: Speaking style
            speed: Speech speed
            pitch: Pitch adjustment
            **synthesis_params: Additional synthesis parameters
            
        Yields:
            Tuple of (chunk_metadata, audio_bytes)
        """
        start_time = time.time()
        
        # Initialize stream state
        self.active_streams[stream_id] = {
            "start_time": start_time,
            "total_chunks": 0,
            "completed_chunks": 0,
            "status": "initializing"
        }
        
        try:
            # Chunk the text
            chunks = self.chunker.chunk_text(text)
            self.active_streams[stream_id]["total_chunks"] = len(chunks)
            self.active_streams[stream_id]["status"] = "processing"
            
            print(f"[STREAMING] Starting synthesis for {len(chunks)} chunks")
            print(f"            Text length: {len(text)} chars")
            print(f"            Stream ID: {stream_id}")
            
            # Process chunks with buffering
            chunk_buffer = []
            synthesis_tasks = []
            
            for i, chunk in enumerate(chunks):
                # Start synthesis task
                task = asyncio.create_task(
                    self._synthesize_chunk(
                        chunk, voice, emotion, speaking_style, 
                        speed, pitch, **synthesis_params
                    )
                )
                synthesis_tasks.append(task)
                
                # Manage buffer size
                if len(synthesis_tasks) >= self.config.buffer_size or i == len(chunks) - 1:
                    # Wait for oldest tasks to complete
                    completed_tasks = await asyncio.gather(*synthesis_tasks[:self.config.buffer_size])
                    
                    for completed_chunk in completed_tasks:
                        if completed_chunk.status == "completed":
                            self.active_streams[stream_id]["completed_chunks"] += 1
                            yield completed_chunk, completed_chunk.audio_bytes
                        else:
                            print(f"[WARNING] Chunk {completed_chunk.id} failed: {completed_chunk.error}")
                    
                    # Remove completed tasks
                    synthesis_tasks = synthesis_tasks[self.config.buffer_size:]
            
            # Process remaining tasks
            if synthesis_tasks:
                remaining_chunks = await asyncio.gather(*synthesis_tasks)
                for chunk in remaining_chunks:
                    if chunk.status == "completed":
                        self.active_streams[stream_id]["completed_chunks"] += 1
                        yield chunk, chunk.audio_bytes
            
            # Update statistics
            total_time = time.time() - start_time
            self.stats["total_chunks_processed"] += len(chunks)
            self.stats["average_chunk_time"] = (
                self.stats["average_chunk_time"] * 0.9 + (total_time / len(chunks)) * 0.1
            )
            
            self.active_streams[stream_id]["status"] = "completed"
            print(f"[OK] Streaming synthesis completed in {total_time:.2f}s")
            
        except Exception as e:
            self.active_streams[stream_id]["status"] = "error"
            print(f"[ERROR] Streaming synthesis failed: {e}")
            raise
        finally:
            # Cleanup
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    async def _synthesize_chunk(
        self,
        chunk: SynthesisChunk,
        voice: str,
        emotion: str,
        speaking_style: str,
        speed: float,
        pitch: float,
        **synthesis_params
    ) -> SynthesisChunk:
        """Synthesize a single chunk with concurrency control"""
        async with self.synthesis_semaphore:
            chunk.status = "processing"
            start_time = time.time()
            
            try:
                # Synthesize the chunk
                audio_bytes = await self.tts_engine.synthesize_speech(
                    text=chunk.text,
                    voice=voice,
                    emotion=emotion,
                    speaking_style=speaking_style,
                    speed=speed,
                    pitch=pitch,
                    **synthesis_params
                )
                
                chunk.audio_bytes = audio_bytes
                chunk.duration_ms = int((time.time() - start_time) * 1000)
                chunk.status = "completed"
                
                self.stats["total_audio_generated"] += len(audio_bytes)
                
            except Exception as e:
                chunk.status = "error"
                chunk.error = str(e)
                print(f"[ERROR] Chunk {chunk.id} synthesis failed: {e}")
            
            return chunk
    
    def get_stream_status(self, stream_id: str) -> Optional[Dict]:
        """Get status of an active stream"""
        return self.active_streams.get(stream_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return self.stats.copy()
    
    async def estimate_processing_time(self, text: str) -> Dict[str, float]:
        """Estimate processing time for text"""
        chunks = self.chunker.chunk_text(text)
        
        if self.stats["average_chunk_time"] > 0:
            estimated_time = len(chunks) * self.stats["average_chunk_time"]
        else:
            # Fallback estimation: ~2s per 100 characters
            estimated_time = len(text) * 0.02
        
        return {
            "estimated_seconds": estimated_time,
            "chunk_count": len(chunks),
            "text_length": len(text),
            "confidence": "high" if self.stats["total_chunks_processed"] > 10 else "low"
        }

# Async context manager for streaming
class StreamingSynthesisSession:
    """Context manager for streaming synthesis sessions"""
    
    def __init__(self, engine: StreamingTTSEngine, stream_id: str):
        self.engine = engine
        self.stream_id = stream_id
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup any remaining stream state
        if self.stream_id in self.engine.active_streams:
            del self.engine.active_streams[self.stream_id]
    
    async def synthesize_stream(self, text: str, **params):
        """Synthesize text as a stream"""
        async for chunk, audio in self.engine.stream_synthesis(
            text, self.stream_id, **params
        ):
            yield chunk, audio

# Factory function
async def create_streaming_engine(
    voice: str = "default",
    model: str = "zonos-v1",
    config: Optional[StreamingConfig] = None
) -> StreamingTTSEngine:
    """Create a streaming TTS engine with initialized TTS backend"""
    tts_engine = ZonosTTS(voice=voice, model=model)
    return StreamingTTSEngine(tts_engine, config)
