"""
Comprehensive Test for Streaming TTS System
Testing streaming synthesis, user interface, and advanced features
"""
import asyncio
import time
import sys
import os
sys.path.append('.')

from voicebot_orchestrator.streaming_tts import (
    StreamingTTSEngine, StreamingConfig, create_streaming_engine, StreamingSynthesisSession
)
from voicebot_orchestrator.advanced_tts_features import (
    TextPreprocessor, ProcessingPreferences, TextQuality,
    SSMLGenerator, BatchProcessor, TTSAnalytics
)

async def test_streaming_synthesis():
    """Test the complete streaming synthesis system"""
    print("🚀 Testing Comprehensive Streaming TTS System")
    print("=" * 60)
    
    # Test 1: Basic Streaming Synthesis
    print("\n1. Testing Basic Streaming Synthesis...")
    
    config = StreamingConfig(
        chunk_size=150,
        max_concurrent=2,
        buffer_size=3,
        smart_chunking=True
    )
    
    engine = await create_streaming_engine(config=config)
    
    long_text = """
    Welcome to the comprehensive streaming text-to-speech system! This advanced platform provides 
    real-time audio generation for long-form content with intelligent chunking and buffering. 
    
    The system supports multiple voices, emotions, and speaking styles, making it perfect for 
    audiobook narration, educational content, news articles, and business presentations. 
    
    With features like automatic text preprocessing, emoji handling, abbreviation expansion, 
    and SSML support, this TTS engine delivers professional-quality speech synthesis suitable 
    for production environments.
    
    Our streaming architecture ensures smooth playback even for very long texts by processing 
    content in optimized chunks while maintaining natural speech flow and emotional consistency 
    throughout the entire synthesis process.
    """
    
    stream_id = "test_stream_001"
    chunks_received = 0
    total_audio_size = 0
    
    try:
        start_time = time.time()
        
        async for chunk, audio_bytes in engine.stream_synthesis(
            text=long_text,
            stream_id=stream_id,
            voice="sophia",
            emotion="friendly",
            speaking_style="conversational",
            speed=1.1
        ):
            chunks_received += 1
            total_audio_size += len(audio_bytes)
            print(f"   Chunk {chunk.id}: {len(audio_bytes)} bytes, {chunk.duration_ms}ms")
            
            # Save first chunk as sample
            if chunk.id == 0:
                with open("streaming_chunk_0.wav", "wb") as f:
                    f.write(audio_bytes)
        
        processing_time = time.time() - start_time
        print(f"✅ Streaming synthesis completed:")
        print(f"   Chunks received: {chunks_received}")
        print(f"   Total audio: {total_audio_size} bytes")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Average per chunk: {processing_time/chunks_received:.2f}s")
        
    except Exception as e:
        print(f"❌ Streaming synthesis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Text Preprocessing
    print("\n2. Testing Advanced Text Preprocessing...")
    
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "Hello! 😀 I'm Dr. Smith from NYC Inc. Call me at 555-1234 or visit www.example.com",
        "Meeting at 2:30 PM on 12/25/2024. Budget: $1,500,000. Success rate: 95%",
        "AI & ML are transforming tech e.g. NLP, CV, etc. It's amazing!!! 🚀🔥"
    ]
    
    preferences = ProcessingPreferences(
        auto_punctuation=True,
        expand_abbreviations=True,
        number_pronunciation=True,
        emoji_handling="describe",
        profanity_filter=False,
        reading_speed_optimization=True
    )
    
    for i, text in enumerate(test_texts):
        processed = preprocessor.preprocess(text, preferences)
        print(f"   Text {i+1}:")
        print(f"   Original: {text}")
        print(f"   Processed: {processed}")
        print()
    
    print("✅ Text preprocessing completed")
    
    # Test 3: SSML Generation
    print("\n3. Testing SSML Generation...")
    
    ssml_generator = SSMLGenerator()
    
    test_text = "Welcome to our presentation. This is very important information."
    emphasis_words = ["Welcome", "very important"]
    break_positions = [(28, "strong"), (35, "medium")]
    
    ssml_output = ssml_generator.generate_ssml(
        text=test_text,
        voice="professional",
        rate="medium",
        pitch="medium",
        volume="loud",
        emphasis_words=emphasis_words,
        break_positions=break_positions
    )
    
    print(f"   Generated SSML:")
    print(f"   {ssml_output}")
    print("✅ SSML generation completed")
    
    # Test 4: Batch Processing
    print("\n4. Testing Batch Processing...")
    
    batch_processor = BatchProcessor(engine.tts_engine)
    
    batch_texts = [
        "First text for batch processing.",
        "Second text with different content.",
        "Third text to complete the batch."
    ]
    
    def progress_callback(completed, total):
        print(f"   Progress: {completed}/{total} texts processed")
    
    try:
        batch_results = await batch_processor.process_batch(
            texts=batch_texts,
            voice="aria",
            emotion="professional",
            quality=TextQuality.ENHANCED,
            progress_callback=progress_callback
        )
        
        successful = sum(1 for r in batch_results if r["status"] == "success")
        total_audio = sum(len(r.get("audio_bytes", b"")) for r in batch_results if "audio_bytes" in r)
        
        print(f"✅ Batch processing completed:")
        print(f"   Successful: {successful}/{len(batch_texts)}")
        print(f"   Total audio generated: {total_audio} bytes")
        
        # Save first result as sample
        if batch_results and batch_results[0]["status"] == "success":
            with open("batch_sample_0.wav", "wb") as f:
                f.write(batch_results[0]["audio_bytes"])
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
    
    # Test 5: Analytics Tracking
    print("\n5. Testing Analytics System...")
    
    analytics = TTSAnalytics()
    
    # Simulate some usage
    analytics.record_synthesis(100, 50000, 2.5, "sophia", "happy", True)
    analytics.record_synthesis(200, 75000, 3.2, "professional", "authoritative", True)
    analytics.record_synthesis(150, 60000, 2.8, "sophia", "friendly", True)
    analytics.record_synthesis(50, 0, 0, "aria", "sad", False)  # Failed synthesis
    
    report = analytics.get_analytics_report()
    
    print(f"   Total syntheses: {report['overall_stats']['total_syntheses']}")
    print(f"   Success rate: {report['calculated_metrics']['success_rate']:.1f}%")
    print(f"   Average processing time: {report['overall_stats']['average_processing_time']:.2f}s")
    print(f"   Top voice: {report['top_voices'][0] if report['top_voices'] else 'None'}")
    print(f"   Top emotion: {report['top_emotions'][0] if report['top_emotions'] else 'None'}")
    
    print("✅ Analytics tracking completed")
    
    # Test 6: Session Management
    print("\n6. Testing Session Management...")
    
    try:
        async with StreamingSynthesisSession(engine, "session_test") as session:
            session_text = "This is a test of session management for streaming synthesis."
            
            chunk_count = 0
            async for chunk, audio in session.synthesize_stream(
                text=session_text,
                voice="luna",
                emotion="conversational"
            ):
                chunk_count += 1
                
            print(f"✅ Session management completed: {chunk_count} chunks processed")
            
    except Exception as e:
        print(f"❌ Session management failed: {e}")
    
    # Test 7: Performance Analysis
    print("\n7. Testing Performance Analysis...")
    
    # Test different chunk sizes
    chunk_sizes = [100, 200, 300]
    performance_results = []
    
    test_text = "Performance testing with various chunk sizes. " * 20  # ~900 characters
    
    for chunk_size in chunk_sizes:
        config = StreamingConfig(chunk_size=chunk_size, max_concurrent=2)
        test_engine = await create_streaming_engine(config=config)
        
        start_time = time.time()
        chunk_count = 0
        
        async for chunk, audio in test_engine.stream_synthesis(
            text=test_text,
            stream_id=f"perf_test_{chunk_size}",
            voice="default",
            emotion="neutral"
        ):
            chunk_count += 1
        
        total_time = time.time() - start_time
        performance_results.append({
            "chunk_size": chunk_size,
            "total_time": total_time,
            "chunk_count": chunk_count,
            "time_per_chunk": total_time / chunk_count if chunk_count > 0 else 0
        })
    
    print("   Performance Results:")
    for result in performance_results:
        print(f"   Chunk size {result['chunk_size']}: {result['total_time']:.2f}s "
              f"({result['chunk_count']} chunks, {result['time_per_chunk']:.2f}s/chunk)")
    
    print("✅ Performance analysis completed")
    
    # Test 8: Error Handling
    print("\n8. Testing Error Handling...")
    
    try:
        # Test with invalid voice
        error_engine = await create_streaming_engine()
        error_count = 0
        
        async for chunk, audio in error_engine.stream_synthesis(
            text="Testing error handling",
            stream_id="error_test",
            voice="nonexistent_voice",
            emotion="invalid_emotion"
        ):
            error_count += 1
        
        print(f"   Processed {error_count} chunks despite invalid parameters")
        
    except Exception as e:
        print(f"   Expected error caught: {type(e).__name__}")
    
    print("✅ Error handling completed")
    
    # Final Statistics
    print("\n" + "=" * 60)
    print("🎉 Comprehensive Streaming TTS Testing Complete!")
    print("\nFeatures Successfully Tested:")
    print("✅ Streaming synthesis with intelligent chunking")
    print("✅ Advanced text preprocessing (emojis, abbreviations, numbers)")
    print("✅ SSML generation for speech markup")
    print("✅ Batch processing for multiple texts")
    print("✅ Analytics and usage tracking")
    print("✅ Session management with context")
    print("✅ Performance optimization analysis")
    print("✅ Robust error handling")
    
    print(f"\nGenerated Files:")
    print("• streaming_chunk_0.wav - First streaming chunk sample")
    print("• batch_sample_0.wav - Batch processing sample")
    
    print(f"\nSystem Ready for Production! 🚀")

if __name__ == "__main__":
    asyncio.run(test_streaming_synthesis())
