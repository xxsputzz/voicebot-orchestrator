#!/usr/bin/env python3
"""
Test script for the improved Tortoise TTS timeout system
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tortoise_timeout_config import get_timeout_manager

def test_timeout_calculations():
    """Test timeout calculations for various scenarios"""
    timeout_manager = get_timeout_manager()
    
    # Test scenarios based on your actual usage
    test_cases = [
        {"text": "Short test", "voice": "angie", "preset": "ultra_fast"},
        {"text": "This is a medium length text that should take a few minutes to process with ultra-high quality synthesis using the angie voice.", "voice": "angie", "preset": "ultra_fast"},
        {"text": "This is a very long text that represents the kind of content that was causing timeout issues. It contains multiple sentences and should require several minutes of processing time with the neural Tortoise TTS system. The timeout calculation should account for the complexity of the angie voice and the ultra-fast preset while providing adequate buffer time for completion.", "voice": "angie", "preset": "ultra_fast"},
    ]
    
    print("🔧 Tortoise TTS Timeout Calculator Test")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        text_length = len(case["text"])
        timeout = timeout_manager.calculate_timeout(
            text_length=text_length,
            voice=case["voice"],
            preset=case["preset"]
        )
        
        retry_timeouts = timeout_manager.get_retry_timeouts(timeout)
        
        print(f"\n🧪 Test Case {i}:")
        print(f"   Text: '{case['text'][:50]}{'...' if len(case['text']) > 50 else ''}'")
        print(f"   Length: {text_length} characters")
        print(f"   Voice: {case['voice']}")
        print(f"   Preset: {case['preset']}")
        print(f"   ⏰ Initial timeout: {timeout:.1f}s ({timeout/60:.1f} minutes)")
        print(f"   🔄 Retry timeouts: {[f'{t:.1f}s' for t in retry_timeouts]}")
        print(f"   📊 Character rate: {timeout/text_length:.1f}s per character")
    
    # Show configuration details
    print(f"\n⚙️ Current Configuration:")
    print(f"   Base overhead: {timeout_manager.config.base_overhead}s")
    print(f"   Char processing: {timeout_manager.config.char_processing_time}s per char")
    print(f"   Safety buffer: {timeout_manager.config.safety_buffer}x")
    print(f"   Min/Max timeout: {timeout_manager.config.min_timeout}s / {timeout_manager.config.max_timeout}s")
    
    # Voice complexity
    print(f"\n🎭 Voice Complexity Multipliers:")
    for voice, multiplier in sorted(timeout_manager.VOICE_COMPLEXITY.items()):
        if voice != 'default':
            print(f"   {voice}: {multiplier}x")
    
    # Performance history
    stats = timeout_manager.get_performance_stats()
    if stats:
        print(f"\n📈 Performance History:")
        for config, data in stats.items():
            print(f"   {config}: {data['samples']} samples, avg {data['avg_char_time']:.1f}s/char")
    else:
        print(f"\n📈 No performance history yet")

if __name__ == "__main__":
    test_timeout_calculations()
