#!/usr/bin/env python3
"""
Test Zonos TTS Interactive Option
=================================
Simple test to verify option 8 works in the interactive pipeline
"""

import asyncio
import sys
from pathlib import Path

# Add the tests directory to sys.path to import the test module
sys.path.insert(0, str(Path(__file__).parent / "tests"))

from test_interactive_pipeline import InteractivePipelineTester

async def test_zonos_option():
    """Test the Zonos TTS option specifically"""
    
    tester = InteractivePipelineTester()
    
    print("🧠 Testing Zonos TTS Option 8")
    print("=" * 40)
    
    # Test the Zonos TTS function directly
    success = await tester.test_direct_zonos_tts()
    
    if success:
        print("\n✅ Zonos TTS Option 8 test PASSED!")
    else:
        print("\n❌ Zonos TTS Option 8 test FAILED!")
    
    return success

if __name__ == "__main__":
    asyncio.run(test_zonos_option())
