#!/usr/bin/env python3
"""
Quick WebSocket Client Test

A simple test to verify the numbered menu and text messaging functionality.
"""

import asyncio
from ws_headset_client import HeadsetClient

async def test_numbered_menu():
    """Test the numbered menu functionality"""
    print("🧪 Testing WebSocket Client with Numbered Menu")
    print("=" * 50)
    
    client = HeadsetClient()
    
    try:
        # Connect
        await client.connect()
        
        # Test text input
        print("\n1️⃣ Testing text input...")
        await client.send_text_input("Hello WebSocket world! This is a test message.")
        
        # Wait for response
        print("⏳ Waiting 3 seconds for response...")
        await asyncio.sleep(3)
        
        # Test another message
        print("\n2️⃣ Testing second message...")
        await client.send_text_input("Can you hear me? Testing the streaming pipeline.")
        
        # Wait for response
        await asyncio.sleep(3)
        
        print("\n✅ Text messaging test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        await client.disconnect()
        print("👋 Test session ended")

if __name__ == "__main__":
    asyncio.run(test_numbered_menu())
