#!/usr/bin/env python3
"""
Simple test to check Tortoise TTS voices and functionality
"""

import requests
import json

def test_voices():
    """Test voice listing"""
    try:
        response = requests.get("http://localhost:8015/voices", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['total']} voices:")
            for i, voice in enumerate(data['voices'], 1):
                print(f"  {i:2d}. {voice}")
            return True
        else:
            print(f"❌ Error getting voices: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health():
    """Test service health"""
    try:
        response = requests.get("http://localhost:8015/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service healthy: {data['voice_count']} voices, ready={data['ready']}")
            return True
        else:
            print(f"❌ Service unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    print("🐢 Testing Tortoise TTS Service")
    print("=" * 50)
    
    print("\n1. Health Check:")
    test_health()
    
    print("\n2. Voice List:")
    test_voices()
    
    print("\n✅ Test complete!")
