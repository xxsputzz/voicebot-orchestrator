#!/usr/bin/env python3
"""Simple endpoint test without importing service code."""

import requests
import time
import sys

def simple_test():
    """Test Zonos endpoints with minimal imports."""
    base_url = "http://localhost:8014"
    
    # Wait for service to be ready
    print("Waiting for service...")
    for i in range(10):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Service is ready!")
                break
        except:
            print(f"Attempt {i+1}/10: Service not ready yet...")
            time.sleep(1)
    else:
        print("❌ Service not responding")
        return
    
    # Test voices
    try:
        response = requests.get(f"{base_url}/voices", timeout=5)
        voices = response.json()
        print(f"Voices: {voices} (type: {type(voices)})")
        if isinstance(voices, list):
            print("✅ Voices format correct!")
        else:
            print("❌ Voices format incorrect!")
    except Exception as e:
        print(f"❌ Voices test failed: {e}")
    
    # Test models
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        models = response.json()
        print(f"Models: {models} (type: {type(models)})")
        if isinstance(models, list):
            print("✅ Models format correct!")
        else:
            print("❌ Models format incorrect!")
    except Exception as e:
        print(f"❌ Models test failed: {e}")

if __name__ == "__main__":
    simple_test()
