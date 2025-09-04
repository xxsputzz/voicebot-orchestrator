#!/usr/bin/env python3
"""
Quick STT Test
=============
"""
import requests

try:
    # Health check
    health = requests.get('http://localhost:8003/health', timeout=5)
    print(f"Health: {health.status_code}")
    
    # Transcribe
    with open('benchmark_kokoro_1.wav', 'rb') as f:
        files = {'audio': f}
        response = requests.post('http://localhost:8003/transcribe', files=files, timeout=60)
    
    print(f"Transcribe: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Text: '{result.get('text', '')}'")
        
except Exception as e:
    print(f"Error: {e}")
