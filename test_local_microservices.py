#!/usr/bin/env python3
"""
Test script for local microservices setup
"""
import os
import sys
import time
import subprocess
import threading
import requests
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_service_health(name, url, timeout=30):
    """Test if a service is healthy"""
    print(f"⏳ Testing {name} service at {url}")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ {name} service is healthy!")
                return True
        except requests.exceptions.RequestException:
            time.sleep(2)
    
    print(f"❌ {name} service failed to start in {timeout}s")
    return False

def test_stt_service(base_url="http://localhost:8001"):
    """Test STT service with fake audio"""
    print("\n🎤 Testing STT Service")
    
    try:
        # Create a fake audio file for testing
        fake_audio = b"fake wav audio data"
        files = {"audio": ("test.wav", fake_audio, "audio/wav")}
        
        response = requests.post(f"{base_url}/transcribe", files=files, timeout=10)
        print(f"STT Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ STT Response: {result}")
            return True
        else:
            print(f"❌ STT Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ STT Test Failed: {e}")
        return False

def test_llm_service(base_url="http://localhost:8002"):
    """Test LLM service"""
    print("\n🧠 Testing LLM Service")
    
    try:
        payload = {
            "text": "Hello, how are you?",
            "model": "mistral"
        }
        
        response = requests.post(f"{base_url}/generate", json=payload, timeout=30)
        print(f"LLM Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ LLM Response: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ LLM Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ LLM Test Failed: {e}")
        return False

def test_tts_service(base_url="http://localhost:8003"):
    """Test TTS service"""
    print("\n🔊 Testing TTS Service")
    
    try:
        payload = {
            "text": "Hello, this is a test.",
            "engine": "pyttsx3"  # Use simpler engine for testing
        }
        
        response = requests.post(f"{base_url}/synthesize", json=payload, timeout=30)
        print(f"TTS Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ TTS Response: Audio generated ({len(result.get('audio', ''))} chars)")
            return True
        else:
            print(f"❌ TTS Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ TTS Test Failed: {e}")
        return False

def run_with_docker_compose():
    """Run tests using Docker Compose"""
    print("🐳 Starting services with Docker Compose...")
    
    # Start services
    subprocess.run([
        "docker-compose", "-f", "aws_microservices/docker-compose.local.yml", 
        "up", "-d"
    ], cwd=project_root)
    
    # Wait for services to start
    time.sleep(10)
    
    # Test services
    services = [
        ("STT", "http://localhost:8001"),
        ("LLM", "http://localhost:8002"), 
        ("TTS", "http://localhost:8003")
    ]
    
    health_results = []
    for name, url in services:
        health_results.append(test_service_health(name, url))
    
    if all(health_results):
        print("\n🧪 Running service tests...")
        test_stt_service()
        test_llm_service()
        test_tts_service()
    
    # Stop services
    print("\n🛑 Stopping services...")
    subprocess.run([
        "docker-compose", "-f", "aws_microservices/docker-compose.local.yml", 
        "down"
    ], cwd=project_root)

def run_with_python_runner():
    """Run tests using Python runner"""
    print("🐍 Starting services with Python runner...")
    
    # Start local runner in background
    runner_process = subprocess.Popen([
        sys.executable, "aws_microservices/local_runner.py"
    ], cwd=project_root)
    
    try:
        # Wait for services to start
        time.sleep(15)
        
        # Test services
        services = [
            ("STT", "http://localhost:8001"),
            ("LLM", "http://localhost:8002"), 
            ("TTS", "http://localhost:8003")
        ]
        
        health_results = []
        for name, url in services:
            health_results.append(test_service_health(name, url))
        
        if all(health_results):
            print("\n🧪 Running service tests...")
            test_stt_service()
            test_llm_service() 
            test_tts_service()
        
    finally:
        # Stop runner
        print("\n🛑 Stopping Python runner...")
        runner_process.terminate()
        runner_process.wait()

def test_orchestrator_client():
    """Test the orchestrator client"""
    print("\n🎭 Testing Orchestrator Client...")
    
    try:
        # Import after ensuring path is set
        from aws_microservices.orchestrator_client import example_voice_conversation
        import asyncio
        
        # Run the example
        asyncio.run(example_voice_conversation())
        print("✅ Orchestrator test completed!")
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")

def main():
    """Main test runner"""
    print("🧪 Local Microservices Test Suite")
    print("=" * 50)
    
    # Check if Docker is available
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        docker_available = True
        print("✅ Docker is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        docker_available = False
        print("❌ Docker not available")
    
    print("\nChoose testing method:")
    print("1. Docker Compose (recommended)")
    print("2. Python runner")
    print("3. Test orchestrator client only")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1" and docker_available:
        run_with_docker_compose()
    elif choice == "2":
        run_with_python_runner()
    elif choice == "3":
        test_orchestrator_client()
    elif choice == "4":
        print("👋 Goodbye!")
        return
    else:
        if choice == "1" and not docker_available:
            print("❌ Docker not available, falling back to Python runner")
            run_with_python_runner()
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
