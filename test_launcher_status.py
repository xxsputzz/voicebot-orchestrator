#!/usr/bin/env python3
"""
Test the launcher's service status detection
"""
import sys
import psutil
import requests

# Copy the service definitions from the launcher
services = {
    "stt_whisper": {
        "name": "Whisper STT Service",
        "type": "stt", 
        "script": "aws_microservices\\ws_stt_whisper_service.py",
        "process": None,
        "engine": "OpenAI Whisper",
        "performance": "High accuracy, moderate speed"
    },
    "llm_gpt": {
        "name": "GPT LLM Service",
        "type": "llm",
        "script": "aws_microservices\\ws_llm_gpt_service.py", 
        "process": None,
        "engine": "OpenAI GPT",
        "performance": "High quality responses"
    },
    "tts_kokoro": {
        "name": "Kokoro TTS Service",
        "type": "tts",
        "script": "aws_microservices\\ws_tts_kokoro_service.py",
        "process": None,
        "engine": "Kokoro",
        "voices": "9",
        "performance": "Fast, natural-sounding"
    }
}

def is_service_actually_running(service_id):
    """Check if service is actually running by looking for process"""
    service = services.get(service_id)
    if not service:
        return False
    
    script_path = service["script"]
    script_name = script_path.split("\\")[-1]  # Get filename from path
    
    print(f"🔍 Checking {service_id}:")
    print(f"  Script: {script_path}")
    print(f"  Script name: {script_name}")
    
    # Check if any python process is running this script
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.info['name'] == 'python.exe':
            try:
                cmdline = ' '.join(process.info['cmdline'])
                if script_name in cmdline:
                    print(f"  ✅ FOUND: PID {process.info['pid']}")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    
    print(f"  ❌ NOT FOUND")
    return False

def check_orchestrator_health():
    """Check if orchestrator is accessible"""
    try:
        # Try the services endpoint instead of health
        response = requests.get('http://localhost:9000/services', timeout=2)
        return response.status_code == 200
    except Exception as e:
        print(f"Orchestrator error: {e}")
        return False

def get_service_status():
    """Get service status from orchestrator"""
    try:
        response = requests.get('http://localhost:9000/services', timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data
    except Exception as e:
        print(f"Service status error: {e}")
    return {'services': [], 'count': 0}

# Test the status detection
print("🔍 Testing Service Status Detection")
print("=" * 50)

# Check orchestrator
orchestrator_healthy = check_orchestrator_health()
status_icon = "✅" if orchestrator_healthy else "❌"
print(f"{status_icon} Orchestrator: {'Healthy' if orchestrator_healthy else 'Not Available'}")

# Get registered services
registered_services = []
if orchestrator_healthy:
    service_status = get_service_status()
    services_data = service_status.get('services', [])
    print(f"📡 Services from orchestrator: {services_data}")
    if isinstance(services_data, list):
        for s in services_data:
            if isinstance(s, dict):
                registered_services.append(s.get('service_id', 'unknown'))
            elif isinstance(s, str):
                registered_services.append(s)
    print(f"📡 Registered Services: {registered_services}")

print("\n🔧 Local Service Processes:")

# Test each service
for service_id, service in services.items():
    print(f"\n--- Testing {service_id} ---")
    
    # Use the improved service detection method
    is_running = is_service_actually_running(service_id)
    if is_running:
        process_status = "✅ Running"
        # Map service ID for orchestrator check
        orchestrator_id = service_id + "_ws"
        if orchestrator_id in registered_services:
            registration_status = "✅ Registered"
        else:
            registration_status = "⏳ Connecting"
    else:
        process_status = "❌ Stopped"
        registration_status = "❌ Not Registered"
    
    print(f"  {process_status} | {registration_status} | {service['name']}")
