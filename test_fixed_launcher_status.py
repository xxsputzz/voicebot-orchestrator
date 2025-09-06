#!/usr/bin/env python3
"""
Test the fixed launcher's service status detection method
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
    "llm_mistral": {
        "name": "Mistral LLM Service",
        "type": "llm",
        "script": "aws_microservices\\ws_llm_mistral_service.py", 
        "process": None,
        "engine": "Mistral",
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
    },
    "tts_zonos": {
        "name": "Zonos TTS Service",
        "type": "tts",
        "script": "aws_microservices\\ws_tts_zonos_service.py",
        "process": None,
        "engine": "Zonos",
        "voices": "6",
        "performance": "Balanced efficiency"
    },
    "tts_tortoise": {
        "name": "Tortoise TTS Service",
        "type": "tts",
        "script": "aws_microservices\\ws_tts_tortoise_service.py",
        "process": None,
        "engine": "Tortoise",
        "voices": "29",
        "performance": "Premium quality"
    },
    "tts_dia": {
        "name": "Dia TTS Service",
        "type": "tts",
        "script": "aws_microservices\\ws_tts_dia_service.py",
        "process": None,
        "engine": "Dia",
        "voices": "10",
        "performance": "Premium balanced"
    }
}

def is_service_actually_running(service_id: str) -> tuple:
    """
    Check if a service is actually running by checking:
    1. Process is running (by checking all python processes)
    2. Service is registered in orchestrator
    Returns (process_running, orchestrator_registered)
    """
    service = services.get(service_id)
    if not service:
        return False, False
    
    # Map launcher service IDs to orchestrator service IDs
    orchestrator_service_id = service_id + "_ws"  # Most services add "_ws" suffix
    
    # Check if registered in orchestrator
    orchestrator_registered = False
    try:
        response = requests.get("http://localhost:8080/services", timeout=2)
        if response.status_code == 200:
            services_data = response.json()
            # The response is a direct array, not wrapped in 'value'
            if isinstance(services_data, list):
                registered_services = [s.get('service_id', '') for s in services_data if isinstance(s, dict)]
                if not registered_services:  # Try string format
                    registered_services = [s for s in services_data if isinstance(s, str)]
            else:
                # Fallback for wrapped format
                registered_services = [s.get('service_id', '') for s in services_data.get('value', [])]
            orchestrator_registered = orchestrator_service_id in registered_services
            print(f"  📡 Orchestrator check: {orchestrator_service_id} in {registered_services} = {orchestrator_registered}")
    except Exception as e:
        print(f"  ❌ Orchestrator check failed for {service_id}: {e}")
        pass
    
    # Check if process is running by looking for the service script
    process_running = False
    service_script = service.get('script', '')  # Fixed: use 'script' instead of 'file'
    if service_script:
        try:
            # Get just the filename for matching
            script_name = service_script.split('\\')[-1].split('/')[-1]
            print(f"  🔍 Looking for script: {script_name}")
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    # Check if the script name is in the command line
                    if script_name in cmdline:
                        process_running = True
                        print(f"  ✅ Process found: PID {proc.info['pid']}")
                        break
            if not process_running:
                print(f"  ❌ Process not found")
        except Exception as e:
            print(f"  ❌ Process check failed for {service_id}: {e}")
            pass
    
    return process_running, orchestrator_registered

# Test the status detection
print("🔍 Testing FIXED Service Status Detection")
print("=" * 60)

print("\n🔧 Local Service Processes:")

# Group services by type
service_types = {"stt": [], "llm": [], "tts": []}
for service_id, service in services.items():
    service_types[service["type"]].append((service_id, service))

for service_type, type_services in service_types.items():
    type_name = {"stt": "Speech-to-Text", "llm": "Language Models", "tts": "Text-to-Speech"}[service_type]
    print(f"\n{type_name.upper()}:")
    
    for service_id, service in type_services:
        print(f"\n--- Testing {service_id} ---")
        
        # Use the fixed service detection method - it returns (process_running, orchestrator_registered)
        process_running, orchestrator_registered = is_service_actually_running(service_id)
        
        if process_running:
            process_status = "✅ Running"
            if orchestrator_registered:
                registration_status = "✅ Registered"
            else:
                registration_status = "⏳ Connecting"
        else:
            process_status = "❌ Stopped"
            registration_status = "❌ Not Registered"
        
        print(f"  RESULT: {process_status} | {registration_status} | {service['name']}")
