#!/usr/bin/env python3
"""
Debug Service Detection
======================
"""
import requests
import psutil
import json

def debug_orchestrator_services():
    """Debug what the orchestrator actually has"""
    print("🔍 Debugging Orchestrator Services...")
    try:
        response = requests.get("http://localhost:8080/services", timeout=2)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            services_data = response.json()
            print(f"Response type: {type(services_data)}")
            print(f"Services found: {len(services_data) if isinstance(services_data, list) else 'Not a list'}")
            
            if isinstance(services_data, list):
                for i, service in enumerate(services_data):
                    print(f"  Service {i+1}: {service.get('service_id', 'UNKNOWN')} ({service.get('service_type', 'UNKNOWN')})")
            else:
                print(f"Unexpected format: {services_data}")
    except Exception as e:
        print(f"Error: {e}")

def debug_running_processes():
    """Debug what Python processes are running"""
    print("\n🔍 Debugging Running Python Processes...")
    try:
        count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                count += 1
                cmdline = ' '.join(proc.info['cmdline'])
                print(f"  PID {proc.info['pid']}: {cmdline}")
    except Exception as e:
        print(f"Error: {e}")
    print(f"Total Python processes: {count}")

def debug_service_matching():
    """Debug service script matching"""
    print("\n🔍 Debugging Service Script Matching...")
    
    # Test services from launcher config
    test_services = {
        'stt_whisper': 'aws_microservices\\ws_stt_whisper_service.py',
        'llm_gpt': 'aws_microservices\\ws_llm_gpt_service.py', 
        'tts_kokoro': 'aws_microservices\\ws_tts_kokoro_service.py'
    }
    
    for service_id, script_path in test_services.items():
        script_name = script_path.split('\\')[-1].split('/')[-1]
        print(f"\nChecking {service_id}:")
        print(f"  Script: {script_path}")
        print(f"  Script name: {script_name}")
        
        found = False
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if script_name in cmdline:
                        print(f"  ✅ FOUND: PID {proc.info['pid']}")
                        found = True
                        break
            
            if not found:
                print(f"  ❌ NOT FOUND")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    debug_orchestrator_services()
    debug_running_processes()
    debug_service_matching()
