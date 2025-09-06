#!/usr/bin/env python3
"""
Test Service ID Mapping
======================
"""
import requests

def test_service_mapping():
    """Test the service ID mapping logic"""
    print("🧪 Testing Service ID Mapping...")
    
    # Test mappings
    launcher_ids = ['stt_whisper', 'llm_gpt', 'tts_kokoro']
    
    # Get orchestrator services
    try:
        response = requests.get("http://localhost:8080/services", timeout=2)
        if response.status_code == 200:
            services_data = response.json()
            registered_services = [s.get('service_id', '') for s in services_data]
            print(f"Orchestrator services: {registered_services}")
            
            for launcher_id in launcher_ids:
                orchestrator_id = launcher_id + "_ws"
                is_registered = orchestrator_id in registered_services
                print(f"  {launcher_id} -> {orchestrator_id}: {'✅ REGISTERED' if is_registered else '❌ NOT FOUND'}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_service_mapping()
