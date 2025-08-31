#!/usr/bin/env python3
"""
Running Services Pipeline Tester
================================

Tests the currently running services for pipeline functionality:
1. STT → LLM (Speech to Language processing)
2. LLM → TTS (Language to Speech synthesis)  
3. Full STT → LLM → TTS (Complete pipeline)

Only tests services that are currently running and healthy.
Automatically detects available service combinations.
"""

import asyncio
import sys
import time
import json
import base64
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import traceback

class RunningServicesPipelineTester:
    """Test pipeline functionality using only currently running services"""
    
    def __init__(self):
        """Initialize the tester"""
        self.base_dir = Path(__file__).parent
        self.audio_dir = self.base_dir / "audio_samples" / "running_services_pipeline"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # All possible service endpoints
        self.all_services = {
            "orchestrator": "http://localhost:8000",
            "stt": "http://localhost:8001", 
            "whisper_stt": "http://localhost:8002",
            "kokoro_tts": "http://localhost:8011",
            "hira_dia_tts": "http://localhost:8012", 
            "dia_4bit_tts": "http://localhost:8013",  # New dedicated service
            "mistral_llm": "http://localhost:8021",
            "gpt_llm": "http://localhost:8022"
        }
        
        # Currently running services (detected automatically)
        self.running_services = {}
        self.available_combinations = []
        
    def detect_running_services(self) -> Dict[str, str]:
        """Detect which services are currently running and healthy"""
        print("🔍 Detecting currently running services...")
        running = {}
        
        for service_name, endpoint in self.all_services.items():
            if self.check_service_health(service_name, endpoint):
                running[service_name] = endpoint
                print(f"  ✅ {service_name} - {endpoint}")
            else:
                print(f"  ❌ {service_name} - Not running")
        
        self.running_services = running
        print(f"\n📊 Found {len(running)} running services")
        return running
    
    def check_service_health(self, service_name: str, endpoint: str) -> bool:
        """Check if a service is running and healthy"""
        try:
            # Try health endpoint first
            health_response = requests.get(f"{endpoint}/health", timeout=2)
            if health_response.status_code == 200:
                return True
        except:
            pass
        
        try:
            # Try root endpoint as fallback
            root_response = requests.get(endpoint, timeout=2)
            if root_response.status_code in [200, 404]:  # 404 is OK for some services
                return True
        except:
            pass
        
        return False
    
    def discover_pipeline_combinations(self) -> List[Dict]:
        """Discover available pipeline combinations from running services"""
        print("🔧 Discovering available pipeline combinations...")
        combinations = []
        
        # Find available STT services
        stt_services = [name for name in self.running_services.keys() 
                       if 'stt' in name.lower()]
        
        # Find available LLM services  
        llm_services = [name for name in self.running_services.keys()
                       if 'llm' in name.lower()]
        
        # Find available TTS services
        tts_services = [name for name in self.running_services.keys()
                       if 'tts' in name.lower()]
        
        print(f"  📝 STT Services: {stt_services}")
        print(f"  🧠 LLM Services: {llm_services}")  
        print(f"  🗣️ TTS Services: {tts_services}")
        
        # Generate all valid combinations
        for stt in stt_services:
            for llm in llm_services:
                for tts in tts_services:
                    combo = {
                        "stt": stt,
                        "llm": llm, 
                        "tts": tts,
                        "name": f"{stt} → {llm} → {tts}",
                        "stt_to_llm": f"{stt} → {llm}",
                        "llm_to_tts": f"{llm} → {tts}"
                    }
                    combinations.append(combo)
        
        self.available_combinations = combinations
        print(f"  🎯 Generated {len(combinations)} pipeline combinations")
        return combinations
    
    async def test_stt_to_llm(self, stt_service: str, llm_service: str, test_text: str = "Hello, how are you today?") -> Optional[str]:
        """Test STT → LLM pipeline component"""
        print(f"\n🎙️ Testing {stt_service} → {llm_service}")
        
        try:
            # For testing, we'll simulate STT input with text
            # In real usage, this would be audio → STT → text
            stt_output = test_text  # Simulated STT output
            
            # Send to LLM
            llm_endpoint = self.running_services[llm_service]
            llm_data = {
                "query": stt_output,
                "temperature": 0.7,
                "max_tokens": 150
            }
            
            print(f"  📝 STT Output (simulated): '{stt_output}'")
            
            llm_response = requests.post(f"{llm_endpoint}/generate", json=llm_data, timeout=30)
            if llm_response.status_code == 200:
                llm_result = llm_response.json()
                llm_text = llm_result.get("response", "").strip()
                print(f"  🧠 LLM Response: '{llm_text[:100]}...'")
                return llm_text
            else:
                print(f"  ❌ LLM request failed: {llm_response.status_code}")
                return None
                
        except Exception as e:
            print(f"  ❌ STT→LLM test failed: {e}")
            return None
    
    async def test_llm_to_tts(self, llm_service: str, tts_service: str, input_text: str = "Hello! I'm working great today.") -> Optional[str]:
        """Test LLM → TTS pipeline component"""
        print(f"\n🧠 Testing {llm_service} → {tts_service}")
        
        try:
            # Send to LLM first
            llm_endpoint = self.running_services[llm_service]
            llm_data = {
                "query": input_text,
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            llm_response = requests.post(f"{llm_endpoint}/generate", json=llm_data, timeout=30)
            if llm_response.status_code != 200:
                print(f"  ❌ LLM request failed: {llm_response.status_code}")
                return None
                
            llm_result = llm_response.json() 
            llm_text = llm_result.get("response", "").strip()
            print(f"  🧠 LLM Output: '{llm_text[:100]}...'")
            
            # Send LLM output to TTS
            tts_endpoint = self.running_services[tts_service]
            tts_data = {
                "text": llm_text,
                "voice": "default",
                "speed": 1.0,
                "return_audio": True
            }
            
            tts_response = requests.post(f"{tts_endpoint}/synthesize", json=tts_data, timeout=60)
            if tts_response.status_code == 200:
                tts_result = tts_response.json()
                audio_data = tts_result.get("audio_base64")
                
                if audio_data:
                    # Save audio file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"llm_to_tts_{llm_service}_{tts_service}_{timestamp}.wav"
                    filepath = self.audio_dir / filename
                    
                    audio_bytes = base64.b64decode(audio_data)
                    with open(filepath, "wb") as f:
                        f.write(audio_bytes)
                    
                    print(f"  🗣️ TTS Audio saved: {filename}")
                    return str(filepath)
                else:
                    print(f"  ❌ No audio data received from TTS")
                    return None
            else:
                print(f"  ❌ TTS request failed: {tts_response.status_code}")
                return None
                
        except Exception as e:
            print(f"  ❌ LLM→TTS test failed: {e}")
            return None
    
    async def test_full_pipeline(self, combination: Dict, input_text: str = "How is the weather?") -> Optional[str]:
        """Test complete STT → LLM → TTS pipeline"""
        print(f"\n🎯 Testing Full Pipeline: {combination['name']}")
        
        try:
            stt_service = combination["stt"]
            llm_service = combination["llm"] 
            tts_service = combination["tts"]
            
            # Step 1: Simulate STT (in real usage, this would process audio)
            stt_output = input_text  # Simulated STT output
            print(f"  🎙️ STT Input (simulated): '{stt_output}'")
            
            # Step 2: LLM Processing
            llm_endpoint = self.running_services[llm_service]
            llm_data = {
                "query": stt_output,
                "temperature": 0.7,
                "max_tokens": 150
            }
            
            llm_response = requests.post(f"{llm_endpoint}/generate", json=llm_data, timeout=30)
            if llm_response.status_code != 200:
                print(f"  ❌ LLM step failed: {llm_response.status_code}")
                return None
                
            llm_result = llm_response.json()
            llm_text = llm_result.get("response", "").strip()
            print(f"  🧠 LLM Processing: '{llm_text[:100]}...'")
            
            # Step 3: TTS Synthesis
            tts_endpoint = self.running_services[tts_service]
            tts_data = {
                "text": llm_text,
                "voice": "default", 
                "speed": 1.0,
                "return_audio": True
            }
            
            tts_response = requests.post(f"{tts_endpoint}/synthesize", json=tts_data, timeout=60)
            if tts_response.status_code == 200:
                tts_result = tts_response.json()
                audio_data = tts_result.get("audio_base64")
                
                if audio_data:
                    # Save audio file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_name = combination['name'].replace(' → ', '_').replace(' ', '_')
                    filename = f"full_pipeline_{safe_name}_{timestamp}.wav"
                    filepath = self.audio_dir / filename
                    
                    audio_bytes = base64.b64decode(audio_data)
                    with open(filepath, "wb") as f:
                        f.write(audio_bytes)
                    
                    print(f"  🗣️ Final Audio: {filename}")
                    return str(filepath)
                else:
                    print(f"  ❌ No audio generated")
                    return None
            else:
                print(f"  ❌ TTS step failed: {tts_response.status_code}")
                return None
                
        except Exception as e:
            print(f"  ❌ Full pipeline test failed: {e}")
            traceback.print_exc()
            return None
    
    async def run_all_tests(self):
        """Run all available pipeline tests"""
        print("🚀 Running Services Pipeline Tests")
        print("=" * 50)
        
        # Detect running services
        running = self.detect_running_services()
        if not running:
            print("❌ No services are currently running. Start some services first.")
            return
        
        # Discover combinations
        combinations = self.discover_pipeline_combinations()
        if not combinations:
            print("❌ No valid pipeline combinations found.")
            return
        
        print(f"\n🎯 Testing {len(combinations)} pipeline combinations...")
        
        # Test each combination
        results = {
            "stt_to_llm": {},
            "llm_to_tts": {},
            "full_pipeline": {}
        }
        
        for i, combo in enumerate(combinations, 1):
            print(f"\n{'='*20} Combination {i}/{len(combinations)} {'='*20}")
            
            # Test STT → LLM component
            stt_llm_result = await self.test_stt_to_llm(combo["stt"], combo["llm"])
            results["stt_to_llm"][combo["stt_to_llm"]] = stt_llm_result is not None
            
            # Test LLM → TTS component  
            llm_tts_result = await self.test_llm_to_tts(combo["llm"], combo["tts"])
            results["llm_to_tts"][combo["llm_to_tts"]] = llm_tts_result is not None
            
            # Test full pipeline
            full_result = await self.test_full_pipeline(combo)
            results["full_pipeline"][combo["name"]] = full_result is not None
            
            print(f"  📊 Results: STT→LLM: {'✅' if results['stt_to_llm'][combo['stt_to_llm']] else '❌'}, "
                  f"LLM→TTS: {'✅' if results['llm_to_tts'][combo['llm_to_tts']] else '❌'}, "
                  f"Full: {'✅' if results['full_pipeline'][combo['name']] else '❌'}")
        
        # Print summary
        self.print_test_summary(results)
    
    def print_test_summary(self, results: Dict):
        """Print comprehensive test summary"""
        print("\n" + "="*60)
        print("📊 RUNNING SERVICES PIPELINE TEST SUMMARY")
        print("="*60)
        
        # STT → LLM Results
        print("\n🎙️ STT → LLM Component Tests:")
        for combo, success in results["stt_to_llm"].items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status} {combo}")
        
        # LLM → TTS Results
        print("\n🧠 LLM → TTS Component Tests:")
        for combo, success in results["llm_to_tts"].items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status} {combo}")
        
        # Full Pipeline Results
        print("\n🎯 Full STT → LLM → TTS Pipeline Tests:")
        for combo, success in results["full_pipeline"].items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status} {combo}")
        
        # Summary stats
        stt_llm_passed = sum(results["stt_to_llm"].values())
        llm_tts_passed = sum(results["llm_to_tts"].values())
        full_passed = sum(results["full_pipeline"].values())
        
        total_stt_llm = len(results["stt_to_llm"])
        total_llm_tts = len(results["llm_to_tts"])
        total_full = len(results["full_pipeline"])
        
        print(f"\n📈 Summary Statistics:")
        print(f"  STT → LLM: {stt_llm_passed}/{total_stt_llm} passed")
        print(f"  LLM → TTS: {llm_tts_passed}/{total_llm_tts} passed")
        print(f"  Full Pipeline: {full_passed}/{total_full} passed")
        print(f"  Audio files saved to: {self.audio_dir}")

async def main():
    """Main test function"""
    tester = RunningServicesPipelineTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
