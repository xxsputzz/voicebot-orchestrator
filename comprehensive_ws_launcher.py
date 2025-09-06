#!/usr/bin/env python3
"""
Comprehensive WebSocket Services Launcher - All Converted Services
Launch and manage all converted WebSocket services for streaming pipeline with combo presets
"""

import asyncio
import json
import logging
import sys
import os
import time
import subprocess
import psutil
import signal
from typing import Dict, Any, List, Optional
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ComprehensiveWebSocketLauncher:
    """Complete launcher for all converted WebSocket services with combo presets"""
    
    def __init__(self):
        # All available WebSocket services
        self.services = {
            # STT Services
            "stt_whisper": {
                "name": "WebSocket Whisper STT",
                "file": "aws_microservices/ws_stt_whisper_service.py",
                "process": None,
                "type": "stt",
                "description": "Speech-to-Text with Whisper model (OpenAI)",
                "engine": "whisper",
                "performance": "high_accuracy",
                "latency": "medium"
            },
            
            # LLM Services  
            "llm_gpt": {
                "name": "WebSocket GPT LLM",
                "file": "aws_microservices/ws_llm_gpt_service.py", 
                "process": None,
                "type": "llm",
                "description": "Language Model with GPT/OpenAI API",
                "engine": "gpt",
                "performance": "high_quality",
                "latency": "fast"
            },
            "llm_mistral": {
                "name": "WebSocket Mistral LLM",
                "file": "aws_microservices/ws_llm_mistral_service.py",
                "process": None,
                "type": "llm", 
                "description": "Language Model with Mistral 7B (Local)",
                "engine": "mistral",
                "performance": "high_quality",
                "latency": "medium"
            },
            
            # TTS Services - Reordered: Kokoro, Zonos, Tortoise, Dia
            "tts_kokoro": {
                "name": "WebSocket Kokoro TTS",
                "file": "aws_microservices/ws_tts_kokoro_service.py",
                "process": None,
                "type": "tts",
                "description": "Text-to-Speech with Kokoro (9 voices, fastest speed)",
                "engine": "kokoro",
                "performance": "high_speed",
                "latency": "fast",
                "voices": 9
            },
            "tts_zonos": {
                "name": "WebSocket Zonos TTS",
                "file": "aws_microservices/ws_tts_zonos_service.py",
                "process": None,
                "type": "tts",
                "description": "Text-to-Speech with Zonos (6 voices, balanced efficiency)",
                "engine": "zonos",
                "performance": "balanced_efficient",
                "latency": "medium",
                "voices": 6
            },
            "tts_tortoise": {
                "name": "WebSocket Tortoise TTS",
                "file": "aws_microservices/ws_tts_tortoise_service.py",
                "process": None,
                "type": "tts",
                "description": "Text-to-Speech with Tortoise (29 voices, highest quality)",
                "engine": "tortoise",
                "performance": "premium_quality",
                "latency": "slow",
                "voices": 29
            },
            "tts_dia": {
                "name": "WebSocket Dia TTS", 
                "file": "aws_microservices/ws_tts_dia_service.py",
                "process": None,
                "type": "tts",
                "description": "Text-to-Speech with Dia (10 voices, premium quality)",
                "engine": "dia",
                "performance": "premium_balanced",
                "latency": "medium",
                "voices": 10
            }
        }
        
        # Predefined service combination presets - Following your specific patterns
        self.combos = {
            "fast_combo": {
                "name": "🚀 Fast Combo",
                "description": "Orchestrator + Whisper STT + Mistral LLM + Kokoro TTS",
                "services": ["stt_whisper", "llm_mistral", "tts_kokoro"],
                "use_case": "Fast local processing with speed optimization",
                "total_latency": "~800ms",
                "pros": ["Local LLM privacy", "Fastest TTS", "Low latency"],
                "cons": ["Limited TTS voices", "GPU intensive for Mistral"]
            },
            
            "efficient_combo": {
                "name": "⚖️ Efficient Combo", 
                "description": "Orchestrator + Whisper STT + Mistral LLM + Zonos TTS",
                "services": ["stt_whisper", "llm_mistral", "tts_zonos"],
                "use_case": "Balanced local processing with good efficiency",
                "total_latency": "~1.2s",
                "pros": ["Local privacy", "Balanced quality/speed", "6 efficient voices"],
                "cons": ["GPU usage for Mistral", "Medium latency"]
            },
            
            "balanced_combo": {
                "name": "⚡ Balanced Combo",
                "description": "Orchestrator + Whisper STT + GPT LLM + Kokoro TTS", 
                "services": ["stt_whisper", "llm_gpt", "tts_kokoro"],
                "use_case": "Balanced cloud/local with speed priority",
                "total_latency": "~600ms",
                "pros": ["Fast GPT responses", "Fastest TTS", "Low latency"],
                "cons": ["Cloud dependency", "Limited TTS voices"]
            },
            
            "quality_combo": {
                "name": "💎 Quality Combo",
                "description": "Orchestrator + Whisper STT + GPT LLM + Zonos TTS",
                "services": ["stt_whisper", "llm_gpt", "tts_zonos"],
                "use_case": "Quality-focused with balanced performance",
                "total_latency": "~900ms", 
                "pros": ["High-quality responses", "Balanced TTS", "Good voice variety"],
                "cons": ["Cloud dependency", "Medium latency"]
            },
            
            "premium_combo": {
                "name": "� Premium Combo",
                "description": "Orchestrator + Whisper STT + GPT LLM + Tortoise TTS",
                "services": ["stt_whisper", "llm_gpt", "tts_tortoise"],
                "use_case": "Premium quality for professional content",
                "total_latency": "~2-3s",
                "pros": ["Highest audio quality", "29 premium voices", "Professional grade"],
                "cons": ["Slower generation", "High GPU usage", "Cloud LLM dependency"]
            },
            
            # Additional comprehensive combos
            "local_powerhouse": {
                "name": "� Local Powerhouse",
                "description": "All-local processing for maximum privacy",
                "services": ["stt_whisper", "llm_mistral", "tts_tortoise", "tts_kokoro", "tts_zonos"],
                "use_case": "Privacy-focused with multiple TTS options",
                "total_latency": "Variable",
                "pros": ["Complete privacy", "No cloud deps", "44 total voices"],
                "cons": ["High resource usage", "Complex management"]
            },
            
            "full_orchestra": {
                "name": "🎛️ Full Orchestra",
                "description": "Complete deployment of all services",
                "services": list(self.services.keys()),
                "use_case": "Development, testing, maximum capabilities",
                "total_latency": "Variable",
                "pros": ["Everything available", "54+ total voices", "Full redundancy"],
                "cons": ["Highest resource usage", "Management complexity"]
            }
        }
        
        # System configuration
        self.orchestrator_url = "http://localhost:8080"
        self.orchestrator_ws_url = "ws://localhost:9000"
        self.auto_restart_enabled = False
        self.monitor_task = None
        self.monitor_thread = None
        self.orchestrator_process = None  # Track orchestrator process
        
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print("\n🛑 Launcher shutting down gracefully...")
            print("💡 Note: Services will continue running in background")
            print("   Use option 5 to stop all services when needed")
            # Don't terminate child processes - let them run independently
            os._exit(0)  # Exit immediately without cleanup that might kill children
        
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)

    def is_service_actually_running(self, service_id: str) -> tuple[bool, bool]:
        """
        Check if a service is actually running by checking:
        1. Process is running (by checking all python processes)
        2. Service is registered in orchestrator
        Returns (process_running, orchestrator_registered)
        """
        import psutil
        import requests
        
        service = self.services.get(service_id)
        if not service:
            return False, False
        
        # Map launcher service IDs to orchestrator service IDs
        orchestrator_service_id = service_id + "_ws"  # Most services add "_ws" suffix
        
        # Check if registered in orchestrator
        orchestrator_registered = False
        try:
            response = requests.get(f"{self.orchestrator_url}/services", timeout=2)
            if response.status_code == 200:
                services_data = response.json()
                # The response is a direct array, not wrapped in 'value'
                if isinstance(services_data, list):
                    registered_services = [s.get('service_id', '') for s in services_data]
                else:
                    # Fallback for wrapped format
                    registered_services = [s.get('service_id', '') for s in services_data.get('value', [])]
                orchestrator_registered = orchestrator_service_id in registered_services
        except Exception as e:
            # Debug: print the error to understand issues
            print(f"Debug: Orchestrator check failed for {service_id}: {e}")
            pass
        
        # Check if process is running by looking for the service script
        process_running = False
        service_script = service.get('script', '')  # Fixed: use 'script' instead of 'file'
        if service_script:
            try:
                # Get just the filename for matching
                script_name = service_script.split('\\')[-1].split('/')[-1]
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                        cmdline = ' '.join(proc.info['cmdline'])
                        # Check if the script name is in the command line
                        if script_name in cmdline:
                            process_running = True
                            # Update the process object if we found it
                            try:
                                service["process"] = psutil.Process(proc.info['pid'])
                            except:
                                pass
                            break
            except Exception as e:
                # Debug: print the error to understand issues
                print(f"Debug: Process check failed for {service_id}: {e}")
                pass
        
        return process_running, orchestrator_registered

    def start_orchestrator(self) -> bool:
        """Start the WebSocket orchestrator service if not running"""
        if self.orchestrator_process is not None:
            # Check if process is still alive
            if self.orchestrator_process.poll() is None:
                return True  # Already running
            else:
                self.orchestrator_process = None
        
        try:
            orchestrator_script = Path("ws_orchestrator_service.py")
            if not orchestrator_script.exists():
                self.safe_print("❌ Orchestrator script not found")
                return False
                
            self.safe_print("🚀 Starting WebSocket Orchestrator...")
            self.orchestrator_process = subprocess.Popen([
                sys.executable, str(orchestrator_script)
            ], cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Give orchestrator time to start
            time.sleep(3)
            
            # Check if it's running
            if self.orchestrator_process.poll() is None:
                self.safe_print("✅ WebSocket Orchestrator started successfully")
                return True
            else:
                stdout, stderr = self.orchestrator_process.communicate()
                self.safe_print("❌ Orchestrator failed to start")
                if stderr:
                    self.safe_print(f"   💥 Error: {stderr.strip()}")
                self.orchestrator_process = None
                return False
                
        except Exception as e:
            self.safe_print(f"❌ Error starting orchestrator: {e}")
            self.orchestrator_process = None
            return False
    
    def stop_orchestrator(self):
        """Stop the orchestrator service"""
        if self.orchestrator_process is not None:
            self.safe_print("🛑 Stopping WebSocket Orchestrator...")
            self.orchestrator_process.terminate()
            try:
                self.orchestrator_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.orchestrator_process.kill()
            self.orchestrator_process = None
            self.safe_print("✅ Orchestrator stopped")
        
    def safe_print(self, text: str):
        """Safe print function that handles Unicode characters for Windows console."""
        try:
            print(text)
        except UnicodeEncodeError:
            safe_text = text.encode('ascii', 'replace').decode('ascii')
            print(safe_text)
    
    def check_orchestrator_health(self) -> bool:
        """Check if orchestrator is running and healthy"""
        try:
            response = requests.get(f"{self.orchestrator_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
        except Exception as e:
            logging.debug(f"Orchestrator health check failed: {e}")
        return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all registered services from orchestrator"""
        try:
            response = requests.get(f"{self.orchestrator_url}/services", timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Handle both list and dict responses
                if isinstance(data, list):
                    return {"services": data, "count": len(data)}
                elif isinstance(data, dict):
                    return data
                else:
                    return {"services": [], "count": 0}
        except Exception:
            pass
        return {"services": [], "count": 0}
    
    def start_service(self, service_id: str) -> bool:
        """Start a specific service"""
        if service_id not in self.services:
            self.safe_print(f"❌ Unknown service: {service_id}")
            return False
        
        service = self.services[service_id]
        
        if service["process"] is not None:
            self.safe_print(f"⚠️  Service {service['name']} is already running")
            return True
        
        # Auto-start orchestrator if needed (for WebSocket services)
        if not self.check_orchestrator_health():
            self.safe_print("📡 Orchestrator not running, starting it first...")
            if not self.start_orchestrator():
                self.safe_print("❌ Cannot start service without orchestrator")
                return False
            # Wait a bit more for orchestrator to be fully ready
            time.sleep(2)
        
        try:
            script_path = Path(service["file"])
            if not script_path.exists():
                self.safe_print(f"❌ Service script not found: {script_path}")
                return False
            
            self.safe_print(f"🚀 Starting {service['name']}...")
            self.safe_print(f"   📂 Script: {script_path}")
            self.safe_print(f"   🔧 Engine: {service.get('engine', 'unknown')}")
            
            # Start service process as independent background process
            if os.name == 'nt':  # Windows
                # Use CREATE_NEW_PROCESS_GROUP to separate from parent's signal handling
                # Don't use DETACHED_PROCESS as it makes tracking difficult
                service["process"] = subprocess.Popen([
                    sys.executable, str(script_path)
                ], 
                cwd=os.getcwd(), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:  # Unix-like systems
                # Use setsid to create new session
                service["process"] = subprocess.Popen([
                    sys.executable, str(script_path)
                ], 
                cwd=os.getcwd(), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                preexec_fn=os.setsid
                )
            
            # Give service time to initialize
            time.sleep(3)  # Increased wait time
            
            # Check if process is still running
            if service["process"].poll() is None:
                self.safe_print(f"✅ {service['name']} started successfully (PID: {service['process'].pid})")
                return True
            else:
                # Process died, get error output
                stdout, stderr = service["process"].communicate()
                self.safe_print(f"❌ {service['name']} failed to start")
                if stderr:
                    self.safe_print(f"   💥 Error: {stderr.strip()}")
                if stdout:
                    self.safe_print(f"   📋 Output: {stdout.strip()}")
                service["process"] = None
                return False
                
        except Exception as e:
            self.safe_print(f"❌ Error starting {service['name']}: {e}")
            service["process"] = None
            return False
    
    def stop_service(self, service_id: str) -> bool:
        """Stop a specific service"""
        if service_id not in self.services:
            self.safe_print(f"❌ Unknown service: {service_id}")
            return False
        
        service = self.services[service_id]
        
        if service["process"] is None:
            self.safe_print(f"⚠️  Service {service['name']} is not running")
            return True
        
        try:
            self.safe_print(f"🛑 Stopping {service['name']}...")
            
            # Gracefully terminate the process
            service["process"].terminate()
            
            # Wait for graceful shutdown
            try:
                service["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                service["process"].kill()
                service["process"].wait()
            
            service["process"] = None
            self.safe_print(f"✅ {service['name']} stopped")
            return True
            
        except Exception as e:
            self.safe_print(f"❌ Error stopping {service['name']}: {e}")
            return False
    
    def start_combo(self, combo_id: str) -> bool:
        """Start a predefined service combination"""
        if combo_id not in self.combos:
            self.safe_print(f"❌ Unknown combo: {combo_id}")
            return False
        
        combo = self.combos[combo_id]
        self.safe_print(f"\n🎛️ Starting {combo['name']}")
        self.safe_print(f"📝 Description: {combo['description']}")
        self.safe_print(f"🎯 Use Case: {combo['use_case']}")
        self.safe_print(f"⏱️  Expected Latency: {combo.get('total_latency', 'Variable')}")
        self.safe_print(f"🔧 Services: {', '.join([self.services[s]['name'] for s in combo['services']])}")
        self.safe_print("=" * 60)
        
        # Auto-start orchestrator first (required for all combos)
        if not self.check_orchestrator_health():
            self.safe_print("📡 Starting orchestrator for combo services...")
            if not self.start_orchestrator():
                self.safe_print("❌ Cannot start combo without orchestrator")
                return False
            # Wait for orchestrator to be fully ready
            time.sleep(2)
        
        # Start all services in the combo
        all_started = True
        for service_id in combo["services"]:
            if not self.start_service(service_id):
                all_started = False
        
        if all_started:
            self.safe_print(f"✅ {combo['name']} started successfully!")
            self.safe_print(f"💡 Pros: {', '.join(combo['pros'])}")
            if combo.get('cons'):
                self.safe_print(f"⚠️  Considerations: {', '.join(combo['cons'])}")
        else:
            self.safe_print(f"❌ Failed to start {combo['name']} completely")
        
        return all_started
    
    def stop_all_services(self):
        """Stop all running services"""
        self.safe_print("\n🛑 Stopping all services...")
        
        stopped_count = 0
        for service_id in self.services:
            if self.services[service_id]["process"] is not None:
                if self.stop_service(service_id):
                    stopped_count += 1
        
        # Also stop orchestrator
        self.stop_orchestrator()
        
        self.safe_print(f"✅ Stopped {stopped_count} services")
    
    def show_service_status(self):
        """Show detailed status of all services"""
        self.safe_print("\n📊 Service Status Report")
        self.safe_print("=" * 60)
        
        # Check orchestrator
        orchestrator_healthy = self.check_orchestrator_health()
        status_icon = "✅" if orchestrator_healthy else "❌"
        self.safe_print(f"{status_icon} Orchestrator: {'Healthy' if orchestrator_healthy else 'Not Available'}")
        
        # Get registered services from orchestrator
        if orchestrator_healthy:
            try:
                service_status = self.get_service_status()
                services_data = service_status.get('services', [])
                # Handle both dict and list formats for services
                if isinstance(services_data, list):
                    registered_services = []
                    for s in services_data:
                        if isinstance(s, dict):
                            registered_services.append(s.get('service_id', 'unknown'))
                        elif isinstance(s, str):
                            registered_services.append(s)
                else:
                    registered_services = []
                self.safe_print(f"📡 Registered Services: {service_status.get('count', len(registered_services))}")
            except Exception as e:
                self.safe_print(f"❌ Error: {e}")
                registered_services = []
        else:
            registered_services = []
        
        self.safe_print("\n🔧 Local Service Processes:")
        
        # Group services by type
        service_types = {"stt": [], "llm": [], "tts": []}
        for service_id, service in self.services.items():
            service_types[service["type"]].append((service_id, service))
        
        for service_type, type_services in service_types.items():
            type_name = {"stt": "Speech-to-Text", "llm": "Language Models", "tts": "Text-to-Speech"}[service_type]
            self.safe_print(f"\n{type_name.upper()}:")
            
            for service_id, service in type_services:
                # Use the improved service detection method - it returns (process_running, orchestrator_registered)
                process_running, orchestrator_registered = self.is_service_actually_running(service_id)
                
                if process_running:
                    process_status = "✅ Running"
                    if orchestrator_registered:
                        registration_status = "✅ Registered"
                    else:
                        registration_status = "⏳ Connecting"
                else:
                    process_status = "❌ Stopped"
                    registration_status = "❌ Not Registered"
                
                # Additional service info
                info_parts = []
                if service.get("engine"):
                    info_parts.append(f"Engine: {service['engine']}")
                if service.get("voices"):
                    info_parts.append(f"{service['voices']} voices")
                if service.get("performance"):
                    info_parts.append(f"{service['performance']}")
                
                info_str = f" ({', '.join(info_parts)})" if info_parts else ""
                
                self.safe_print(f"  {process_status} | {registration_status} | {service['name']}{info_str}")
    
    def show_combo_menu(self):
        """Show available service combinations"""
        self.safe_print("\n🎛️ Available Service Combinations:")
        self.safe_print("=" * 60)
        
        combo_items = list(self.combos.items())
        for i, (combo_id, combo) in enumerate(combo_items, 1):
            self.safe_print(f"{i}. {combo['name']}")
            self.safe_print(f"   📝 {combo['description']}")
            self.safe_print(f"   🎯 Use Case: {combo['use_case']}")
            self.safe_print(f"   ⏱️  Latency: {combo.get('total_latency', 'Variable')}")
            self.safe_print(f"   🔧 Services: {len(combo['services'])} ({', '.join([self.services[s]['engine'] for s in combo['services']])})")
            if i < len(combo_items):
                self.safe_print("")
        
        return combo_items
    
    def show_individual_services_menu(self):
        """Show menu for individual service selection"""
        self.safe_print("\nAvailable services:")
        
        # Maintain specific order for services by iterating through services dict directly
        service_list = []
        counter = 1
        
        # STT Services
        self.safe_print(f"\n🎤 STT Services:")
        for service_id, service in self.services.items():
            if service["type"] == "stt":
                process_running, orchestrator_registered = self.is_service_actually_running(service_id)
                print(f"Debug STT {service_id}: process={process_running}, orchestrator={orchestrator_registered}")
                if process_running and orchestrator_registered:
                    status = "✅ Running | ✅ Registered"
                elif process_running:
                    status = "✅ Running | ⏳ Connecting"
                elif orchestrator_registered:
                    status = "❌ Stopped | 🔄 Registered"
                else:
                    status = "❌ Stopped | ❌ Not Registered"
                extra_info = f" - {service.get('engine', 'unknown')}"
                self.safe_print(f"     {counter}. {status} {service['name']}{extra_info}")
                service_list.append(service_id)
                counter += 1
        
        # LLM Services
        self.safe_print(f"\n🧠 LLM Services:")
        for service_id, service in self.services.items():
            if service["type"] == "llm":
                process_running, orchestrator_registered = self.is_service_actually_running(service_id)
                print(f"Debug LLM {service_id}: process={process_running}, orchestrator={orchestrator_registered}")
                if process_running and orchestrator_registered:
                    status = "✅ Running | ✅ Registered"
                elif process_running:
                    status = "✅ Running | ⏳ Connecting"
                elif orchestrator_registered:
                    status = "❌ Stopped | 🔄 Registered"
                else:
                    status = "❌ Stopped | ❌ Not Registered"
                extra_info = f" - {service.get('engine', 'unknown')}"
                self.safe_print(f"     {counter}. {status} {service['name']}{extra_info}")
                service_list.append(service_id)
                counter += 1
        
        # TTS Services (will now follow the dictionary order: kokoro, zonos, tortoise, dia)
        self.safe_print(f"\n🗣️  TTS Services:")
        for service_id, service in self.services.items():
            if service["type"] == "tts":
                process_running, orchestrator_registered = self.is_service_actually_running(service_id)
                if process_running and orchestrator_registered:
                    status = "✅ Running | ✅ Registered"
                elif process_running:
                    status = "✅ Running | ⏳ Connecting"
                elif orchestrator_registered:
                    status = "❌ Stopped | 🔄 Registered"
                else:
                    status = "❌ Stopped | ❌ Not Registered"
                extra_info = f" - {service.get('voices', 0)} voices"
                self.safe_print(f"     {counter}. {status} {service['name']}{extra_info}")
                service_list.append(service_id)
                counter += 1
        
        return service_list
    
    async def monitor_services(self):
        """Monitor services and restart if needed"""
        self.safe_print("🔍 Starting service monitoring...")
        
        while self.auto_restart_enabled:
            try:
                for service_id, service in self.services.items():
                    if service["process"] is not None:
                        # Check if process died
                        if service["process"].poll() is not None:
                            self.safe_print(f"⚠️  Service {service['name']} died, restarting...")
                            service["process"] = None
                            self.start_service(service_id)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logging.error(f"Monitor error: {e}")
                await asyncio.sleep(5)
    
    async def test_streaming_pipeline(self):
        """Test the complete streaming pipeline"""
        self.safe_print("\n🧪 Testing Streaming Pipeline...")
        
        if not self.check_orchestrator_health():
            self.safe_print("❌ Orchestrator not available for testing")
            return False
        
        try:
            # Import test client
            from ws_headset_client import HeadsetClient
            
            self.safe_print("🔌 Connecting to streaming pipeline...")
            client = HeadsetClient()
            
            await client.connect()
            self.safe_print("✅ Connected to orchestrator")
            
            # Test message
            test_message = "Hello, this is a test of the complete streaming pipeline."
            self.safe_print(f"📤 Sending test message: '{test_message}'")
            
            await client.send_text_input(test_message)
            
            # Wait for response
            self.safe_print("⏳ Waiting for pipeline response...")
            await asyncio.sleep(3)
            
            await client.disconnect()
            self.safe_print("✅ Pipeline test completed")
            
            return True
            
        except Exception as e:
            self.safe_print(f"❌ Pipeline test failed: {e}")
            return False
    
    async def run_microphone_tests(self):
        """Run the microphone testing suite"""
        self.safe_print("\n🎤 Launching Microphone Testing Suite...")
        self.safe_print("=" * 60)
        self.safe_print("This suite provides comprehensive testing with real microphone input:")
        self.safe_print("  🎤 STT Only - Test speech recognition")
        self.safe_print("  🎤➡️🧠 STT → LLM - Speech to AI response") 
        self.safe_print("  🧠➡️🗣️ LLM → TTS - Text to speech synthesis")
        self.safe_print("  🗣️ TTS Only - Direct text to speech")
        self.safe_print("  🎤➡️🧠➡️🗣️ Full Pipeline - Complete voice conversation")
        self.safe_print("  🔄 Conversation Loop - Continuous voice chat")
        self.safe_print("")
        
        try:
            from microphone_test_suite import run_microphone_tests
            await run_microphone_tests()
        except ImportError as e:
            self.safe_print(f"❌ Microphone test suite not available: {e}")
            self.safe_print("   Make sure microphone_test_suite.py is in the project directory")
        except Exception as e:
            self.safe_print(f"❌ Microphone testing failed: {e}")
        
        self.safe_print("\n🎤 Microphone testing suite completed")
    
    def main_menu(self):
        """Main interactive menu"""
        while True:
            self.safe_print("\n" + "=" * 60)
            self.safe_print("🎛️  WebSocket Services Launcher - All Services")
            self.safe_print("=" * 60)
            self.safe_print("     1. Show detailed service status")
            self.safe_print("     2. Start Service Combos")
            self.safe_print("     3. Start Individual Service")
            self.safe_print("     4. Stop Individual Service")
            self.safe_print("     5. Stop All Services")
            self.safe_print("     6. Monitor Services (with auto-restart)")
            self.safe_print("     7. Test Streaming Pipeline")
            self.safe_print("     8. 🎤 Microphone Testing Suite")
            self.safe_print("     0. Exit")
            self.safe_print("-" * 60)
            
            try:
                choice = input("Enter your choice (0-8): ").strip().lower()
                
                if choice == "1":
                    self.show_service_status()
                elif choice == "2":
                    self.combos_menu()
                elif choice == "3":
                    self.handle_individual_start()
                elif choice == "4":
                    self.handle_individual_stop()
                elif choice == "5":
                    self.stop_all_services()
                elif choice == "6":
                    self.handle_monitoring()
                elif choice == "7":
                    asyncio.run(self.test_streaming_pipeline())
                elif choice == "8":
                    asyncio.run(self.run_microphone_tests())
                elif choice in ["0", "q", "quit", "exit"]:
                    self.safe_print("\n👋 Shutting down launcher...")
                    # Stop monitoring if active
                    if self.auto_restart_enabled:
                        self.auto_restart_enabled = False
                        if self.monitor_thread and self.monitor_thread.is_alive():
                            self.safe_print("🔍 Stopping background monitoring...")
                    # DO NOT stop services - leave them running
                    self.safe_print("💡 Note: Services are still running in the background")
                    self.safe_print("   Use option 5 to manually stop services if needed")
                    self.safe_print("✅ Launcher shutdown complete")
                    break
                else:
                    self.safe_print("❌ Invalid choice. Please enter 0-8.")
                    
            except KeyboardInterrupt:
                self.safe_print("\n\n👋 Interrupted by user. Shutting down launcher...")
                # Stop monitoring if active
                if self.auto_restart_enabled:
                    self.auto_restart_enabled = False
                    if self.monitor_thread and self.monitor_thread.is_alive():
                        self.safe_print("🔍 Stopping background monitoring...")
                # DO NOT stop services - leave them running
                self.safe_print("💡 Note: Services are still running in the background")
                self.safe_print("   Use 'python comprehensive_ws_launcher.py' and option 5 to stop services")
                self.safe_print("✅ Launcher shutdown complete")
                break
            except Exception as e:
                self.safe_print(f"❌ Error: {e}")
    
    def combos_menu(self):
        """Service combinations menu"""
        while True:
            self.safe_print("\n" + "=" * 60)
            self.safe_print("🎛️  Service Combos - Predefined Combinations")
            self.safe_print("=" * 60)
            self.safe_print("     1. Fast Combo (Whisper STT + Mistral LLM + Kokoro TTS)")
            self.safe_print("     2. Efficient Combo (Whisper STT + Mistral LLM + Zonos TTS)")
            self.safe_print("     3. Balanced Combo (Whisper STT + GPT LLM + Kokoro TTS)")
            self.safe_print("     4. Quality Combo (Whisper STT + GPT LLM + Zonos TTS)")
            self.safe_print("     5. Premium Combo (Whisper STT + GPT LLM + Tortoise TTS)")
            self.safe_print("     6. Advanced Combos")
            self.safe_print("     7. Back to Main Menu")
            self.safe_print("-" * 60)
            
            try:
                choice = input("Enter your choice (1-7): ").strip()
                
                if choice == "1":
                    self.start_combo("fast_combo")
                    break
                elif choice == "2":
                    self.start_combo("efficient_combo")
                    break
                elif choice == "3":
                    self.start_combo("balanced_combo")
                    break
                elif choice == "4":
                    self.start_combo("quality_combo")
                    break
                elif choice == "5":
                    self.start_combo("premium_combo")
                    break
                elif choice == "6":
                    self.advanced_combos_menu()
                    break
                elif choice == "7":
                    break
                else:
                    self.safe_print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.safe_print(f"❌ Error: {e}")
    
    def advanced_combos_menu(self):
        """Advanced combinations menu"""
        while True:
            self.safe_print("\n" + "=" * 60)
            self.safe_print("🔬 Advanced Service Combinations")
            self.safe_print("=" * 60)
            self.safe_print("     1. Local Powerhouse (All Local + Multiple TTS)")
            self.safe_print("     2. Full Orchestra (All Services)")
            self.safe_print("     3. Custom Combo Builder")
            self.safe_print("     4. Back to Combos Menu")
            self.safe_print("-" * 60)
            
            try:
                choice = input("Enter your choice (1-4): ").strip()
                
                if choice == "1":
                    self.start_combo("local_powerhouse")
                    break
                elif choice == "2":
                    self.start_combo("full_orchestra")
                    break
                elif choice == "3":
                    self.custom_combo_builder()
                    break
                elif choice == "4":
                    break
                else:
                    self.safe_print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.safe_print(f"❌ Error: {e}")
    
    def custom_combo_builder(self):
        """Build custom service combinations"""
        self.safe_print("\n🛠️  Custom Combo Builder")
        self.safe_print("Select services to include in your custom combo:")
        self.safe_print("-" * 50)
        
        service_list = self.show_individual_services_menu()
        selected_services = []
        
        while True:
            try:
                self.safe_print(f"\nCurrently selected: {len(selected_services)} services")
                if selected_services:
                    selected_names = [self.services[s]["name"] for s in selected_services]
                    self.safe_print(f"Selected: {', '.join(selected_names)}")
                
                choice = input(f"\nAdd service (1-{len(service_list)}), 'start' to launch, 'clear' to reset, or 'cancel': ").strip().lower()
                
                if choice == 'start':
                    if selected_services:
                        self.safe_print(f"\n🚀 Starting custom combo with {len(selected_services)} services...")
                        
                        # Auto-start orchestrator first (required for all services)
                        if not self.check_orchestrator_health():
                            self.safe_print("📡 Starting orchestrator for custom combo...")
                            if not self.start_orchestrator():
                                self.safe_print("❌ Cannot start custom combo without orchestrator")
                                break
                            # Wait for orchestrator to be fully ready
                            time.sleep(2)
                        
                        all_started = True
                        for service_id in selected_services:
                            if not self.start_service(service_id):
                                all_started = False
                        
                        if all_started:
                            self.safe_print("✅ Custom combo started successfully!")
                        else:
                            self.safe_print("⚠️  Some services failed to start")
                    else:
                        self.safe_print("❌ No services selected")
                    break
                    
                elif choice == 'clear':
                    selected_services.clear()
                    self.safe_print("🗑️  Selection cleared")
                    
                elif choice == 'cancel':
                    break
                    
                elif choice.isdigit():
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(service_list):
                        service_id = service_list[choice_idx]
                        if service_id not in selected_services:
                            selected_services.append(service_id)
                            self.safe_print(f"✅ Added {self.services[service_id]['name']}")
                        else:
                            self.safe_print(f"⚠️  {self.services[service_id]['name']} already selected")
                    else:
                        self.safe_print("❌ Invalid selection")
                else:
                    self.safe_print("❌ Invalid input")
                    
            except ValueError:
                self.safe_print("❌ Invalid input")
            except KeyboardInterrupt:
                break
    
    def handle_combo_selection(self):
        """Handle combo selection menu"""
        combo_items = self.show_combo_menu()
        
        try:
            choice = input(f"\nSelect combo (1-{len(combo_items)}) or press Enter to cancel: ").strip()
            if not choice:
                return
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(combo_items):
                combo_id, _ = combo_items[choice_idx]
                self.start_combo(combo_id)
            else:
                self.safe_print("❌ Invalid selection")
                
        except ValueError:
            self.safe_print("❌ Invalid input")
        except KeyboardInterrupt:
            self.safe_print("\n⏸️  Cancelled")
    
    def handle_individual_start(self):
        """Handle individual service start"""
        service_list = self.show_individual_services_menu()
        
        try:
            choice = input(f"\nSelect service (1-{len(service_list)}) or press Enter to cancel: ").strip()
            if not choice:
                return
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(service_list):
                service_id = service_list[choice_idx]
                self.start_service(service_id)
            else:
                self.safe_print("❌ Invalid selection")
                
        except ValueError:
            self.safe_print("❌ Invalid input")
        except KeyboardInterrupt:
            self.safe_print("\n⏸️  Cancelled")
    
    def handle_individual_stop(self):
        """Handle individual service stop"""
        service_list = self.show_individual_services_menu()
        
        try:
            choice = input(f"\nSelect service to stop (1-{len(service_list)}) or press Enter to cancel: ").strip()
            if not choice:
                return
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(service_list):
                service_id = service_list[choice_idx]
                self.stop_service(service_id)
            else:
                self.safe_print("❌ Invalid selection")
                
        except ValueError:
            self.safe_print("❌ Invalid input")
        except KeyboardInterrupt:
            self.safe_print("\n⏸️  Cancelled")
    
    def handle_monitoring(self):
        """Handle service monitoring"""
        if self.auto_restart_enabled:
            self.safe_print("⏹️  Stopping service monitoring...")
            self.auto_restart_enabled = False
            if self.monitor_task:
                self.monitor_task.cancel()
            return
        
        self.safe_print("🔍 Enabling auto-restart monitoring...")
        self.safe_print("   (Monitoring will run in background - you can use other menu options)")
        self.safe_print("   (Select option 6 again to stop monitoring)")
        self.auto_restart_enabled = True
        
        # Start monitoring in background thread to avoid blocking menu
        import threading
        
        def run_monitor_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.monitor_services())
            except Exception as e:
                self.safe_print(f"🔍 Monitoring stopped: {e}")
            finally:
                loop.close()
        
        self.monitor_thread = threading.Thread(target=run_monitor_thread, daemon=True)
        self.monitor_thread.start()

def main():
    """Main entry point"""
    print("🎛️  Starting Comprehensive WebSocket Services Launcher...")
    print()
    print("🚀 Supporting all TTS engines: Tortoise (29 voices), Kokoro (9 voices), Zonos (6 voices), Dia (10 voices)")
    print("🧠 Supporting all LLM engines: GPT, Mistral") 
    print("🎤 Supporting STT: Whisper")
    print("⚡ Total: 54+ voices, Professional-grade service combinations")
    
    launcher = ComprehensiveWebSocketLauncher()
    
    try:
        launcher.main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Forced exit detected. Cleaning up launcher...")
        # Final cleanup in case main menu didn't handle it
        launcher.auto_restart_enabled = False
        # DO NOT stop services - leave them running
        print("💡 Note: Services are still running in the background")
        print("✅ Launcher cleanup complete")
    finally:
        # Ensure we exit cleanly
        import sys
        sys.exit(0)

if __name__ == "__main__":
    main()
