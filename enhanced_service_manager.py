#!/usr/bin/env python3
"""
Enhanced Service Manager - Addresses service persistence and status display issues
"""
import asyncio
import signal
import psutil
import requests
import subprocess
import time
from pathlib import Path

class EnhancedServiceManager:
    def __init__(self):
        self.orchestrator_url = "http://localhost:8080"
        self.orchestrator_ws_url = "ws://localhost:9001"
        self.services = {
            "stt_whisper": {
                "name": "Whisper STT Service",
                "type": "stt", 
                "script": "aws_microservices/ws_stt_whisper_service.py",
                "process": None,
                "engine": "OpenAI Whisper",
                "restart_count": 0,
                "max_restarts": 3
            },
            "llm_gpt": {
                "name": "GPT LLM Service",
                "type": "llm",
                "script": "aws_microservices/ws_llm_gpt_service.py", 
                "process": None,
                "engine": "OpenAI GPT",
                "restart_count": 0,
                "max_restarts": 3
            },
            "tts_kokoro": {
                "name": "Kokoro TTS Service",
                "type": "tts",
                "script": "aws_microservices/ws_tts_kokoro_service.py",
                "process": None,
                "engine": "Kokoro",
                "voices": "9",
                "restart_count": 0,
                "max_restarts": 3
            }
        }
        
    def start_service(self, service_id: str) -> bool:
        """Start a service with proper process group isolation"""
        service = self.services.get(service_id)
        if not service:
            print(f"❌ Service {service_id} not found")
            return False
            
        # Check if already running
        if self.is_service_running(service_id):
            print(f"✅ Service {service_id} already running")
            return True
            
        script_path = service["script"]
        if not Path(script_path).exists():
            print(f"❌ Script not found: {script_path}")
            return False
            
        try:
            print(f"🚀 Starting {service['name']}...")
            
            # Start with process group isolation for Windows
            if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP'):
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                creation_flags = 0
                
            process = subprocess.Popen(
                ["python", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags,
                cwd="."
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            # Check if it's still running
            if process.poll() is None:
                service["process"] = process
                service["restart_count"] = 0
                print(f"✅ {service['name']} started successfully (PID: {process.pid})")
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ {service['name']} failed to start")
                print(f"   stdout: {stdout.decode()[:200]}...")
                print(f"   stderr: {stderr.decode()[:200]}...")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start {service['name']}: {e}")
            return False
    
    def is_service_running(self, service_id: str) -> tuple:
        """Check if service is running and registered with orchestrator"""
        service = self.services.get(service_id)
        if not service:
            return False, False
        
        # Check process
        process_running = False
        script_path = service["script"]
        script_name = script_path.split("/")[-1]
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if script_name in cmdline:
                        process_running = True
                        # Update process reference
                        try:
                            service["process"] = psutil.Process(proc.info['pid'])
                        except:
                            pass
                        break
        except Exception as e:
            print(f"Debug: Process check error for {service_id}: {e}")
        
        # Check orchestrator registration
        orchestrator_registered = False
        orchestrator_id = service_id + "_ws"
        
        try:
            response = requests.get(f"{self.orchestrator_url}/services", timeout=3)
            if response.status_code == 200:
                services_data = response.json()
                if isinstance(services_data, list):
                    registered_services = []
                    for s in services_data:
                        if isinstance(s, dict):
                            registered_services.append(s.get('service_id', ''))
                        elif isinstance(s, str):
                            registered_services.append(s)
                    orchestrator_registered = orchestrator_id in registered_services
        except Exception as e:
            print(f"Debug: Orchestrator check error for {service_id}: {e}")
        
        return process_running, orchestrator_registered
    
    def restart_service(self, service_id: str) -> bool:
        """Restart a service with exponential backoff"""
        service = self.services.get(service_id)
        if not service:
            return False
        
        # Check restart limit
        if service["restart_count"] >= service["max_restarts"]:
            print(f"❌ {service['name']} has reached maximum restart attempts ({service['max_restarts']})")
            return False
        
        print(f"🔄 Restarting {service['name']} (attempt {service['restart_count'] + 1}/{service['max_restarts']})")
        
        # Stop existing process
        if service.get("process"):
            try:
                service["process"].terminate()
                time.sleep(2)
                if service["process"].poll() is None:
                    service["process"].kill()
            except:
                pass
        
        service["restart_count"] += 1
        
        # Wait before restart (exponential backoff)
        wait_time = min(2 ** service["restart_count"], 10)
        print(f"⏱️ Waiting {wait_time} seconds before restart...")
        time.sleep(wait_time)
        
        return self.start_service(service_id)
    
    def show_status(self):
        """Show comprehensive service status"""
        print("\n" + "="*70)
        print("📊 Enhanced Service Status Report")
        print("="*70)
        
        # Check orchestrator
        orchestrator_healthy = False
        try:
            response = requests.get(f"{self.orchestrator_url}/health", timeout=3)
            orchestrator_healthy = response.status_code in [200, 426]  # 426 is websocket upgrade required
        except:
            pass
        
        status_icon = "✅" if orchestrator_healthy else "❌"
        print(f"{status_icon} Orchestrator: {'Healthy' if orchestrator_healthy else 'Not Available'}")
        
        # Service status
        print(f"\n🔧 Service Status:")
        for service_id, service in self.services.items():
            process_running, orchestrator_registered = self.is_service_running(service_id)
            
            if process_running:
                process_status = "✅ Running"
                if orchestrator_registered:
                    registration_status = "✅ Registered"
                else:
                    registration_status = "⏳ Connecting"
            else:
                process_status = "❌ Stopped"
                registration_status = "❌ Not Registered"
            
            restart_info = f" (restarts: {service['restart_count']}/{service['max_restarts']})" if service['restart_count'] > 0 else ""
            print(f"  {process_status} | {registration_status} | {service['name']}{restart_info}")
    
    def monitor_services(self, duration_minutes: int = 60):
        """Monitor services and restart if needed"""
        print(f"\n🔍 Starting service monitoring for {duration_minutes} minutes...")
        start_time = time.time()
        check_interval = 30  # Check every 30 seconds
        
        while time.time() - start_time < duration_minutes * 60:
            print(f"\n--- Service Health Check [{time.strftime('%H:%M:%S')}] ---")
            
            for service_id in self.services:
                process_running, orchestrator_registered = self.is_service_running(service_id)
                service_name = self.services[service_id]['name']
                
                if not process_running:
                    print(f"⚠️ {service_name} is not running - attempting restart...")
                    self.restart_service(service_id)
                elif process_running and not orchestrator_registered:
                    print(f"⚠️ {service_name} is running but not registered - may be reconnecting...")
                else:
                    print(f"✅ {service_name} is healthy")
            
            print(f"💤 Sleeping for {check_interval} seconds...")
            time.sleep(check_interval)
        
        print("🛑 Monitoring period complete")

def main():
    manager = EnhancedServiceManager()
    
    print("🎛️ Enhanced Service Manager")
    print("🔧 Features: Auto-restart, Status monitoring, Process isolation")
    
    while True:
        print("\n" + "="*50)
        print("1. Show service status")
        print("2. Start individual service")  
        print("3. Start all services")
        print("4. Restart service")
        print("5. Monitor services")
        print("0. Exit")
        print("-"*50)
        
        choice = input("Enter choice (0-5): ").strip()
        
        if choice == "0":
            print("👋 Exiting Enhanced Service Manager...")
            break
        elif choice == "1":
            manager.show_status()
        elif choice == "2":
            print("\nAvailable services:")
            for i, (service_id, service) in enumerate(manager.services.items(), 1):
                print(f"  {i}. {service['name']}")
            
            try:
                service_num = int(input("Select service (number): ")) - 1
                service_id = list(manager.services.keys())[service_num]
                manager.start_service(service_id)
            except (ValueError, IndexError):
                print("❌ Invalid selection")
        elif choice == "3":
            print("🚀 Starting all services...")
            for service_id in manager.services:
                manager.start_service(service_id)
        elif choice == "4":
            print("\nAvailable services:")
            for i, (service_id, service) in enumerate(manager.services.items(), 1):
                print(f"  {i}. {service['name']}")
            
            try:
                service_num = int(input("Select service to restart (number): ")) - 1
                service_id = list(manager.services.keys())[service_num]
                manager.restart_service(service_id)
            except (ValueError, IndexError):
                print("❌ Invalid selection")
        elif choice == "5":
            try:
                duration = int(input("Monitor duration in minutes (default 60): ") or "60")
                manager.monitor_services(duration)
            except ValueError:
                print("❌ Invalid duration")
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
