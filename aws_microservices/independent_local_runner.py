#!/usr/bin/env python3
"""
Independent Services Local Runner
Manages separate TTS and LLM services independently
"""
import os
import sys
import subprocess
import threading
import time
import signal
import requests
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class IndependentServiceManager:
    """Manager for independent microservices"""
    
    def __init__(self):
        self.services = {}
        self.running = False
        
        # Service configurations
        self.service_configs = {
            "stt": {
                "script": "stt_service.py",
                "port": 8001,
                "description": "Speech-to-Text Service",
                "required": True
            },
            "kokoro_tts": {
                "script": "tts_kokoro_service.py", 
                "port": 8011,
                "description": "Kokoro TTS Service (Fast)",
                "required": False
            },
            "hira_dia_tts": {
                "script": "tts_hira_dia_service.py",
                "port": 8012, 
                "description": "Hira Dia TTS Service (High Quality)",
                "required": False
            },
            "mistral_llm": {
                "script": "llm_mistral_service.py",
                "port": 8021,
                "description": "Mistral LLM Service",
                "required": False
            },
            "gpt_llm": {
                "script": "llm_gpt_service.py", 
                "port": 8022,
                "description": "GPT LLM Service",
                "required": False
            }
        }
    
    def start_service(self, service_name: str) -> bool:
        """Start a specific service"""
        if service_name not in self.service_configs:
            print(f"❌ Unknown service: {service_name}")
            return False
        
        if service_name in self.services:
            print(f"⚠️ Service {service_name} already running")
            return True
        
        config = self.service_configs[service_name]
        script_path = Path(__file__).parent / config["script"]
        
        if not script_path.exists():
            print(f"❌ Service script not found: {script_path}")
            return False
        
        print(f"🚀 Starting {config['description']} on port {config['port']}...")
        
        try:
            # Start the service process
            process = subprocess.Popen([
                sys.executable, str(script_path)
            ], cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.services[service_name] = {
                "process": process,
                "config": config,
                "start_time": time.time()
            }
            
            # Wait a moment for startup
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                print(f"✅ {config['description']} started successfully")
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ {config['description']} failed to start")
                print(f"STDOUT: {stdout.decode()}")
                print(f"STDERR: {stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start {service_name}: {e}")
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a specific service"""
        if service_name not in self.services:
            print(f"⚠️ Service {service_name} not running")
            return True
        
        service_info = self.services[service_name]
        process = service_info["process"]
        
        print(f"🛑 Stopping {service_info['config']['description']}...")
        
        try:
            # Try graceful shutdown first
            process.terminate()
            
            # Wait up to 10 seconds for graceful shutdown
            try:
                process.wait(timeout=10)
                print(f"✅ {service_name} stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                process.kill()
                process.wait()
                print(f"⚡ {service_name} force killed")
            
            del self.services[service_name]
            return True
            
        except Exception as e:
            print(f"❌ Failed to stop {service_name}: {e}")
            return False
    
    def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy"""
        if service_name not in self.services:
            return False
        
        config = self.service_configs[service_name]
        
        try:
            response = requests.get(f"http://localhost:{config['port']}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_service_status(self) -> dict:
        """Get status of all services"""
        status = {}
        
        for service_name, config in self.service_configs.items():
            if service_name in self.services:
                service_info = self.services[service_name]
                process = service_info["process"]
                
                # Check if process is running
                if process.poll() is None:
                    # Check health endpoint
                    healthy = self.check_service_health(service_name)
                    status[service_name] = {
                        "status": "healthy" if healthy else "unhealthy",
                        "port": config["port"],
                        "uptime": time.time() - service_info["start_time"],
                        "description": config["description"]
                    }
                else:
                    status[service_name] = {
                        "status": "crashed",
                        "port": config["port"], 
                        "description": config["description"]
                    }
            else:
                status[service_name] = {
                    "status": "stopped",
                    "port": config["port"],
                    "description": config["description"]
                }
        
        return status
    
    def start_selected_services(self, services_to_start: list):
        """Start only selected services"""
        print("🎭 Independent Microservices Manager")
        print("=" * 50)
        
        success_count = 0
        
        for service_name in services_to_start:
            if self.start_service(service_name):
                success_count += 1
            time.sleep(1)  # Brief pause between starts
        
        print(f"\n✅ Started {success_count}/{len(services_to_start)} services")
        
        # Show status
        self.show_status()
        
        if success_count > 0:
            print(f"\n🔗 Service URLs:")
            for service_name in services_to_start:
                if service_name in self.services:
                    config = self.service_configs[service_name]
                    print(f"  {config['description']}: http://localhost:{config['port']}")
    
    def start_all_services(self):
        """Start all available services"""
        services_to_start = list(self.service_configs.keys())
        self.start_selected_services(services_to_start)
    
    def stop_all_services(self):
        """Stop all running services"""
        print("\n🛑 Stopping all services...")
        
        for service_name in list(self.services.keys()):
            self.stop_service(service_name)
        
        print("✅ All services stopped")
    
    def show_status(self):
        """Show current status of all services"""
        print("\n📊 Service Status:")
        status = self.get_service_status()
        
        for service_name, info in status.items():
            status_icon = {
                "healthy": "✅",
                "unhealthy": "⚠️",
                "crashed": "❌", 
                "stopped": "⏹️"
            }.get(info["status"], "❓")
            
            print(f"  {status_icon} {info['description']} (Port {info['port']}): {info['status']}")
            
            if info["status"] == "healthy" and "uptime" in info:
                uptime_min = info["uptime"] / 60
                print(f"      Uptime: {uptime_min:.1f} minutes")
    
    def run_interactive(self):
        """Run interactive service management"""
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        while self.running:
            try:
                print("\n🎭 Independent Microservices Manager")
                print("=" * 50)
                
                # Show current status
                self.show_status()
                
                print("\n📋 Service Combinations:")
                print("1. Fast Setup (Kokoro TTS + Mistral LLM)")
                print("2. Quality Setup (Hira Dia TTS + Mistral LLM)")
                print("3. Premium Setup (Hira Dia TTS + GPT LLM)")
                print("4. Testing Setup (Kokoro TTS + GPT LLM)")
                print("5. STT Only (Speech-to-Text)")
                
                print("\n🔧 Individual Services:")
                print("6. Start Kokoro TTS Only")
                print("7. Start Hira Dia TTS Only")
                print("8. Start Mistral LLM Only")
                print("9. Start GPT LLM Only")
                print("10. Start STT Service Only")
                
                print("\n⚙️ Management:")
                print("11. Stop All Services")
                print("12. Test Running Services")
                print("13. Service Health Check")
                print("14. Exit")
                
                choice = input("\nEnter your choice (1-14): ").strip()
                
                if choice == "1":
                    self.start_combination("fast")
                elif choice == "2":
                    self.start_combination("quality")
                elif choice == "3":
                    self.start_combination("premium")
                elif choice == "4":
                    self.start_combination("testing")
                elif choice == "5":
                    self.start_selected_services(["stt"])
                elif choice == "6":
                    self.start_selected_services(["kokoro_tts"])
                elif choice == "7":
                    self.start_selected_services(["hira_dia_tts"])
                elif choice == "8":
                    self.start_selected_services(["mistral_llm"])
                elif choice == "9":
                    self.start_selected_services(["gpt_llm"])
                elif choice == "10":
                    self.start_selected_services(["stt"])
                elif choice == "11":
                    self.stop_all_services()
                elif choice == "12":
                    self.test_running_services()
                elif choice == "13":
                    self.health_check_all()
                elif choice == "14":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-14.")
                
                input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        self.stop_all_services()
    
    def start_combination(self, combo_type: str):
        """Start predefined service combinations"""
        combinations = {
            "fast": {
                "services": ["stt", "kokoro_tts", "mistral_llm"],
                "description": "Fast Setup (Kokoro TTS + Mistral LLM)"
            },
            "quality": {
                "services": ["stt", "hira_dia_tts", "mistral_llm"], 
                "description": "Quality Setup (Hira Dia TTS + Mistral LLM)"
            },
            "premium": {
                "services": ["stt", "hira_dia_tts", "gpt_llm"],
                "description": "Premium Setup (Hira Dia TTS + GPT LLM)"
            },
            "testing": {
                "services": ["stt", "kokoro_tts", "gpt_llm"],
                "description": "Testing Setup (Kokoro TTS + GPT LLM)"
            }
        }
        
        if combo_type not in combinations:
            print(f"❌ Unknown combination: {combo_type}")
            return
        
        combo = combinations[combo_type]
        print(f"\n🚀 Starting {combo['description']}...")
        self.start_selected_services(combo["services"])
    
    def test_running_services(self):
        """Test all currently running services"""
        print("\n🧪 Testing Running Services...")
        running_services = [name for name, info in self.get_service_status().items() 
                          if info["status"] == "healthy"]
        
        if not running_services:
            print("❌ No healthy services to test")
            return
        
        print(f"Found {len(running_services)} healthy services")
        
        # Import and run test functions
        try:
            from test_independent_services import test_stt_service, test_tts_service, test_llm_service
            
            for service_name in running_services:
                config = self.service_configs[service_name]
                base_url = f"http://localhost:{config['port']}"
                
                print(f"\n🔍 Testing {config['description']}...")
                
                if service_name == "stt":
                    test_stt_service(base_url)
                elif service_name.endswith("_tts"):
                    engine = service_name.replace("_tts", "")
                    test_tts_service(config["description"], base_url, engine)
                elif service_name.endswith("_llm"):
                    model = service_name.replace("_llm", "")
                    test_llm_service(config["description"], base_url, model)
                    
        except ImportError:
            print("❌ Test functions not available")
        except Exception as e:
            print(f"❌ Testing failed: {e}")
    
    def health_check_all(self):
        """Perform health check on all configured services"""
        print("\n🏥 Health Check Results:")
        status = self.get_service_status()
        
        healthy_count = 0
        for service_name, info in status.items():
            status_icon = {
                "healthy": "✅",
                "unhealthy": "⚠️", 
                "crashed": "❌",
                "stopped": "⏹️"
            }.get(info["status"], "❓")
            
            print(f"  {status_icon} {info['description']} (Port {info['port']}): {info['status']}")
            
            if info["status"] == "healthy":
                healthy_count += 1
                if "uptime" in info:
                    uptime_min = info["uptime"] / 60
                    print(f"      Uptime: {uptime_min:.1f} minutes")
        
        print(f"\n📊 Summary: {healthy_count}/{len(status)} services healthy")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.running = False
        self.stop_all_services()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Independent Microservices Manager")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--start", nargs="+", help="Start specific services")
    parser.add_argument("--all", action="store_true", help="Start all services")
    parser.add_argument("--status", action="store_true", help="Show service status")
    
    args = parser.parse_args()
    
    manager = IndependentServiceManager()
    
    try:
        if args.interactive:
            manager.run_interactive()
        elif args.status:
            manager.show_status()
        elif args.all:
            manager.start_all_services()
            print("\n🔄 Services running. Press Ctrl+C to stop all.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                manager.stop_all_services()
        elif args.start:
            manager.start_selected_services(args.start)
            print("\n🔄 Services running. Press Ctrl+C to stop all.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                manager.stop_all_services()
        else:
            # Default: show help and start interactive mode
            parser.print_help()
            print("\nStarting interactive mode...\n")
            manager.run_interactive()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
