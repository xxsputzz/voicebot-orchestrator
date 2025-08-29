#!/usr/bin/env python3
"""
Enhanced Independent Microservices Manager
Matches existing test patterns and provides numbered menu interface
"""
import os
import sys
import subprocess
import time
import signal
import requests
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class EnhancedServiceManager:
    """Enhanced manager for independent microservices following existing test patterns"""
    
    def __init__(self):
        self.services = {}
        self.running = False
        self._status_cache = {}
        self._status_cache_time = {}
        self._cache_duration = 30  # Cache for 30 seconds (increased from 10)
        
        # Service configurations matching existing patterns
        self.service_configs = {
            "orchestrator": {
                "script": "start_server.py",
                "port": 8000,
                "description": "Orchestrator (FastAPI)",
                "required": False,
                "type": "orchestrator"
            },
            "whisper_stt": {
                "script": "stt_whisper_service.py",
                "port": 8002,
                "description": "Whisper STT (OpenAI)",
                "required": False,
                "type": "stt"
            },
            "kokoro_tts": {
                "script": "tts_kokoro_service.py", 
                "port": 8011,
                "description": "Kokoro TTS (Fast)",
                "required": False,
                "type": "tts"
            },
            "hira_dia_tts": {
                "script": "tts_hira_dia_service.py",
                "port": 8012, 
                "description": "Hira Dia TTS (High Quality)",
                "required": False,
                "type": "tts"
            },
            "mistral_llm": {
                "script": "llm_mistral_service.py",
                "port": 8021,
                "description": "Mistral LLM",
                "required": False,
                "type": "llm"
            },
            "gpt_llm": {
                "script": "llm_gpt_service.py", 
                "port": 8022,
                "description": "GPT LLM",
                "required": False,
                "type": "llm"
            }
        }
        
        # Predefined combinations following existing patterns - Ordered by preference
        self.combinations = {
            "fast": {
                "name": "Fast Combo",
                "description": "Orchestrator + Whisper STT + Kokoro TTS + Mistral LLM (Real-time)",
                "services": ["orchestrator", "whisper_stt", "kokoro_tts", "mistral_llm"],
                "use_case": "Real-time conversation, quick responses with Whisper accuracy"
            },
            "balanced": {
                "name": "Balanced Combo",
                "description": "Orchestrator + Whisper STT + Kokoro TTS + GPT LLM (Fast TTS, advanced reasoning)",
                "services": ["orchestrator", "whisper_stt", "kokoro_tts", "gpt_llm"],
                "use_case": "Quick TTS with advanced language processing and Whisper accuracy"
            },
            "efficient": {
                "name": "Efficient Combo",
                "description": "Orchestrator + Whisper STT + Hira Dia TTS + Mistral LLM (Quality TTS, efficient LLM)",
                "services": ["orchestrator", "whisper_stt", "hira_dia_tts", "mistral_llm"],
                "use_case": "Quality output with reasonable processing time and Whisper accuracy"
            },
            "quality": {
                "name": "Quality Combo",
                "description": "Orchestrator + Whisper STT + Hira Dia TTS + GPT LLM (Maximum quality)",
                "services": ["orchestrator", "whisper_stt", "hira_dia_tts", "gpt_llm"],
                "use_case": "High-quality content, professional presentations with Whisper accuracy"
            }
        }
    
    def start_service(self, service_name: str) -> bool:
        """Start a specific service (optimized for speed)"""
        if service_name not in self.service_configs:
            print(f"❌ Unknown service: {service_name}")
            return False
        
        # First check if service is already running independently
        config = self.service_configs[service_name]
        if self.check_service_health(service_name, timeout=1, use_cache=False):
            print(f"✅ {config['description']} is already running independently")
            # Invalidate cache for this service since we know it's running
            self.invalidate_cache(service_name)
            return True
        
        if service_name in self.services:
            print(f"✅ Service {service_name} already managed by this process")
            return True
        
        # Handle different script paths
        if service_name == "orchestrator":
            script_path = project_root / config["script"]
        else:
            script_path = Path(__file__).parent / config["script"]
        
        if not script_path.exists():
            print(f"❌ Service script not found: {script_path}")
            print(f"   Expected: {script_path}")
            return False
        
        print(f"🚀 Starting {config['description']} on port {config['port']}...")
        
        try:
            # Start the service process
            if service_name in ["orchestrator", "whisper_stt"]:
                # For orchestrator and whisper_stt, use --direct flag to run server directly
                process = subprocess.Popen([
                    sys.executable, str(script_path), "--direct"
                ], cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                # For other services, run normally
                process = subprocess.Popen([
                    sys.executable, str(script_path)
                ], cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.services[service_name] = {
                "process": process,
                "config": config,
                "start_time": time.time()
            }
            
            # Shorter wait for faster UX (reduced from 3 seconds)
            print("⏳ Waiting for service to start...")
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                # Quick health check with short timeout and no cache
                healthy = self.check_service_health(service_name, timeout=1, use_cache=False)
                # Invalidate cache for this service since status changed
                self.invalidate_cache(service_name)
                
                if healthy:
                    print(f"✅ {config['description']} started and healthy")
                    return True
                else:
                    print(f"⏳ {config['description']} starting... (may need a moment to initialize)")
                    return True  # Still consider it started
            else:
                stdout, stderr = process.communicate()
                print(f"❌ {config['description']} failed to start")
                if stderr:
                    error_msg = stderr.decode().strip()
                    if error_msg:
                        print(f"Error: {error_msg}")
                if service_name in self.services:
                    del self.services[service_name]
                # Invalidate cache for this service since status changed
                self.invalidate_cache(service_name)
                return False
                
        except Exception as e:
            print(f"❌ Failed to start {service_name}: {e}")
            if service_name in self.services:
                del self.services[service_name]
            # Invalidate cache for this service since status changed
            self.invalidate_cache(service_name)
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a specific service (both managed and independent)"""
        config = self.service_configs[service_name]
        
        # First try to stop if managed by this process
        if service_name in self.services:
            service_info = self.services[service_name]
            process = service_info["process"]
            
            print(f"🛑 Stopping managed {config['description']}...")
            
            try:
                # Try graceful shutdown first
                process.terminate()
                
                # Wait up to 5 seconds for graceful shutdown (reduced from 10)
                try:
                    process.wait(timeout=5)
                    print(f"✅ {service_name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    process.kill()
                    process.wait()
                    print(f"⚡ {service_name} force killed")
                
                del self.services[service_name]
                # Invalidate cache since service status changed
                self.invalidate_cache(service_name)
                return True
                
            except Exception as e:
                print(f"❌ Failed to stop managed service {service_name}: {e}")
                return False
        
        # Try to stop independent service by finding its process
        else:
            print(f"🛑 Stopping independent {config['description']}...")
            try:
                import psutil
                port = config['port']
                
                # Find process using the port
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        connections = proc.connections()
                        for conn in connections:
                            if conn.laddr.port == port:
                                print(f"🎯 Found process {proc.pid} using port {port}")
                                proc.terminate()
                                proc.wait(timeout=5)
                                print(f"✅ Independent {service_name} stopped")
                                # Invalidate cache since service status changed
                                self.invalidate_cache(service_name)
                                return True
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                
                print(f"⚠️ No process found using port {port}")
                return True
                
            except ImportError:
                print("⚠️ psutil not available - cannot stop independent services")
                print("   Install with: pip install psutil")
                return False
            except Exception as e:
                print(f"❌ Failed to stop independent service: {e}")
                return False
    
    def check_service_health(self, service_name: str, timeout: float = 2, use_cache: bool = True) -> bool:
        """Check if a service is healthy (with caching for better UX)"""
        
        # Check cache first if enabled
        if use_cache and service_name in self._status_cache_time:
            cache_age = time.time() - self._status_cache_time[service_name]
            if cache_age < self._cache_duration:
                return self._status_cache.get(service_name, False)
        
        config = self.service_configs[service_name]
        
        try:
            response = requests.get(f"http://localhost:{config['port']}/health", timeout=timeout)
            is_healthy = response.status_code == 200
            
            # Update cache
            if use_cache:
                self._status_cache[service_name] = is_healthy
                self._status_cache_time[service_name] = time.time()
            
            return is_healthy
        except:
            # Update cache with failure
            if use_cache:
                self._status_cache[service_name] = False
                self._status_cache_time[service_name] = time.time()
            return False
    
    def invalidate_cache(self, service_name: str = None):
        """Invalidate status cache for specific service or all services"""
        if service_name:
            self._status_cache.pop(service_name, None)
            self._status_cache_time.pop(service_name, None)
        else:
            self._status_cache.clear()
            self._status_cache_time.clear()
    
    def get_service_status(self, fast_mode: bool = False) -> dict:
        """Get status of all services (including those running independently)"""
        status = {}
        
        # In fast mode, use shorter timeouts and more aggressive caching
        timeout = 1 if fast_mode else 2
        
        for service_name, config in self.service_configs.items():
            if service_name in self.services:
                # Service started by this manager
                service_info = self.services[service_name]
                process = service_info["process"]
                
                # Check if process is running
                if process.poll() is None:
                    # Check health endpoint
                    healthy = self.check_service_health(service_name, timeout=timeout)
                    status[service_name] = {
                        "status": "healthy" if healthy else "unhealthy",
                        "port": config["port"],
                        "uptime": time.time() - service_info["start_time"],
                        "description": config["description"],
                        "managed": True
                    }
                else:
                    status[service_name] = {
                        "status": "crashed",
                        "port": config["port"], 
                        "description": config["description"],
                        "managed": True
                    }
            else:
                # Check if service is running independently
                healthy = self.check_service_health(service_name, timeout=timeout)
                if healthy:
                    status[service_name] = {
                        "status": "healthy",
                        "port": config["port"],
                        "description": config["description"],
                        "managed": False  # Running independently
                    }
                else:
                    status[service_name] = {
                        "status": "stopped",
                        "port": config["port"],
                        "description": config["description"],
                        "managed": False
                    }
        
        return status
    
    def show_status(self):
        """Show current status of all services with improved formatting"""
        print("\n📊 Service Status:")
        print("-" * 60)
        print("🔍 Searching for service status...")
        
        status = self.get_service_status(fast_mode=True)
        print("✅ Status check complete!")
        print()
        
        for service_name, info in status.items():
            status_icon = {
                "healthy": "✅",
                "unhealthy": "⚠️",
                "crashed": "❌", 
                "stopped": "⏹️"
            }.get(info["status"], "❓")
            
            # Extract service name without descriptive text for cleaner display
            clean_name = info['description'].replace(" Service", "").replace(" (OpenAI)", "").replace(" (Fast)", "").replace(" (High Quality)", "")
            
            # Format with aligned columns
            port_text = f"(Port {info['port']}):"
            status_text = info['status']
            
            # Add managed indicator only for healthy services
            if info["status"] == "healthy":
                managed_indicator = " (Managed)" if info.get("managed", False) else " (Independent)"
                status_text += managed_indicator
            
            # Align columns: Service name (25 chars), Port (15 chars), Status
            print(f"  {status_icon} {clean_name:<23} {port_text:<15} {status_text}")
            
            # Show uptime for healthy managed services
            if info["status"] == "healthy" and "uptime" in info and info.get("managed", False):
                uptime_min = info["uptime"] / 60
                print(f"      ⏱️  Uptime: {uptime_min:.1f} minutes")
        
        print()  # Extra line for spacing
    
    def start_combination(self, combo_type: str):
        """Start a predefined service combination"""
        if combo_type not in self.combinations:
            print(f"❌ Unknown combination: {combo_type}")
            return
        
        combo = self.combinations[combo_type]
        print(f"\n🚀 Starting {combo['name']}")
        print(f"Description: {combo['description']}")
        print(f"Use case: {combo['use_case']}")
        print("-" * 50)
        
        success_count = 0
        for service_name in combo["services"]:
            if self.start_service(service_name):
                success_count += 1
            time.sleep(1)  # Brief pause between starts
        
        print(f"\n📊 Results: {success_count}/{len(combo['services'])} services started")
        
        if success_count == len(combo["services"]):
            print(f"🎉 {combo['name']} is ready!")
            self.show_service_urls(combo["services"])
        else:
            print("⚠️ Some services failed to start. Check the status for details.")
    
    def show_service_urls(self, service_names: list):
        """Show URLs for specific services"""
        print(f"\n🔗 Service URLs:")
        for service_name in service_names:
            if service_name in self.services:
                config = self.service_configs[service_name]
                print(f"  {config['description']}: http://localhost:{config['port']}")
    
    def stop_all_services(self):
        """Stop all running services"""
        if not self.services:
            print("\n⏹️ No services are currently running")
            return
        
        print(f"\n🛑 Stopping all {len(self.services)} services...")
        
        for service_name in list(self.services.keys()):
            self.stop_service(service_name)
        
        print("✅ All services stopped")
    
    def test_running_services(self):
        """Test all currently running services"""
        running_services = list(self.services.keys())
        
        if not running_services:
            print("\n⚠️ No services are currently running")
            return
        
        print(f"\n🧪 Testing {len(running_services)} running services...")
        print("-" * 50)
        
        # Basic health checks
        for service_name in running_services:
            config = self.service_configs[service_name]
            healthy = self.check_service_health(service_name)
            status_icon = "✅" if healthy else "❌"
            print(f"  {status_icon} {config['description']} (Port {config['port']})")
        
        # Test service functionality if available
        self.test_service_functionality()
    
    def test_service_functionality(self):
        """Test basic functionality of running services"""
        print(f"\n🔬 Testing Service Functionality:")
        print("-" * 30)
        
        # Test STT if running
        if "stt" in self.services:
            self.test_stt_service()
        
        # Test TTS services
        for tts_service in ["kokoro_tts", "hira_dia_tts"]:
            if tts_service in self.services:
                self.test_tts_service(tts_service)
        
        # Test LLM services
        for llm_service in ["mistral_llm", "gpt_llm"]:
            if llm_service in self.services:
                self.test_llm_service(llm_service)
    
    def test_stt_service(self):
        """Test STT service functionality"""
        try:
            # Create fake audio data
            fake_audio = b"fake wav audio data"
            files = {"audio": ("test.wav", fake_audio, "audio/wav")}
            
            response = requests.post("http://localhost:8001/transcribe", files=files, timeout=10)
            
            if response.status_code == 200:
                print("  ✅ STT Service: Basic functionality test passed")
            else:
                print(f"  ❌ STT Service: Test failed (Status: {response.status_code})")
        except Exception as e:
            print(f"  ❌ STT Service: Test error - {e}")
    
    def test_tts_service(self, service_name: str):
        """Test TTS service functionality"""
        try:
            config = self.service_configs[service_name]
            port = config["port"]
            
            payload = {
                "text": "Hello, this is a test.",
                "return_audio": True
            }
            
            response = requests.post(f"http://localhost:{port}/synthesize", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("audio_base64"):
                    print(f"  ✅ {config['description']}: Audio generation test passed")
                else:
                    print(f"  ⚠️ {config['description']}: No audio data returned")
            else:
                print(f"  ❌ {config['description']}: Test failed (Status: {response.status_code})")
        except Exception as e:
            print(f"  ❌ {config['description']}: Test error - {e}")
    
    def test_llm_service(self, service_name: str):
        """Test LLM service functionality"""
        try:
            config = self.service_configs[service_name]
            port = config["port"]
            
            payload = {
                "text": "Hello, how are you?",
                "use_cache": True
            }
            
            response = requests.post(f"http://localhost:{port}/generate", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("response"):
                    print(f"  ✅ {config['description']}: Text generation test passed")
                else:
                    print(f"  ⚠️ {config['description']}: No response text returned")
            else:
                print(f"  ❌ {config['description']}: Test failed (Status: {response.status_code})")
        except Exception as e:
            print(f"  ❌ {config['description']}: Test error - {e}")
    
    def run_interactive(self):
        """Run interactive service management with numbered menu (following existing test patterns)"""
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        while self.running:
            try:
                self.show_main_menu()
                choice = input("\nEnter your choice (0-9): ").strip()
                
                if choice == "0":
                    print("👋 Goodbye!")
                    break
                elif choice == "1":
                    self.show_status()
                elif choice == "2":
                    self.start_combination("fast")
                elif choice == "3":
                    self.start_combination("balanced")
                elif choice == "4":
                    self.start_combination("efficient")
                elif choice == "5":
                    self.start_combination("quality")
                elif choice == "6":
                    self.manage_individual_services()
                elif choice == "7":
                    self.stop_all_services()
                elif choice == "8":
                    self.test_running_services()
                elif choice == "9":
                    self.launch_comprehensive_tests()
                else:
                    print("❌ Invalid choice. Please enter 0-9.")
                
                # Only pause for status/info displays, not for interactive actions
                if choice in ["1"]:  # Only for status display
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        self.stop_all_services()
    
    def show_main_menu(self):
        """Show the main menu following existing test patterns"""
        print("\n🎭 Enhanced Independent Microservices Manager")
        print("=" * 60)
        
        # Check cache status
        current_time = time.time()
        cached_count = 0
        for service_name in self.service_configs.keys():
            if service_name in self._status_cache_time:
                cache_age = current_time - self._status_cache_time[service_name]
                if cache_age < self._cache_duration:
                    cached_count += 1
        
        # Only show loading if we have no cached data or very little
        show_loading = cached_count == 0
        
        if show_loading:
            print("🔍 Searching for service status...")
        else:
            print("📋 Using cached service status...")
        
        # Get status with fast mode and caching
        status = self.get_service_status(fast_mode=True)
        
        if show_loading:
            print("✅ Status check complete!")
        
        running_services = [info["description"] for name, info in status.items() if info["status"] == "healthy"]
        
        if running_services:
            print(f"🟢 Currently running: {', '.join(running_services)}")
        else:
            print("🔴 No services currently running")
        
        print("\n📋 Service Combinations (Following existing patterns):")
        print("  1. Show detailed service status")
        print("  2. Start Fast Combo (Orchestrator + Whisper STT + Kokoro TTS + Mistral LLM)")
        print("  3. Start Balanced Combo (Orchestrator + Whisper STT + Kokoro TTS + GPT LLM)")
        print("  4. Start Efficient Combo (Orchestrator + Whisper STT + Hira Dia TTS + Mistral LLM)")
        print("  5. Start Quality Combo (Orchestrator + Whisper STT + Hira Dia TTS + GPT LLM)")
        
        print("\n⚙️  Service Management:")
        print("  6. Manage individual services")
        print("  7. Stop all services")
        print("  8. Test running services")
        print("  9. Launch comprehensive test suite")
        
        print("\n  0. Exit")
    
    def manage_individual_services(self):
        """Manage individual services with numbered menu (optimized for speed with caching)"""
        while True:
            print("\n🔧 Individual Service Management")
            print("-" * 40)
            
            # Check cache status
            current_time = time.time()
            cached_count = 0
            for service_name in self.service_configs.keys():
                if service_name in self._status_cache_time:
                    cache_age = current_time - self._status_cache_time[service_name]
                    if cache_age < self._cache_duration:
                        cached_count += 1
            
            # Only show loading if we have no cached data
            show_loading = cached_count == 0
            
            if show_loading:
                print("🔍 Searching for service status...")
            else:
                print("📋 Using cached service status...")
            
            # Use cached status for faster response (this will use cache if available)
            status = self.get_service_status(fast_mode=True)
            
            if show_loading:
                print("✅ Status check complete!")
            
            service_list = list(self.service_configs.items())
            
            # Create simplified service names without descriptions
            service_names = {
                "orchestrator": "Orchestrator",
                "whisper_stt": "Whisper STT", 
                "kokoro_tts": "Kokoro TTS",
                "hira_dia_tts": "Hira Dia TTS",
                "mistral_llm": "Mistral LLM",
                "gpt_llm": "GPT LLM"
            }
            
            for i, (service_name, config) in enumerate(service_list, 1):
                service_status = status[service_name]["status"]
                status_icon = {
                    "healthy": "✅",
                    "unhealthy": "⚠️",
                    "crashed": "❌",
                    "stopped": "⏹️"
                }.get(service_status, "❓")
                
                action = "Stop" if service_status in ["healthy", "unhealthy"] else "Start"
                
                # Simplified formatting like the example with adjusted spacing
                simple_name = service_names.get(service_name, config['description'])
                service_desc = f"{action} {simple_name}"
                
                # Add extra space for "stopped" status to match "healthy" alignment
                if service_status == "stopped":
                    status_text = f"({status_icon}  {service_status})"  # Extra space for stopped
                else:
                    status_text = f"({status_icon} {service_status})"   # Single space for others
                
                # Align the status text to column 40
                print(f"  {i}. {service_desc:<35} {status_text}")
            
            print("  r. Refresh status (clear cache)")
            print("  0. Back to main menu")
            
            try:
                choice = input(f"\nSelect service (0-{len(service_list)}, r): ").strip().lower()
                
                if choice == "0":
                    break
                elif choice == "r":
                    self.invalidate_cache()
                    print("✅ Status cache cleared - refreshing...")
                    continue
                elif choice.isdigit() and 1 <= int(choice) <= len(service_list):
                    choice_num = int(choice)
                    service_name = service_list[choice_num - 1][0]
                    config = service_list[choice_num - 1][1]
                    service_status = status[service_name]["status"]
                    simple_name = service_names.get(service_name, config['description'])
                    
                    print(f"\n🎯 Managing {simple_name}...")
                    
                    if service_status in ["healthy", "unhealthy"]:
                        success = self.stop_service(service_name)
                        if success:
                            print(f"✅ {simple_name} stopped successfully")
                        else:
                            print(f"❌ Failed to stop {simple_name}")
                    else:
                        success = self.start_service(service_name)
                        if success:
                            print(f"✅ {simple_name} started successfully")
                        else:
                            print(f"❌ Failed to start {simple_name}")
                    
                    # Auto-refresh status after any start/stop operation
                    print("🔄 Refreshing status...")
                    self.invalidate_cache()
                    time.sleep(1)  # Give service a moment to change state
                    continue  # This will refresh the menu automatically
                else:
                    print("❌ Invalid choice")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Returning to main menu...")
                break
    
    def launch_comprehensive_tests(self):
        """Launch comprehensive test suite following existing patterns"""
        print("\n🧪 Launching Comprehensive Test Suite")
        print("-" * 50)
        
        try:
            # Try to run the existing test script
            test_script = project_root / "test_independent_services.py"
            
            if test_script.exists():
                print("🔄 Running comprehensive test script...")
                subprocess.run([sys.executable, str(test_script)], cwd=project_root)
            else:
                print("⚠️ Comprehensive test script not found, running basic tests...")
                self.test_running_services()
        except Exception as e:
            print(f"❌ Test launch failed: {e}")
            print("Running basic tests instead...")
            self.test_running_services()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.running = False
        self.stop_all_services()

def main():
    """Main entry point following existing patterns"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Independent Microservices Manager")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode (default)")
    parser.add_argument("--start", nargs="+", choices=["fast", "quality", "balanced", "efficient"], 
                        help="Start specific combination")
    parser.add_argument("--status", action="store_true", help="Show service status")
    parser.add_argument("--test", action="store_true", help="Test running services")
    
    args = parser.parse_args()
    
    manager = EnhancedServiceManager()
    
    try:
        if args.status:
            manager.show_status()
        elif args.test:
            manager.test_running_services()
        elif args.start:
            for combo in args.start:
                manager.start_combination(combo)
            print("\n🔄 Services running. Press Ctrl+C to stop all.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                manager.stop_all_services()
        else:
            # Default: interactive mode (following existing patterns)
            manager.run_interactive()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
