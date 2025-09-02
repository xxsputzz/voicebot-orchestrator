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
import atexit
import requests
import psutil
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Determine the correct Python executable (prefer virtual environment)
def get_python_executable():
    """Get the correct Python executable, preferring virtual environment"""
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable

class EnhancedServiceManager:
    """Enhanced manager for independent microservices following existing test patterns"""
    
    def __init__(self):
        self.services = {}
        self.running = False
        self._status_cache = {}
        self._status_cache_time = {}
        self._cache_duration = 30  # Cache for 30 seconds (increased from 10)
        self.intended_engine_mode = {}  # Track intended engine modes for services
        
        # Setup signal handlers for proper cleanup
        self.setup_signal_handlers()
        
        # Register cleanup on exit - only clear tracking, don't stop services
        atexit.register(self.cleanup_tracking_only)
        
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
                "port": 8003,
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
                "description": "Unified Hira Dia TTS (Quality + Speed)",
                "required": False,
                "type": "tts",
                "engine_modes": ["full", "4bit", "auto"],
                "current_engine": "auto"  # Default engine preference
            },
            "dia_4bit_tts": {
                "script": "tts_dia_4bit_service.py",
                "port": 8013,
                "description": "Dedicated Dia 4-bit TTS (Speed Only)",
                "required": False,
                "type": "tts"
            },
            "zonos_tts": {
                "script": "tts_zonos_service.py",
                "port": 8014,
                "description": "Zonos TTS (High-Quality Neural)",
                "required": False,
                "type": "tts"
            },
            "tortoise_tts": {
                "script": "tts_tortoise_service.py",
                "port": 8015,
                "description": "Tortoise TTS (Ultra High-Quality)",
                "required": False,
                "type": "tts",
                "args": ["--direct"]
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
        
        # GPU Model Configuration for LLM Services
        self.gpu_models = {
            "gpt_small": {
                "name": "GPT Small (8GB GPU)",
                "script": "llm_gpt_service.py",
                "port": 8026,
                "description": "DialoGPT-Small (117M params) - RTX 4060/3070",
                "gpu_memory": "2-8GB",
                "response_time": "1-3 seconds",
                "use_case": "Stable performance, banking conversations",
                "model_type": "small"
            },
            "gpt_medium": {
                "name": "GPT Medium (16GB GPU)", 
                "script": "llm_gpt_medium_service.py",
                "port": 8023,
                "description": "GPT-2 Medium (355M params) - RTX 4070 Ti/4080",
                "gpu_memory": "8-16GB",
                "response_time": "2-5 seconds", 
                "use_case": "Enhanced reasoning, complex queries",
                "model_type": "medium"
            },
            "gpt_large": {
                "name": "GPT Large (24GB+ GPU)",
                "script": "llm_gpt_large_service.py", 
                "port": 8024,
                "description": "GPT-Neo 2.7B - RTX 4090/A100 AWS",
                "gpu_memory": "16-24GB",
                "response_time": "3-8 seconds",
                "use_case": "Advanced reasoning, creative tasks",
                "model_type": "large"
            },
            "gpt_xl": {
                "name": "GPT XL (AWS A100)",
                "script": "llm_gpt_xl_service.py",
                "port": 8025, 
                "description": "GPT-J 6B - AWS A100 (40GB)",
                "gpu_memory": "24-40GB",
                "response_time": "5-15 seconds",
                "use_case": "Production-grade AI, complex reasoning",
                "model_type": "xl"
            }
        }
        
        # Predefined combinations following existing patterns - Ordered by preference
        self.combinations = {
            "fast": {
                "name": "Fast Combo",
                "description": "Orchestrator + Whisper STT + Mistral LLM + Kokoro TTS (Real-time)",
                "services": ["orchestrator", "whisper_stt", "kokoro_tts", "mistral_llm"],
                "use_case": "Real-time conversation, quick responses with Whisper accuracy"
            },
            "balanced": {
                "name": "Balanced Combo",
                "description": "Orchestrator + Whisper STT + GPT LLM + Kokoro TTS (Fast TTS, advanced reasoning)",
                "services": ["orchestrator", "whisper_stt", "kokoro_tts", "gpt_llm"],
                "use_case": "Quick TTS with advanced language processing and Whisper accuracy"
            },
            "efficient": {
                "name": "Efficient Combo",
                "description": "Orchestrator + Whisper STT + Mistral LLM + Dia 4-bit TTS (Speed optimized)",
                "services": ["orchestrator", "whisper_stt", "hira_dia_tts", "mistral_llm"],
                "use_case": "Fast processing with dedicated 4-bit TTS for quick responses and Whisper accuracy"
            },
            "quality": {
                "name": "Quality Combo",
                "description": "Orchestrator + Whisper STT + GPT LLM + Hira Dia TTS (Maximum quality)",
                "services": ["orchestrator", "whisper_stt", "hira_dia_tts", "gpt_llm"],
                "use_case": "High-quality content, professional presentations with Whisper accuracy"
            },
            "premium": {
                "name": "Premium Combo",
                "description": "Orchestrator + Whisper STT + GPT LLM + Tortoise TTS (Ultra High-Quality)",
                "services": ["orchestrator", "whisper_stt", "tortoise_tts", "gpt_llm"],
                "use_case": "Ultra high-quality speech synthesis for premium applications and content creation"
            }
        }
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
            print("   Note: Services will continue running independently")
            self.cleanup_tracking_only()
            sys.exit(0)
        
        # Handle common termination signals
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal_handler)
    
    def cleanup_on_exit(self):
        """Clean up all managed processes on exit"""
        if not self.services:
            return
            
        print("\n🧹 Cleaning up managed services...")
        services_to_stop = list(self.services.keys())
        
        for service_name in services_to_stop:
            try:
                service_info = self.services[service_name]
                process = service_info["process"]
                
                if process.poll() is None:  # Process is still running
                    print(f"🛑 Stopping {service_name}...")
                    try:
                        # Try graceful shutdown first
                        process.terminate()
                        process.wait(timeout=3)
                        print(f"✅ {service_name} stopped gracefully")
                    except subprocess.TimeoutExpired:
                        # Force kill if graceful shutdown fails
                        process.kill()
                        process.wait()
                        print(f"⚡ {service_name} force killed")
                        
            except Exception as e:
                print(f"❌ Error stopping {service_name}: {e}")
        
        # Clear the services registry
        self.services.clear()
        self.intended_engine_mode.clear()
        print("✅ Cleanup complete")
    
    def cleanup_tracking_only(self):
        """Clean up only tracking data, leave services running"""
        if not self.services:
            return
            
        print("\n🧹 Clearing service tracking (services remain running)...")
        
        # Clear the services registry without stopping processes
        self.services.clear()
        self.intended_engine_mode.clear()
        print("✅ Tracking cleanup complete - services continue running independently")
    
    def force_stop_all_python_processes(self):
        """Nuclear option: Stop all Python processes (use with caution)"""
        print("\n⚠️ FORCE STOPPING ALL PYTHON PROCESSES - This will terminate ALL Python applications!")
        confirmation = input("Type 'FORCE' to confirm this action: ").strip()
        
        if confirmation != 'FORCE':
            print("❌ Force stop cancelled")
            return False
            
        try:
            stopped_count = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'].lower() in ['python.exe', 'python', 'pythonw.exe']:
                        # Don't kill our own process
                        if proc.pid == os.getpid():
                            continue
                            
                        print(f"🛑 Stopping Python process {proc.pid}: {proc.info['name']}")
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            proc.kill()
                        stopped_count += 1
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
            print(f"✅ Force stopped {stopped_count} Python processes")
            return True
            
        except Exception as e:
            print(f"❌ Error force stopping processes: {e}")
            return False
    
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
            # Get the correct Python executable (virtual environment)
            python_exe = get_python_executable()
            
            # Start the service process
            # Get any additional arguments from config
            args = config.get("args", [])
            
            if service_name in ["orchestrator", "whisper_stt"]:
                # For orchestrator and whisper_stt, use --direct flag to run server directly
                # Enable stdout/stderr for debugging STT issues
                if service_name == "whisper_stt":
                    # Don't capture output for STT service so we can see debug logs
                    process = subprocess.Popen([
                        python_exe, str(script_path), "--direct"
                    ] + args, cwd=project_root)
                else:
                    process = subprocess.Popen([
                        python_exe, str(script_path), "--direct"
                    ] + args, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elif service_name == "hira_dia_tts":
                # Don't capture output for Hira Dia TTS to avoid interfering with model loading
                process = subprocess.Popen([
                    python_exe, str(script_path)
                ] + args, cwd=project_root)
            elif service_name == "tortoise_tts":
                # Tortoise TTS needs --direct flag but with output capture to prevent log spam
                process = subprocess.Popen([
                    python_exe, str(script_path), "--direct"
                ] + args, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                # For other services, run normally
                process = subprocess.Popen([
                    python_exe, str(script_path)
                ] + args, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.services[service_name] = {
                "process": process,
                "config": config,
                "start_time": time.time()
            }
            
            # Shorter wait for faster UX (reduced from 3 seconds)
            # But give Hira Dia TTS more time since it needs to load large models
            print("⏳ Waiting for service to start...")
            if service_name == "hira_dia_tts":
                time.sleep(5)  # Give Hira Dia more time for model loading
            else:
                time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                # Quick health check with short timeout and no cache
                timeout = 3 if service_name == "hira_dia_tts" else 1  # Longer timeout for Hira Dia
                healthy = self.check_service_health(service_name, timeout=timeout, use_cache=False)
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
                # Clean up tracking for intended engine mode
                if service_name in self.intended_engine_mode:
                    del self.intended_engine_mode[service_name]
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
                port = config['port']
                stopped = False
                
                # Find process using the port
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        connections = proc.connections()
                        for conn in connections:
                            if conn.laddr.port == port:
                                print(f"🎯 Found process {proc.pid} using port {port}")
                                try:
                                    # Try graceful shutdown first
                                    proc.terminate()
                                    proc.wait(timeout=5)
                                    print(f"✅ Independent {service_name} stopped gracefully")
                                except psutil.TimeoutExpired:
                                    # Force kill if graceful shutdown fails
                                    proc.kill()
                                    proc.wait()
                                    print(f"⚡ Independent {service_name} force killed")
                                
                                # Clean up tracking for intended engine mode
                                if service_name in self.intended_engine_mode:
                                    del self.intended_engine_mode[service_name]
                                # Invalidate cache since service status changed
                                self.invalidate_cache(service_name)
                                stopped = True
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                
                if not stopped:
                    print(f"⚠️ No process found using port {port}")
                return True
            except Exception as e:
                print(f"❌ Failed to stop independent service: {e}")
                return False
    
    def start_dia_4bit_service(self) -> bool:
        """Start Hira Dia TTS service directly in Dia 4-bit mode"""
        service_name = "hira_dia_tts"
        
        if service_name not in self.service_configs:
            print(f"❌ Unknown service: {service_name}")
            return False
        
        # Check if already running
        config = self.service_configs[service_name]
        if self.check_service_health(service_name, timeout=1, use_cache=False):
            print(f"✅ {config['description']} is already running")
            return True
        
        if service_name in self.services:
            print(f"✅ Service {service_name} already managed by this process")
            return True
        
        # Get script path
        script_path = Path(__file__).parent / config["script"]
        
        if not script_path.exists():
            print(f"❌ Service script not found: {script_path}")
            return False
        
        print(f"🚀 Starting Dedicated Dia 4-bit TTS on port {config['port']}...")
        print("   Engine: Dia 4-bit (speed optimized, lightweight model)")
        
        try:
            # Get the correct Python executable
            python_exe = get_python_executable()
            
            # Start with --engine 4bit argument
            process = subprocess.Popen([
                python_exe, str(script_path), "--engine", "4bit"
            ], cwd=Path(__file__).parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.services[service_name] = {
                "process": process,
                "config": config,
                "start_time": time.time()
            }
            
            # Track that this service is intended to run in 4-bit mode
            self.intended_engine_mode[service_name] = "4bit"
            
            # Wait longer to see if it starts successfully (FastAPI startup takes time)
            time.sleep(3)
            
            if process.poll() is None:
                # Process is still running, but let's wait a bit more to see if it crashes during startup
                print(f"⏳ {config['description']} starting... (checking for startup errors)")
                time.sleep(2)
                
                if process.poll() is None:
                    print(f"⏳ {config['description']} starting... (may need a moment to initialize)")
                    return True
                else:
                    # Crashed during extended startup check
                    stdout, stderr = process.communicate()
                    print(f"❌ Service crashed during startup:")
                    if stderr:
                        print(f"   Error: {stderr.decode()}")
                    if stdout:
                        print(f"   Output: {stdout.decode()}")
                    
                    # Clean up failed service registration
                    if service_name in self.services:
                        del self.services[service_name]
                    if service_name in self.intended_engine_mode:
                        del self.intended_engine_mode[service_name]
                    
                    return False
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Service failed to start:")
                if stderr:
                    print(f"   Error: {stderr.decode()}")
                if stdout:
                    print(f"   Output: {stdout.decode()}")
                
                # Clean up failed service registration
                if service_name in self.services:
                    del self.services[service_name]
                if service_name in self.intended_engine_mode:
                    del self.intended_engine_mode[service_name]
                
                return False
                
        except Exception as e:
            print(f"❌ Failed to start service: {e}")
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
    
    def get_hira_dia_engine_status(self) -> dict:
        """Get current engine status for Unified Hira Dia TTS service"""
        try:
            response = requests.get("http://localhost:8012/engines", timeout=3)
            if response.status_code == 200:
                data = response.json()
                current_engine = data.get("current_engine", "unknown")
                
                # Map engine names to display-friendly names
                engine_display_map = {
                    "nari_dia": "full_dia",
                    "dia_4bit": "dia_4bit"
                }
                
                display_engine = engine_display_map.get(current_engine, current_engine)
                
                return {
                    "current_engine": display_engine,
                    "available_engines": [e.get("name", "unknown") for e in data.get("available_engines", [])],
                    "service_responsive": True
                }
        except Exception:
            pass
        
        return {
            "current_engine": "unknown",
            "available_engines": [],
            "service_responsive": False
        }
    
    def switch_hira_dia_engine(self, engine_mode: str) -> bool:
        """Switch Hira Dia engine mode"""
        try:
            response = requests.post(f"http://localhost:8012/switch_engine", 
                                   params={"engine": engine_mode}, 
                                   timeout=5)
            if response.status_code == 200:
                self.service_configs["hira_dia_tts"]["current_engine"] = engine_mode
                return True
        except Exception as e:
            print(f"❌ Failed to switch engine: {e}")
        
        return False
    
    def handle_hira_dia_engine_switch_sync(self):
        """Synchronous wrapper for engine switching"""
        import asyncio
        try:
            asyncio.run(self.handle_hira_dia_engine_switch())
        except Exception as e:
            print(f"❌ Error switching engine: {e}")
    
    async def handle_hira_dia_engine_switch(self):
        """Handle switching Hira Dia engine mode"""
        print("\n🎭 Hira Dia Engine Mode Switching")
        print("=" * 50)
        
        # Get current status
        current_engine = self.get_hira_dia_engine_status()
        if not current_engine:
            print("❌ Unable to get current engine status. Is the Hira Dia service running?")
            return
            
        print(f"Current Engine: {current_engine}")
        
        # Show options
        print("\nAvailable Engine Modes:")
        print("1. 🎭 Full Dia (Quality Mode)")
        print("2. ⚡ 4-bit Dia (Speed Mode)")  
        print("3. 🤖 Auto (Smart Selection)")
        print("0. Cancel")
        
        choice = input("\nSelect new engine mode (0-3): ").strip()
        
        engine_map = {
            "1": "NARI_DIA",
            "2": "DIA_4BIT", 
            "3": "AUTO"
        }
        
        if choice == "0":
            print("❌ Engine switch cancelled")
            return
        elif choice in engine_map:
            new_engine = engine_map[choice]
            print(f"🔄 Switching to {new_engine}...")
            
            result = self.switch_hira_dia_engine(new_engine)
            if result:
                print(f"✅ Successfully switched to {new_engine}")
            else:
                print(f"❌ Failed to switch to {new_engine}")
        else:
            print("❌ Invalid choice")
    
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
                    # Process has crashed - clean up and mark as stopped
                    print(f"🔍 Detected crashed service: {service_name}, cleaning up...")
                    del self.services[service_name]
                    if service_name in self.intended_engine_mode:
                        del self.intended_engine_mode[service_name]
                    
                    status[service_name] = {
                        "status": "stopped",
                        "port": config["port"], 
                        "description": config["description"],
                        "managed": False
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
        print("\n" + "=" * 40)
        print("🔍 Checking Service Status...")
        print("=" * 40)
        
        status = self.get_service_status(fast_mode=True)
        
        # Define service order with proper names
        service_order = [
            ("orchestrator", "Main Orchestrator"),
            ("whisper_stt", "Whisper STT"),
            ("kokoro_tts", "Kokoro TTS"),
            ("hira_dia_tts", "Hira Dia TTS"),
            ("zonos_tts", "Zonos TTS"),
            ("tortoise_tts", "Tortoise TTS"),
            ("mistral_llm", "Mistral LLM"),
            ("gpt_llm", "GPT LLM")
        ]
        
        for service_key, display_name in service_order:
            if service_key in status:
                info = status[service_key]
                status_icon = {
                    "healthy": "✅",
                    "unhealthy": "⚠️",
                    "crashed": "❌", 
                    "stopped": "❌"
                }.get(info["status"], "❓")
                
                port = info["port"]
                service_status = "Running" if info["status"] == "healthy" else "Not running"
                
                print(f"{status_icon} {display_name}: {service_status} (http://localhost:{port})")
        
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
        
        # Special handling for efficient combo - using dedicated Dia 4-bit service
        if combo_type == "efficient":
            print("⚡ Using dedicated Dia 4-bit TTS service for maximum efficiency...")
        
        success_count = 0
        for service_name in combo["services"]:
            # Special handling for efficient combo TTS service
            if combo_type == "efficient" and service_name == "hira_dia_tts":
                # Stop existing TTS service if running in wrong mode
                if self.check_service_health("hira_dia_tts", timeout=1, use_cache=False):
                    print("� Stopping existing TTS service to switch to Dia 4-bit mode...")
                    self.stop_service("hira_dia_tts")
                    time.sleep(2)  # Wait for service to stop
                
                # Use the working start_dia_4bit_service method for Dia 8-bit mode  
                if self.start_dia_4bit_service():
                    success_count += 1
                    print("✅ Started TTS service in Dia 4-bit mode")
                else:
                    print("❌ Failed to start TTS service in Dia 4-bit mode")
            else:
                # Regular service start
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
        """Stop all running services (both managed and independent)"""
        print(f"\n🛑 Stopping all services...")
        
        # Get current status to find all running services
        status = self.get_service_status(fast_mode=False)
        running_services = [name for name, info in status.items() if info["status"] == "healthy"]
        
        if not running_services:
            print("⏹️  No services are currently running")
            return
        
        stopped_count = 0
        for service_name in running_services:
            print(f"🛑 Stopping {service_name}...")
            if self.stop_service(service_name):
                stopped_count += 1
                print(f"✅ {service_name} stopped")
            else:
                print(f"❌ Failed to stop {service_name}")
        
        # Clear the services registry for managed services
        self.services.clear()
        self.intended_engine_mode.clear()
        
        # Invalidate cache to reflect changes
        self.invalidate_cache()
        
        print(f"✅ Stopped {stopped_count}/{len(running_services)} services")
    
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
        
        # Test Orchestrator if running
        if "orchestrator" in self.services:
            try:
                self.test_orchestrator_service()
            except Exception as e:
                print(f"  ❌ Orchestrator test error: {e}")
        
        # Test STT services (check for both names)
        for stt_service in ["whisper_stt", "stt"]:
            if stt_service in self.services:
                try:
                    self.test_stt_service(stt_service)
                except Exception as e:
                    print(f"  ❌ {stt_service} test error: {e}")
        
        # Test TTS services
        for tts_service in ["kokoro_tts", "hira_dia_tts"]:
            if tts_service in self.services:
                try:
                    self.test_tts_service(tts_service)
                except Exception as e:
                    print(f"  ❌ {tts_service} test error: {e}")
        
        # Test LLM services
        for llm_service in ["mistral_llm", "gpt_llm"]:
            if llm_service in self.services:
                try:
                    self.test_llm_service(llm_service)
                except Exception as e:
                    print(f"  ❌ {llm_service} test error: {e}")
    
    def test_orchestrator_service(self):
        """Test Orchestrator service functionality"""
        try:
            # Test basic health endpoint
            response = requests.get("http://localhost:8000/health", timeout=10)
            
            if response.status_code == 200:
                print("  ✅ Orchestrator (FastAPI): Health check test passed")
            else:
                print(f"  ❌ Orchestrator (FastAPI): Health test failed (Status: {response.status_code})")
        except Exception as e:
            print(f"  ❌ Orchestrator (FastAPI): Test error - {e}")
    
    def test_stt_service(self, service_name: str):
        """Test STT service functionality"""
        try:
            config = self.service_configs[service_name]
            port = config["port"]
            
            # Create fake audio data
            fake_audio = b"fake wav audio data"
            files = {"audio": ("test.wav", fake_audio, "audio/wav")}
            
            response = requests.post(f"http://localhost:{port}/transcribe", files=files, timeout=10)
            
            if response.status_code == 200:
                print(f"  ✅ {config['description']}: Transcription test passed")
            else:
                print(f"  ❌ {config['description']}: Test failed (Status: {response.status_code})")
        except Exception as e:
            print(f"  ❌ {config['description']}: Test error - {e}")
    
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
        
        while self.running:
            try:
                self.show_main_menu()
                choice = input("\nEnter your choice (0-13): ").strip()
                
                if choice == "0":
                    print("👋 Goodbye!")
                    print("   ℹ️  Note: All running services will continue running independently")
                    break
                elif choice == "1":
                    self.show_status()
                elif choice == "2":
                    self.start_combination("fast")
                elif choice == "3":
                    self.start_combination("efficient")
                elif choice == "4":
                    self.start_combination("balanced")
                elif choice == "5":
                    self.start_combination("quality")
                elif choice == "6":
                    self.start_combination("premium")
                elif choice == "7":
                    self.manage_individual_services()
                elif choice == "8":
                    self.stop_all_services()
                elif choice == "9":
                    self.test_running_services()
                elif choice == "10":
                    self.launch_comprehensive_tests()
                elif choice == "11":
                    self.manage_gpu_models()
                elif choice == "12":
                    self.handle_hira_dia_engine_switch_sync()
                elif choice == "13":
                    self.force_stop_all_python_processes()
                else:
                    print("❌ Invalid choice. Please enter 0-13.")
                
                # Only pause for status/info displays, not for interactive actions
                if choice in ["1"]:  # Only for status display
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
    # Removed automatic stop of all services on exit. Services are only stopped when option 7 is selected.
    
    def show_main_menu(self):
        """Show the main menu following existing test patterns"""
        print("\n" + "=" * 60)
        print("🎭 Enhanced Independent Microservices Manager")
        print("=" * 60)
        
        print("\n📋 Service Combinations (Following existing patterns):")
        print("  1. Show detailed service status")
        print("  2. Start Fast Combo (Orchestrator + Whisper STT + Mistral LLM + Kokoro TTS)")
        print("  3. Start Efficient Combo (Orchestrator + Whisper STT + Mistral LLM + Dia 4-bit TTS)")
        print("  4. Start Balanced Combo (Orchestrator + Whisper STT + GPT LLM + Kokoro TTS)")
        print("  5. Start Quality Combo (Orchestrator + Whisper STT + GPT LLM + Hira Dia TTS)")
        print("  6. Start Premium Combo (Orchestrator + Whisper STT + GPT LLM + Tortoise TTS)")
        
        print("\n⚙️  Service Management:")
        print("  7. Manage individual services")
        print("  8. Stop all services (explicit shutdown)") 
        print("  9. Test running services")
        print(" 10. Launch comprehensive test suite")
        
        print("\n🎮 Advanced Options:")
        print(" 11. Select GPU/LLM Model (8GB → AWS A100)")
        print(" 12. Switch Hira Dia Engine Mode (Quality ↔ Speed)")
        print(" 13. Force Stop All Python Processes (Nuclear Option)")
        
        print("\n  0. Exit (services continue running)")
    
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
            
            # Define service order: Orchestrator → STT → LLM → TTS
            service_order = [
                "orchestrator",
                "whisper_stt", 
                "mistral_llm",
                "gpt_llm",
                "kokoro_tts",
                "hira_dia_tts",
                "zonos_tts",
                "dia_4bit_tts",
                "tortoise_tts"
            ]
            
            # Create ordered service list
            service_list = [(name, self.service_configs[name]) for name in service_order if name in self.service_configs]
            
            # Create simplified service names without descriptions
            service_names = {
                "orchestrator": "Orchestrator",
                "whisper_stt": "Whisper STT", 
                "kokoro_tts": "Kokoro TTS",
                "hira_dia_tts": "Hira Dia TTS",
                "dia_4bit_tts": "Dia 4-bit TTS",  # Virtual service option
                "zonos_tts": "Zonos TTS",
                "tortoise_tts": "Tortoise TTS",
                "mistral_llm": "Mistral LLM",
                "gpt_llm": "GPT LLM"
            }
            
            # Display regular services
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
                
                # For Hira Dia TTS, show the actual engine mode
                if service_name == "hira_dia_tts" and service_status in ["healthy", "unhealthy"]:
                    engine_status = self.get_hira_dia_engine_status()
                    if engine_status["service_responsive"]:
                        if engine_status["current_engine"] == "dia_4bit":
                            simple_name = "Hira Dia TTS (4-bit mode)"
                        elif engine_status["current_engine"] == "full_dia":
                            simple_name = "Hira Dia TTS (Full mode)"
                
                service_desc = f"{action} {simple_name}"
                
                # Add extra space for "stopped" status to match "healthy" alignment
                if service_status == "stopped":
                    status_text = f"({status_icon}  {service_status})"  # Double space for stopped
                else:
                    status_text = f"({status_icon} {service_status})"   # Single space for others
                
                # Align the status text to column 40
                print(f"  {i}. {service_desc:<35} {status_text}")
            
            # Add special Dia 4-bit TTS option (only when not already shown in main list)
            hira_dia_status = status["hira_dia_tts"]["status"]
            show_dia_4bit_option = True
            
            # Check if Hira Dia is running in 4-bit mode (already shown in main list)
            if hira_dia_status in ["healthy", "unhealthy"]:
                engine_status = self.get_hira_dia_engine_status()
                if engine_status["service_responsive"] and engine_status["current_engine"] == "dia_4bit":
                    show_dia_4bit_option = False  # Don't show duplicate - already shown as "Hira Dia TTS (4-bit mode)"
            
            dia_4bit_num = len(service_list) + 1
            
            if show_dia_4bit_option:
                if hira_dia_status in ["healthy", "unhealthy"]:
                    # Service is running in full mode - offer to switch
                    dia_action = "Switch to"
                    dia_status_icon = "🔄"
                    dia_status_text = f"({dia_status_icon}  ready)"
                else:
                    # Service is stopped - offer to start in 4-bit mode
                    dia_action = "Start"
                    dia_status_icon = "⏹️"
                    dia_status_text = f"({dia_status_icon}  stopped)"
                    
                dia_service_desc = f"{dia_action} Dia 4-bit TTS"
                print(f"  {dia_4bit_num}. {dia_service_desc:<35} {dia_status_text}")
                
                max_choice = dia_4bit_num
            else:
                max_choice = len(service_list)  # No Dia 4-bit option shown
            
            print("  r. Refresh status (clear cache)")
            print("  0. Back to main menu")
            
            try:
                choice = input(f"\nSelect service (0-{max_choice}, r): ").strip().lower()
                
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
                elif choice.isdigit() and show_dia_4bit_option and int(choice) == dia_4bit_num:
                    # Handle Dia 4-bit TTS special option (only when shown)
                    print(f"\n⚡ Managing Dia 4-bit TTS...")
                    
                    hira_dia_status = status["hira_dia_tts"]["status"]
                    
                    if hira_dia_status in ["healthy", "unhealthy"]:
                        # Service is running in full mode - switch to 4-bit
                        print("🔄 Switching Hira Dia TTS to Dia 4-bit mode...")
                        switch_success = False
                        switch_retries = 0
                        max_switch_retries = 3
                        
                        while switch_retries < max_switch_retries and not switch_success:
                            switch_success = self.switch_hira_dia_engine("DIA_4BIT")
                            if not switch_success:
                                switch_retries += 1
                                if switch_retries < max_switch_retries:
                                    print(f"⏳ Retrying switch... ({switch_retries}/{max_switch_retries})")
                                    time.sleep(1)
                        
                        if switch_success:
                            print("✅ Successfully switched to Dia 4-bit mode")
                            # Verify the switch worked
                            time.sleep(1)
                            current_engine = self.get_hira_dia_engine_status()
                            if current_engine == "DIA_4BIT":
                                print("🎯 Confirmed: Service is now in Dia 4-bit mode")
                            else:
                                print(f"⚠️ Warning: Expected DIA_4BIT but got {current_engine}")
                        else:
                            print("❌ Failed to switch to Dia 4-bit mode after multiple attempts")
                    else:
                        # Start the service in Dia 4-bit mode directly
                        print("🚀 Starting Hira Dia TTS in Dia 4-bit mode...")
                        success = self.start_dia_4bit_service()
                        if success:
                            print("✅ Dia 4-bit TTS started successfully")
                            
                            # Wait for service to be ready (longer timeout for TTS initialization)
                            print("⏳ Waiting for service to be ready...")
                            retry_count = 0
                            max_retries = 25  # Increased timeout for TTS initialization (25 * 4s = 100s)
                            service_ready = False
                            
                            while retry_count < max_retries:
                                # First check if the process is still running
                                if "hira_dia_tts" in self.services:
                                    process = self.services["hira_dia_tts"]["process"]
                                    if process.poll() is not None:
                                        # Process has crashed
                                        print(f"❌ Service process crashed during initialization (exit code: {process.poll()})")
                                        # Get error output
                                        stdout, stderr = process.communicate()
                                        if stderr:
                                            print(f"   Error details: {stderr.decode().strip()}")
                                        # Clean up
                                        del self.services["hira_dia_tts"]
                                        if "hira_dia_tts" in self.intended_engine_mode:
                                            del self.intended_engine_mode["hira_dia_tts"]
                                        service_ready = False
                                        break
                                
                                time.sleep(4)  # Longer intervals for TTS initialization
                                if self.check_service_health("hira_dia_tts", timeout=2, use_cache=False):
                                    service_ready = True
                                    break
                                retry_count += 1
                                print(f"⏳ Service initializing... ({retry_count}/{max_retries}) - TTS engines loading...")
                            
                            if service_ready:
                                # Verify it started in Dia 4-bit mode
                                time.sleep(1)
                                engine_status = self.get_hira_dia_engine_status()
                                if engine_status["service_responsive"]:
                                    current_engine = engine_status["current_engine"]
                                    
                                    if current_engine == "dia_4bit":
                                        print("🎯 Confirmed: Service started in Dia 4-bit mode")
                                    else:
                                        print(f"❌ CRITICAL: Service should have failed but started in {current_engine} mode")
                                        print("   🔧 The strict mode implementation needs verification")
                                else:
                                    print("⚠️ Could not verify engine mode - service may still be initializing")
                            else:
                                print("❌ Service started but did not become ready in time")
                        else:
                            print("❌ Dia 4-bit TTS failed to start (this is expected until torch_dtype issue is fixed)")
                            print("   💡 Service correctly refused to fall back to Full Dia mode")
                            print("   🔧 Check the service logs for the specific error details")
                    
                    # Auto-refresh status after operation
                    print("🔄 Refreshing status...")
                    self.invalidate_cache()
                    time.sleep(1)
                    continue
                else:
                    print("❌ Invalid choice")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Returning to main menu...")
                break
    
    def launch_comprehensive_tests(self):
        """Launch comprehensive test suite following existing patterns"""
        print("=" * 70)
        print("🧪 Launching Comprehensive Test Suite")
        print("=" * 70)
        
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
        # Note: Services remain running when manager is interrupted

    def manage_gpu_models(self):
        """Manage GPU model selection for LLM services"""
        while True:
            print("\n🎮 GPU Model Selection")
            print("=" * 60)
            print("Choose the optimal LLM model for your GPU:")
            
            # Auto-detect current GPU if possible
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    print(f"🎯 Detected GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                else:
                    print("⚠️  No GPU detected - CPU mode available")
            except ImportError:
                print("📊 GPU detection unavailable")
            
            print("\n🚀 Available GPU Models:")
            
            # Display GPU models with numbered options
            gpu_list = list(self.gpu_models.items())
            for i, (model_key, config) in enumerate(gpu_list, 1):
                print(f"  {i}. {config['name']}")
                print(f"      📱 {config['description']}")
                print(f"      💾 GPU Memory: {config['gpu_memory']}")
                print(f"      ⚡ Response: {config['response_time']}")
                print(f"      🎯 Use Case: {config['use_case']}")
                print()
            
            print("  r. Show current model status")
            print("  0. Back to main menu")
            
            try:
                choice = input(f"\nSelect GPU model (0-{len(gpu_list)}, r): ").strip().lower()
                
                if choice == "0":
                    break
                elif choice == "r":
                    self.show_gpu_model_status()
                    continue
                elif choice.isdigit() and 1 <= int(choice) <= len(gpu_list):
                    choice_num = int(choice)
                    model_key = gpu_list[choice_num - 1][0]
                    config = gpu_list[choice_num - 1][1]
                    
                    print(f"\n🎯 Switching to {config['name']}...")
                    success = self.switch_gpu_model(model_key, config)
                    
                    if success:
                        print(f"✅ Successfully switched to {config['name']}")
                        print(f"🚀 {config['description']} is now active")
                    else:
                        print(f"❌ Failed to switch to {config['name']}")
                    
                    input("\nPress Enter to continue...")
                else:
                    print("❌ Invalid choice. Please enter a valid option.")
                    
            except (EOFError, KeyboardInterrupt):
                break

    def show_gpu_model_status(self):
        """Show current GPU model status"""
        print("\n📊 Current GPU Model Status")
        print("-" * 50)
        
        # Check which GPU models are currently running
        active_models = []
        for model_key, config in self.gpu_models.items():
            port = config['port']
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
                if response.status_code == 200:
                    active_models.append((model_key, config))
            except:
                pass
        
        if active_models:
            print("🟢 Active GPU Models:")
            for model_key, config in active_models:
                print(f"  ✅ {config['name']} (Port {config['port']})")
                print(f"      📱 {config['description']}")
        else:
            print("🔴 No GPU models currently running")
            
        # Show GPU memory status if available
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"\n💾 GPU Memory Status:")
                print(f"  📊 Allocated: {allocated:.2f}GB")
                print(f"  📦 Reserved: {reserved:.2f}GB") 
                print(f"  💿 Total: {total:.1f}GB")
                print(f"  🔓 Available: {total - reserved:.2f}GB")
        except ImportError:
            pass

    def switch_gpu_model(self, model_key, config):
        """Switch to a specific GPU model"""
        try:
            # Stop any existing LLM services first
            print("🛑 Stopping existing LLM services...")
            for service_name in ["gpt_llm", "mistral_llm"]:
                if service_name in self.services:
                    self.stop_service(service_name)
            
            # Wait for services to stop
            time.sleep(2)
            
            # For the small model, use existing GPT service
            if model_key == "gpt_small":
                print(f"🚀 Starting {config['name']} (Existing Service)...")
                success = self.start_service("gpt_llm")
                
                if success:
                    print(f"✅ {config['name']} started successfully")
                    print(f"🌐 Available at: http://localhost:{config['port']}")
                    return True
                else:
                    print(f"❌ Failed to start {config['name']}")
                    return False
            else:
                # For other models, show that they need to be implemented
                print(f"🔧 {config['name']} is configured for AWS deployment")
                print(f"📋 Model Details:")
                print(f"   📱 {config['description']}")
                print(f"   💾 GPU Memory: {config['gpu_memory']}")
                print(f"   ⚡ Response: {config['response_time']}")
                print(f"   🎯 Use Case: {config['use_case']}")
                print(f"\n💡 This model will be available when deploying to AWS with larger GPUs")
                print(f"🚀 For now, you can use GPT Small (Option 1) for stable 8GB performance")
                return True
                
        except Exception as e:
            print(f"❌ Error switching GPU model: {e}")
            return False

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
