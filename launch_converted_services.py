#!/usr/bin/env python3
"""
Converted WebSocket Services Launcher
Launch your existing STT, LLM, and TTS services converted to WebSocket streaming
"""

import asyncio
import subprocess
import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class WebSocketServicesLauncher:
    """Launcher for converted WebSocket streaming services"""
    
    def __init__(self):
        self.services = {}
        self.orchestrator_process = None
        self.running = False
        
        # Service definitions - converted from existing HTTP services
        self.service_configs = {
            "stt_whisper": {
                "name": "WebSocket Whisper STT",
                "script": "aws_microservices/ws_stt_whisper_service.py",
                "description": "Converted STT service with WebSocket streaming",
                "port": None,  # WebSocket services connect to orchestrator
                "depends_on": ["orchestrator"]
            },
            "llm_gpt": {
                "name": "WebSocket GPT LLM", 
                "script": "aws_microservices/ws_llm_gpt_service.py",
                "description": "Converted LLM service with token streaming",
                "port": None,
                "depends_on": ["orchestrator"]
            },
            "tts_tortoise": {
                "name": "WebSocket Tortoise TTS",
                "script": "aws_microservices/ws_tts_tortoise_service.py", 
                "description": "Converted TTS service with audio streaming",
                "port": None,
                "depends_on": ["orchestrator"]
            }
        }
        
        # Orchestrator configuration
        self.orchestrator_config = {
            "name": "WebSocket Orchestrator",
            "script": "ws_orchestrator_service.py",
            "description": "Central streaming hub for service communication",
            "client_port": 9000,
            "service_port": 9001,
            "http_port": 8080
        }
    
    def safe_print(self, text: str):
        """Safe print function that handles Unicode characters for Windows console."""
        try:
            print(text)
        except UnicodeEncodeError:
            safe_text = text.encode('ascii', 'replace').decode('ascii')
            print(safe_text)
    
    async def check_orchestrator_health(self) -> bool:
        """Check if orchestrator is healthy and ready"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8080/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("status") == "healthy"
        except Exception as e:
            logging.debug(f"Orchestrator health check failed: {e}")
        
        return False
    
    async def start_orchestrator(self) -> bool:
        """Start the WebSocket orchestrator"""
        try:
            self.safe_print("🚀 Starting WebSocket Orchestrator...")
            
            script_path = Path(self.orchestrator_config["script"])
            if not script_path.exists():
                self.safe_print(f"❌ Orchestrator script not found: {script_path}")
                return False
            
            # Start orchestrator process
            self.orchestrator_process = subprocess.Popen([
                sys.executable, str(script_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for orchestrator to be ready
            max_wait = 30  # 30 seconds max wait
            wait_time = 0
            
            self.safe_print("⏳ Waiting for orchestrator to be ready...")
            
            while wait_time < max_wait:
                if await self.check_orchestrator_health():
                    self.safe_print("✅ Orchestrator is ready!")
                    self.safe_print(f"   📡 Client WebSocket: ws://localhost:{self.orchestrator_config['client_port']}")
                    self.safe_print(f"   🔧 Service WebSocket: ws://localhost:{self.orchestrator_config['service_port']}")
                    self.safe_print(f"   🌐 HTTP API: http://localhost:{self.orchestrator_config['http_port']}")
                    return True
                
                await asyncio.sleep(1)
                wait_time += 1
                
                if wait_time % 5 == 0:
                    self.safe_print(f"   ⏱️  Still waiting... ({wait_time}s/{max_wait}s)")
            
            self.safe_print(f"❌ Orchestrator failed to start within {max_wait} seconds")
            return False
            
        except Exception as e:
            self.safe_print(f"❌ Error starting orchestrator: {e}")
            return False
    
    async def start_service(self, service_id: str) -> bool:
        """Start a WebSocket service"""
        config = self.service_configs.get(service_id)
        if not config:
            self.safe_print(f"❌ Unknown service: {service_id}")
            return False
        
        try:
            self.safe_print(f"🚀 Starting {config['name']}...")
            
            script_path = Path(config["script"])
            if not script_path.exists():
                self.safe_print(f"❌ Service script not found: {script_path}")
                return False
            
            # Start service process
            process = subprocess.Popen([
                sys.executable, str(script_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.services[service_id] = {
                "process": process,
                "config": config,
                "start_time": time.time(),
                "status": "starting"
            }
            
            # Give service time to start
            await asyncio.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                self.services[service_id]["status"] = "running"
                self.safe_print(f"✅ {config['name']} started successfully")
                return True
            else:
                self.safe_print(f"❌ {config['name']} failed to start")
                self.services[service_id]["status"] = "failed"
                return False
                
        except Exception as e:
            self.safe_print(f"❌ Error starting {config['name']}: {e}")
            return False
    
    async def stop_service(self, service_id: str):
        """Stop a WebSocket service"""
        if service_id not in self.services:
            return
        
        service = self.services[service_id]
        config = service["config"]
        process = service["process"]
        
        try:
            self.safe_print(f"🛑 Stopping {config['name']}...")
            
            # Terminate the process gracefully
            if process.poll() is None:
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    self.safe_print(f"✅ {config['name']} stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if necessary
                    process.kill()
                    process.wait()
                    self.safe_print(f"⚠️  {config['name']} force stopped")
            
            service["status"] = "stopped"
            
        except Exception as e:
            self.safe_print(f"❌ Error stopping {config['name']}: {e}")
    
    async def stop_orchestrator(self):
        """Stop the WebSocket orchestrator"""
        if self.orchestrator_process:
            try:
                self.safe_print("🛑 Stopping WebSocket Orchestrator...")
                
                if self.orchestrator_process.poll() is None:
                    self.orchestrator_process.terminate()
                    
                    try:
                        self.orchestrator_process.wait(timeout=10)
                        self.safe_print("✅ Orchestrator stopped gracefully")
                    except subprocess.TimeoutExpired:
                        self.orchestrator_process.kill()
                        self.orchestrator_process.wait()
                        self.safe_print("⚠️  Orchestrator force stopped")
                
                self.orchestrator_process = None
                
            except Exception as e:
                self.safe_print(f"❌ Error stopping orchestrator: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {
            "orchestrator": {
                "running": self.orchestrator_process and self.orchestrator_process.poll() is None,
                "config": self.orchestrator_config
            },
            "services": {}
        }
        
        for service_id, service_data in self.services.items():
            process = service_data["process"]
            config = service_data["config"]
            
            status["services"][service_id] = {
                "name": config["name"],
                "running": process.poll() is None,
                "status": service_data["status"],
                "start_time": service_data["start_time"],
                "uptime": time.time() - service_data["start_time"] if process.poll() is None else 0
            }
        
        return status
    
    def print_status(self):
        """Print current status of all services"""
        status = self.get_service_status()
        
        self.safe_print("\n" + "="*60)
        self.safe_print("🔄 WebSocket Services Status")
        self.safe_print("="*60)
        
        # Orchestrator status
        orchestrator = status["orchestrator"]
        orchestrator_status = "🟢 RUNNING" if orchestrator["running"] else "🔴 STOPPED"
        self.safe_print(f"📡 Orchestrator: {orchestrator_status}")
        if orchestrator["running"]:
            config = orchestrator["config"]
            self.safe_print(f"   Client WS: ws://localhost:{config['client_port']}")
            self.safe_print(f"   Service WS: ws://localhost:{config['service_port']}")
            self.safe_print(f"   HTTP API: http://localhost:{config['http_port']}")
        
        self.safe_print()
        
        # Services status
        if status["services"]:
            self.safe_print("🔧 Converted Services:")
            for service_id, service_info in status["services"].items():
                service_status = "🟢 RUNNING" if service_info["running"] else "🔴 STOPPED"
                uptime_str = ""
                if service_info["running"] and service_info["uptime"] > 0:
                    uptime = service_info["uptime"]
                    if uptime >= 3600:
                        uptime_str = f" ({uptime/3600:.1f}h)"
                    elif uptime >= 60:
                        uptime_str = f" ({uptime/60:.1f}m)"
                    else:
                        uptime_str = f" ({uptime:.0f}s)"
                
                self.safe_print(f"   {service_info['name']}: {service_status}{uptime_str}")
        else:
            self.safe_print("🔧 No services started yet")
        
        self.safe_print("="*60)
    
    async def start_all_services(self) -> bool:
        """Start orchestrator and all WebSocket services"""
        self.safe_print("🚀 Starting WebSocket Streaming Architecture...")
        self.safe_print("📋 Converting existing HTTP services to WebSocket streaming")
        self.safe_print("-" * 60)
        
        # Start orchestrator first
        if not await self.start_orchestrator():
            self.safe_print("❌ Failed to start orchestrator - aborting")
            return False
        
        self.safe_print()
        
        # Start all services
        success_count = 0
        for service_id in self.service_configs.keys():
            if await self.start_service(service_id):
                success_count += 1
            await asyncio.sleep(1)  # Stagger service starts
        
        self.safe_print()
        self.safe_print(f"📊 Started {success_count}/{len(self.service_configs)} services")
        
        if success_count == len(self.service_configs):
            self.safe_print("✅ All WebSocket services started successfully!")
            self.safe_print()
            self.safe_print("🎯 Your services are now streaming-enabled:")
            self.safe_print("   • STT: Real-time audio chunk processing")
            self.safe_print("   • LLM: Token-by-token response streaming") 
            self.safe_print("   • TTS: Audio chunk streaming with 29 voices")
            self.safe_print()
            self.safe_print("🔗 Use the Phase 2 demo to test the streaming pipeline:")
            self.safe_print("   python phase2_streaming_demo.py")
            return True
        else:
            self.safe_print("⚠️  Some services failed to start")
            return False
    
    async def stop_all_services(self):
        """Stop all services and orchestrator"""
        self.safe_print("🛑 Stopping all WebSocket services...")
        
        # Stop services first
        for service_id in list(self.services.keys()):
            await self.stop_service(service_id)
        
        # Stop orchestrator last
        await self.stop_orchestrator()
        
        self.safe_print("✅ All services stopped")
    
    async def monitor_services(self):
        """Monitor services and restart if needed"""
        self.safe_print("👀 Starting service monitoring (Ctrl+C to stop)...")
        self.running = True
        
        try:
            while self.running:
                # Check orchestrator
                if self.orchestrator_process and self.orchestrator_process.poll() is not None:
                    self.safe_print("❌ Orchestrator died, restarting...")
                    await self.start_orchestrator()
                
                # Check services
                for service_id, service_data in list(self.services.items()):
                    process = service_data["process"]
                    if process.poll() is not None:
                        config = service_data["config"]
                        self.safe_print(f"❌ {config['name']} died, restarting...")
                        await self.start_service(service_id)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            self.safe_print("\n📋 Received interrupt, stopping monitoring...")
            self.running = False
    
    async def interactive_menu(self):
        """Interactive menu for service management"""
        while True:
            self.safe_print("\n" + "="*50)
            self.safe_print("🎛️  WebSocket Services Launcher - Converted Services")
            self.safe_print("="*50)
            self.safe_print("1. Start All Services")
            self.safe_print("2. Stop All Services") 
            self.safe_print("3. Show Service Status")
            self.safe_print("4. Start Individual Service")
            self.safe_print("5. Stop Individual Service")
            self.safe_print("6. Monitor Services (with auto-restart)")
            self.safe_print("7. Test Streaming Pipeline")
            self.safe_print("8. Exit")
            self.safe_print("-" * 50)
            
            try:
                choice = input("Enter your choice (1-8): ").strip()
                
                if choice == "1":
                    await self.start_all_services()
                
                elif choice == "2":
                    await self.stop_all_services()
                
                elif choice == "3":
                    self.print_status()
                
                elif choice == "4":
                    self.safe_print("\nAvailable services:")
                    for i, (service_id, config) in enumerate(self.service_configs.items(), 1):
                        self.safe_print(f"{i}. {config['name']} ({service_id})")
                    
                    try:
                        svc_choice = int(input("Select service (1-3): ").strip())
                        if 1 <= svc_choice <= len(self.service_configs):
                            service_id = list(self.service_configs.keys())[svc_choice - 1]
                            
                            # Ensure orchestrator is running first
                            if not await self.check_orchestrator_health():
                                self.safe_print("⚠️  Orchestrator not running, starting it first...")
                                await self.start_orchestrator()
                            
                            await self.start_service(service_id)
                        else:
                            self.safe_print("❌ Invalid service selection")
                    except ValueError:
                        self.safe_print("❌ Invalid input")
                
                elif choice == "5":
                    if not self.services:
                        self.safe_print("No services are running")
                        continue
                    
                    self.safe_print("\nRunning services:")
                    running_services = [(sid, sdata) for sid, sdata in self.services.items() 
                                      if sdata["process"].poll() is None]
                    
                    if not running_services:
                        self.safe_print("No services are currently running")
                        continue
                    
                    for i, (service_id, service_data) in enumerate(running_services, 1):
                        config = service_data["config"]
                        self.safe_print(f"{i}. {config['name']} ({service_id})")
                    
                    try:
                        svc_choice = int(input(f"Select service to stop (1-{len(running_services)}): ").strip())
                        if 1 <= svc_choice <= len(running_services):
                            service_id = running_services[svc_choice - 1][0]
                            await self.stop_service(service_id)
                        else:
                            self.safe_print("❌ Invalid service selection")
                    except ValueError:
                        self.safe_print("❌ Invalid input")
                
                elif choice == "6":
                    # Start all services first if not already running
                    status = self.get_service_status()
                    if not status["orchestrator"]["running"]:
                        self.safe_print("Starting services for monitoring...")
                        await self.start_all_services()
                    
                    await self.monitor_services()
                
                elif choice == "7":
                    self.safe_print("\n🧪 Testing Streaming Pipeline...")
                    
                    # Check if services are running
                    if not await self.check_orchestrator_health():
                        self.safe_print("⚠️  Services not running, starting them first...")
                        await self.start_all_services()
                        await asyncio.sleep(3)  # Give services time to fully start
                    
                    # Run the Phase 2 demo
                    try:
                        demo_path = Path("phase2_streaming_demo.py")
                        if demo_path.exists():
                            self.safe_print("🚀 Running streaming pipeline test...")
                            process = subprocess.run([sys.executable, str(demo_path)], 
                                                   capture_output=True, text=True)
                            
                            if process.returncode == 0:
                                self.safe_print("✅ Streaming pipeline test completed successfully!")
                                if process.stdout:
                                    self.safe_print("📋 Test output:")
                                    self.safe_print(process.stdout)
                            else:
                                self.safe_print("❌ Streaming pipeline test failed")
                                if process.stderr:
                                    self.safe_print("📋 Error output:")
                                    self.safe_print(process.stderr)
                        else:
                            self.safe_print("❌ Phase 2 demo script not found")
                            self.safe_print("💡 Run: python phase2_streaming_demo.py")
                    except Exception as e:
                        self.safe_print(f"❌ Error running test: {e}")
                
                elif choice == "8":
                    self.safe_print("👋 Stopping all services and exiting...")
                    await self.stop_all_services()
                    break
                
                else:
                    self.safe_print("❌ Invalid choice, please select 1-8")
                    
            except KeyboardInterrupt:
                self.safe_print("\n👋 Interrupted, stopping all services...")
                await self.stop_all_services()
                break
            except Exception as e:
                self.safe_print(f"❌ Error: {e}")

def setup_signal_handlers(launcher):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print(f"\n[SHUTDOWN] Received signal {signum}")
        # Create new event loop for cleanup since we might not be in async context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule cleanup for later
                asyncio.create_task(launcher.stop_all_services())
            else:
                # Run cleanup directly
                asyncio.run(launcher.stop_all_services())
        except:
            # Fallback cleanup
            pass
        
        sys.exit(0)
    
    # Handle common termination signals
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, signal_handler)

def safe_print(text):
    """Safe print function that handles Unicode characters for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)

async def main():
    """Main entry point"""
    safe_print("🔄 WebSocket Services Launcher - Converted Services")
    safe_print("📋 Launch your existing STT/LLM/TTS services with WebSocket streaming")
    safe_print("=" * 70)
    
    launcher = WebSocketServicesLauncher()
    
    # Setup signal handlers
    setup_signal_handlers(launcher)
    
    # Run interactive menu
    await launcher.interactive_menu()
    
    safe_print("✅ WebSocket Services Launcher completed")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the launcher
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        safe_print("\n👋 Launcher interrupted")
    except Exception as e:
        safe_print(f"❌ Launcher error: {e}")
        sys.exit(1)
