#!/usr/bin/env python3
"""
WebSocket Streaming Services Launcher

Launch and manage all streaming services for the WebSocket architecture:
- STT Streaming Service
- LLM Streaming Service 
- TTS Streaming Service
"""

import asyncio
import subprocess
import sys
import time
import logging
import signal
from pathlib import Path
from typing import Dict, List, Optional
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamingServicesLauncher:
    """Launcher for WebSocket streaming services"""
    
    def __init__(self):
        self.processes = {}
        self.service_configs = {
            'stt': {
                'script': 'ws_streaming_stt_service.py',
                'port': 8003,
                'args': ['--model', 'whisper-base']
            },
            'llm': {
                'script': 'ws_streaming_llm_service.py', 
                'port': 8022,
                'args': ['--model', 'gpt-3.5-turbo']
            },
            'tts': {
                'script': 'ws_streaming_tts_service.py',
                'port': 8015,
                'args': ['--engine', 'tortoise']
            }
        }
        
    def check_orchestrator_running(self) -> bool:
        """Check if the orchestrator is running"""
        try:
            response = requests.get("http://localhost:8080/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def start_service(self, service_name: str, wait_time: float = 2.0) -> bool:
        """Start a specific streaming service"""
        if service_name not in self.service_configs:
            logger.error(f"Unknown service: {service_name}")
            return False
        
        config = self.service_configs[service_name]
        script_path = Path(__file__).parent / config['script']
        
        if not script_path.exists():
            logger.error(f"Service script not found: {script_path}")
            return False
        
        try:
            # Build command
            cmd = [
                sys.executable,
                str(script_path),
                '--port', str(config['port'])
            ] + config['args']
            
            logger.info(f"Starting {service_name.upper()} service on port {config['port']}...")
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.processes[service_name] = process
            
            # Wait a bit for the service to start
            await asyncio.sleep(wait_time)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info(f"✅ {service_name.upper()} service started successfully (PID: {process.pid})")
                return True
            else:
                # Process exited, get error output
                stdout, stderr = process.communicate()
                logger.error(f"❌ {service_name.upper()} service failed to start:")
                if stderr:
                    logger.error(f"Error: {stderr}")
                if stdout:
                    logger.info(f"Output: {stdout}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start {service_name} service: {e}")
            return False
    
    async def stop_service(self, service_name: str) -> bool:
        """Stop a specific service"""
        if service_name not in self.processes:
            logger.warning(f"Service {service_name} not found in managed processes")
            return True
        
        process = self.processes[service_name]
        
        try:
            logger.info(f"Stopping {service_name.upper()} service...")
            
            # Try graceful termination first
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
                logger.info(f"✅ {service_name.upper()} service stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if needed
                logger.warning(f"Force-killing {service_name.upper()} service...")
                process.kill()
                process.wait()
                logger.info(f"✅ {service_name.upper()} service force-stopped")
            
            del self.processes[service_name]
            return True
            
        except Exception as e:
            logger.error(f"Error stopping {service_name} service: {e}")
            return False
    
    async def start_all_services(self) -> bool:
        """Start all streaming services"""
        logger.info("🚀 Starting all WebSocket streaming services...")
        
        # Check if orchestrator is running
        if not self.check_orchestrator_running():
            logger.error("❌ WebSocket orchestrator is not running!")
            logger.info("   Please start the orchestrator first:")
            logger.info("   python ws_test_config.py --interactive")
            return False
        
        logger.info("✅ Orchestrator is running")
        
        # Start services in order
        services = ['stt', 'llm', 'tts']
        success_count = 0
        
        for service in services:
            if await self.start_service(service):
                success_count += 1
            else:
                logger.error(f"Failed to start {service} service")
                break
        
        if success_count == len(services):
            logger.info("✅ All streaming services started successfully!")
            logger.info("\n📊 Service Status:")
            self.print_service_status()
            return True
        else:
            logger.error(f"❌ Only {success_count}/{len(services)} services started")
            return False
    
    async def stop_all_services(self):
        """Stop all running services"""
        logger.info("🛑 Stopping all streaming services...")
        
        for service_name in list(self.processes.keys()):
            await self.stop_service(service_name)
        
        logger.info("✅ All services stopped")
    
    def print_service_status(self):
        """Print status of all services"""
        for service_name, config in self.service_configs.items():
            if service_name in self.processes:
                process = self.processes[service_name]
                status = "🟢 Running" if process.poll() is None else "🔴 Stopped"
                pid = process.pid if process.poll() is None else "N/A"
                print(f"  {service_name.upper():3}: {status} (Port: {config['port']}, PID: {pid})")
            else:
                print(f"  {service_name.upper():3}: 🔴 Not Started (Port: {config['port']})")
    
    async def monitor_services(self):
        """Monitor running services and restart if they crash"""
        logger.info("👀 Monitoring services (Ctrl+C to stop)...")
        
        try:
            while True:
                # Check each service
                for service_name in list(self.processes.keys()):
                    process = self.processes[service_name]
                    
                    if process.poll() is not None:
                        # Process has exited
                        logger.warning(f"⚠️  {service_name.upper()} service has crashed!")
                        
                        # Get exit info
                        stdout, stderr = process.communicate()
                        if stderr:
                            logger.error(f"Error output: {stderr}")
                        
                        # Attempt restart
                        logger.info(f"🔄 Attempting to restart {service_name.upper()} service...")
                        del self.processes[service_name]
                        
                        if await self.start_service(service_name):
                            logger.info(f"✅ {service_name.upper()} service restarted successfully")
                        else:
                            logger.error(f"❌ Failed to restart {service_name.upper()} service")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WebSocket Streaming Services Launcher")
    parser.add_argument("command", choices=['start', 'stop', 'status', 'monitor'], 
                       help="Command to execute")
    parser.add_argument("--service", choices=['stt', 'llm', 'tts'], 
                       help="Specific service to control (optional)")
    
    args = parser.parse_args()
    
    launcher = StreamingServicesLauncher()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, stopping services...")
        asyncio.create_task(launcher.stop_all_services())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.command == 'start':
            if args.service:
                success = await launcher.start_service(args.service)
                if not success:
                    return 1
            else:
                success = await launcher.start_all_services()
                if success:
                    # Keep services running
                    await launcher.monitor_services()
                else:
                    return 1
                    
        elif args.command == 'stop':
            if args.service:
                await launcher.stop_service(args.service)
            else:
                await launcher.stop_all_services()
                
        elif args.command == 'status':
            logger.info("📊 Current Service Status:")
            launcher.print_service_status()
            
        elif args.command == 'monitor':
            await launcher.monitor_services()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        await launcher.stop_all_services()
        return 0
    except Exception as e:
        logger.error(f"Launcher error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
