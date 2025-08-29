"""
Local Development Server
Runs all microservices on localhost with different ports for easy development
"""
import asyncio
import subprocess
import sys
import time
import signal
import os
from pathlib import Path
import threading
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class LocalMicroservicesRunner:
    """Runs all microservices locally for development"""
    
    def __init__(self):
        self.processes = []
        self.base_dir = Path(__file__).parent.parent
        
        # Service configurations
        self.services = {
            'stt': {
                'script': 'aws_microservices/stt_service.py',
                'port': 8001,
                'name': 'STT Service'
            },
            'llm': {
                'script': 'aws_microservices/llm_service.py', 
                'port': 8002,
                'name': 'LLM Service'
            },
            'tts': {
                'script': 'aws_microservices/tts_service.py',
                'port': 8003,
                'name': 'TTS Service'
            }
        }
        
        # Track running status
        self.running = False
        self.shutdown_event = threading.Event()
    
    def start_service(self, service_name: str, config: dict):
        """Start a single microservice"""
        script_path = self.base_dir / config['script']
        
        if not script_path.exists():
            logger.error(f"Service script not found: {script_path}")
            return None
        
        logger.info(f"🚀 Starting {config['name']} on port {config['port']}...")
        
        # Set environment variables
        env = os.environ.copy()
        env['SERVICE_PORT'] = str(config['port'])
        env['ENVIRONMENT'] = 'local'
        env['PYTHONPATH'] = str(self.base_dir)
        
        try:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(self.base_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Start thread to handle output
            threading.Thread(
                target=self._handle_service_output,
                args=(process, config['name']),
                daemon=True
            ).start()
            
            return process
            
        except Exception as e:
            logger.error(f"Failed to start {config['name']}: {e}")
            return None
    
    def _handle_service_output(self, process, service_name):
        """Handle service output in a separate thread"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    logger.info(f"[{service_name}] {line.strip()}")
        except Exception as e:
            logger.error(f"Error reading output from {service_name}: {e}")
    
    async def start_all_services(self):
        """Start all microservices"""
        logger.info("🎭 Starting Local Microservices Development Server")
        logger.info("=" * 60)
        
        self.running = True
        
        # Start each service
        for service_name, config in self.services.items():
            process = self.start_service(service_name, config)
            if process:
                self.processes.append((service_name, process))
                # Wait a bit between service starts
                await asyncio.sleep(2)
            else:
                logger.error(f"Failed to start {service_name}")
        
        if not self.processes:
            logger.error("❌ No services started successfully")
            return False
        
        # Wait for services to be ready
        logger.info("⏳ Waiting for services to be ready...")
        await asyncio.sleep(10)
        
        # Check service health
        await self.check_service_health()
        
        # Display service URLs
        self.display_service_info()
        
        return True
    
    async def check_service_health(self):
        """Check health of all running services"""
        logger.info("🏥 Checking service health...")
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                for service_name, config in self.services.items():
                    url = f"http://localhost:{config['port']}/health"
                    try:
                        async with session.get(url, timeout=5) as response:
                            if response.status == 200:
                                data = await response.json()
                                status = "✅ Healthy" if data.get("ready") else "⚠️ Starting"
                                logger.info(f"{status} - {config['name']} ({url})")
                            else:
                                logger.warning(f"❌ Unhealthy - {config['name']} ({response.status})")
                    except Exception as e:
                        logger.warning(f"❌ Cannot reach {config['name']}: {e}")
                        
        except ImportError:
            logger.warning("aiohttp not available - install with: pip install aiohttp")
            logger.info("Manual health check:")
            for service_name, config in self.services.items():
                logger.info(f"  curl http://localhost:{config['port']}/health")
    
    def display_service_info(self):
        """Display information about running services"""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 LOCAL MICROSERVICES READY!")
        logger.info("=" * 60)
        
        for service_name, config in self.services.items():
            logger.info(f"📡 {config['name']}: http://localhost:{config['port']}")
        
        logger.info("\n🔗 API Endpoints:")
        logger.info("  STT:  POST http://localhost:8001/transcribe")
        logger.info("  LLM:  POST http://localhost:8002/generate") 
        logger.info("  TTS:  POST http://localhost:8003/synthesize")
        
        logger.info("\n🏥 Health Checks:")
        for service_name, config in self.services.items():
            logger.info(f"  {config['name']}: GET http://localhost:{config['port']}/health")
        
        logger.info("\n💡 Usage:")
        logger.info("  • Test with: python orchestrator_client.py")
        logger.info("  • Stop with: Ctrl+C")
        logger.info("  • Logs: Check terminal output above")
        
        logger.info("=" * 60)
    
    def stop_all_services(self):
        """Stop all running services"""
        logger.info("🛑 Stopping all microservices...")
        
        self.running = False
        self.shutdown_event.set()
        
        for service_name, process in self.processes:
            try:
                logger.info(f"Stopping {service_name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {service_name}...")
                    process.kill()
                    process.wait()
                    
                logger.info(f"✅ {service_name} stopped")
                
            except Exception as e:
                logger.error(f"Error stopping {service_name}: {e}")
        
        self.processes.clear()
        logger.info("✅ All services stopped")
    
    async def run_forever(self):
        """Run services until interrupted"""
        try:
            if await self.start_all_services():
                logger.info("\n🎯 Services running - Press Ctrl+C to stop")
                
                # Keep running until interrupted
                while self.running and not self.shutdown_event.is_set():
                    await asyncio.sleep(1)
                    
                    # Check if any process has died
                    for service_name, process in self.processes[:]:
                        if process.poll() is not None:
                            logger.error(f"❌ {service_name} has stopped unexpectedly")
                            self.processes.remove((service_name, process))
                    
                    # If all processes died, exit
                    if not self.processes:
                        logger.error("❌ All services have stopped")
                        break
            else:
                logger.error("❌ Failed to start services")
                
        except KeyboardInterrupt:
            logger.info("\n⚠️ Interrupt received")
        finally:
            self.stop_all_services()

def setup_signal_handlers(runner):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"\n⚠️ Received signal {signum}")
        runner.stop_all_services()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point"""
    runner = LocalMicroservicesRunner()
    setup_signal_handlers(runner)
    
    logger.info("🎭 Local Microservices Development Server")
    logger.info("Starting all services on localhost...")
    
    await runner.run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
