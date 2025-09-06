#!/usr/bin/env python3
"""
WebSocket Test Configuration and Launcher

Quick test launcher for the WebSocket streaming architecture.
"""

import asyncio
import subprocess
import sys
import time
import json
import requests
from pathlib import Path

class WebSocketTester:
    """Test configuration for WebSocket services"""
    
    def __init__(self):
        self.orchestrator_process = None
        self.test_services = []
        self.cleanup_requested = False  # Track if user requested cleanup
        
    def check_orchestrator_ports(self):
        """Check if orchestrator ports are in use"""
        import socket
        
        orchestrator_ports = [9000, 9001, 8080]  # Client, Service, HTTP
        ports_in_use = []
        
        for port in orchestrator_ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', port))
                if result == 0:  # Port is in use
                    ports_in_use.append(port)
        
        return ports_in_use
    
    def check_ports(self):
        """Check if required ports are available"""
        import socket
        
        ports_to_check = [9000, 9001, 8080]  # Client, Service, HTTP
        available_ports = []
        
        for port in ports_to_check:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', port))
                if result == 0:
                    print(f"❌ Port {port} is already in use")
                else:
                    print(f"✅ Port {port} is available")
                    available_ports.append(port)
        
        return len(available_ports) == len(ports_to_check)
    
    def start_orchestrator(self):
        """Start the WebSocket orchestrator"""
        print("\n🚀 Starting WebSocket Orchestrator...")
        
        # First check if orchestrator is already running
        if self.check_orchestrator_health(quiet=True):
            print("\n✅ Orchestrator is already running and healthy")
            return True
        
        # Check if ports are in use (might be starting up or unhealthy)
        ports_in_use = self.check_orchestrator_ports()
        if ports_in_use:
            print(f"\n⚠️  Orchestrator ports {ports_in_use} are in use but health check failed")
            print("   The orchestrator might be starting up or unhealthy")
            
            # Wait a bit and try health check again
            time.sleep(2)
            if self.check_orchestrator_health(quiet=True):
                print("\n✅ Orchestrator is now healthy")
                return True
            else:
                print("\n❌ Orchestrator ports are occupied but service is not responding")
                return False
        
        try:
            # Start orchestrator in background
            self.orchestrator_process = subprocess.Popen([
                sys.executable, 
                "ws_orchestrator_service.py"
            ], cwd=Path.cwd())
            
            # Wait for startup
            time.sleep(3)
            
            # Check if it's running
            if self.orchestrator_process.poll() is None:
                # Double-check with health endpoint
                if self.check_orchestrator_health(quiet=True):
                    print("\n✅ Orchestrator started successfully")
                    return True
                else:
                    print("\n❌ Orchestrator process started but health check failed")
                    return False
            else:
                print("\n❌ Orchestrator failed to start (process exited)")
                return False
                
        except Exception as e:
            print(f"\n❌ Error starting orchestrator: {e}")
            return False
    
    def format_uptime(self, seconds):
        """Format uptime in a human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        
        if minutes < 60:
            if remaining_seconds < 1:
                return f"{minutes}m"
            else:
                return f"{minutes}m {remaining_seconds:.0f}s"
        
        hours = minutes // 60
        remaining_minutes = minutes % 60
        
        if hours < 24:
            if remaining_minutes == 0:
                return f"{hours}h"
            else:
                return f"{hours}h {remaining_minutes}m"
        
        days = hours // 24
        remaining_hours = hours % 24
        
        if remaining_hours == 0:
            return f"{days}d"
        else:
            return f"{days}d {remaining_hours}h"
    
    def check_orchestrator_health(self, quiet=False):
        """Check orchestrator health endpoint"""
        try:
            response = requests.get("http://localhost:8080/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if not quiet:
                    print("\n✅ Orchestrator health check passed")
                    print(f"   Status: {data.get('status')}")
                    uptime_formatted = self.format_uptime(data.get('uptime', 0))
                    print(f"   Uptime: {uptime_formatted}")
                return True
            else:
                if not quiet:
                    print(f"\n❌ Orchestrator health check failed: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            if not quiet:
                print("\n❌ Orchestrator is not running")
                print("   Use option 1 to start the orchestrator")
            return False
        except requests.exceptions.Timeout:
            if not quiet:
                print("\n❌ Orchestrator health check timed out")
                print("   The orchestrator might be unresponsive")
            return False
        except Exception as e:
            if not quiet:
                print(f"\n❌ Orchestrator health check error: {type(e).__name__}")
                print(f"   The orchestrator might not be running properly")
            return False
    
    def create_mock_service_registration(self, service_type: str, port: int):
        """Create a mock service registration for testing"""
        from ws_service_registry import ServiceRegistration, ServiceCapabilities
        
        return ServiceRegistration(
            service_id=f"mock_{service_type}_001",
            service_type=service_type,
            service_name=f"Mock {service_type.upper()} Service",
            version="1.0.0",
            endpoint=f"ws://localhost:{port}",
            websocket_port=port,
            http_port=port + 100,
            capabilities=ServiceCapabilities(
                realtime=True,
                streaming=True,
                languages=["en"],
                max_concurrent=5,
                latency_ms=100
            ),
            metadata={
                "test_mode": True,
                "description": f"Mock {service_type} service for testing"
            }
        )
    
    async def register_mock_services(self):
        """Register mock services for testing"""
        print("\n📋 Registering mock services...")
        
        # Mock services configuration
        mock_services = [
            ("stt", 7001),
            ("llm", 7002), 
            ("tts", 7003)
        ]
        
        for service_type, port in mock_services:
            registration = self.create_mock_service_registration(service_type, port)
            
            try:
                # Send registration to orchestrator
                response = requests.post(
                    "http://localhost:8080/register_service",
                    json=registration.to_dict(),
                    timeout=5
                )
                
                if response.status_code == 200:
                    print(f"✅ Registered mock {service_type.upper()} service")
                else:
                    print(f"❌ Failed to register {service_type} service: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error registering {service_type} service: {e}")
    
    async def test_client_connection(self):
        """Test client connection to orchestrator"""
        print("\n🔌 Testing client connection...")
        
        try:
            # Import and run headset client test
            from ws_headset_client import HeadsetClient
            
            client = HeadsetClient()
            
            # Quick connection test
            await client.connect()
            print("✅ Client connected successfully")
            
            # Send test message
            await client.send_text_input("Hello WebSocket world!")
            print("✅ Test message sent")
            
            # Wait for potential responses
            await asyncio.sleep(2)
            
            await client.disconnect()
            print("✅ Client disconnected cleanly")
            
            return True
            
        except Exception as e:
            print(f"❌ Client connection test failed: {e}")
            return False
    
    def show_service_status(self):
        """Show current service status"""
        print("\n📊 Service Status:")
        print("-" * 40)
        
        # First check orchestrator status
        orchestrator_healthy = self.check_orchestrator_health(quiet=True)
        orchestrator_status = "🟢 Running" if orchestrator_healthy else "🔴 Not Running"
        print(f"  ORCHESTRATOR: {orchestrator_status}")
        
        if not orchestrator_healthy:
            print("  └─ WebSocket Orchestrator is not responding")
            print("     Use option 1 to start the orchestrator")
            return
        
        # Show orchestrator details
        try:
            response = requests.get("http://localhost:8080/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                uptime_formatted = self.format_uptime(data.get('uptime', 0))
                print(f"  └─ Uptime: {uptime_formatted}")
                print(f"  └─ Active Sessions: {data.get('active_sessions', 0)}")
        except:
            pass
        
        print()  # Empty line before services
        
        # Show registered services
        try:
            response = requests.get("http://localhost:8080/services", timeout=5)
            if response.status_code == 200:
                services = response.json()
                
                if services:
                    print("  Registered Services:")
                    service_types = {}
                    for service in services:
                        service_id = service.get('service_id', 'unknown')
                        service_type = service.get('service_type', 'unknown').upper()
                        healthy = service.get('healthy', False)
                        status = "🟢 Online" if healthy else "🔴 Offline"
                        
                        if service_type not in service_types:
                            service_types[service_type] = []
                        service_types[service_type].append(f"{service_id} {status}")
                    
                    for service_type, service_list in service_types.items():
                        print(f"  {service_type}:")
                        for service_info in service_list:
                            print(f"    └─ {service_info}")
                else:
                    print("  No services registered")
                    print("  Use option 3 to register mock services for testing")
            else:
                print(f"  ❌ Failed to get service status: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error getting service status: {e}")
        
        # Show port status
        print()  # Empty line
        ports_in_use = self.check_orchestrator_ports()
        if ports_in_use:
            print("  Active Ports:")
            port_names = {9000: "Client WebSocket", 9001: "Service WebSocket", 8080: "HTTP API"}
            for port in ports_in_use:
                port_name = port_names.get(port, f"Port {port}")
                print(f"    └─ {port_name} ({port}) 🟢 Active")
        else:
            print("  No orchestrator ports active")
    
    def cleanup(self):
        """Clean up test processes"""
        print("\n🧹 Cleaning up...")
        
        # Check if we have an orchestrator process to clean up
        if self.orchestrator_process and self.orchestrator_process.poll() is None:
            print("  └─ Stopping orchestrator process...")
            self.orchestrator_process.terminate()
            try:
                self.orchestrator_process.wait(timeout=5)
                print("  └─ ✅ Orchestrator stopped successfully")
            except subprocess.TimeoutExpired:
                print("  └─ ⚠️  Orchestrator didn't stop gracefully, forcing termination...")
                self.orchestrator_process.kill()
                print("  └─ ✅ Orchestrator force-stopped")
        else:
            # Check if orchestrator is still running (started by another instance)
            if self.check_orchestrator_health(quiet=True):
                print("  └─ ⚠️  Orchestrator is still running (started independently)")
                print("      Use 'taskkill /f /im python.exe' to stop all Python processes")
                print("      Or manually stop the orchestrator process")
            else:
                print("  └─ ✅ No orchestrator processes to clean up")
    
    async def run_full_test(self):
        """Run full test suite"""
        print("🧪 WebSocket Streaming Architecture Test")
        print("=" * 50)
        
        try:
            # Step 1: Check ports
            print("\n1️⃣ Checking port availability...")
            if not self.check_ports():
                print("❌ Required ports are not available")
                return False
            
            # Step 2: Start orchestrator
            print("\n2️⃣ Starting orchestrator...")
            if not self.start_orchestrator():
                print("❌ Failed to start orchestrator")
                return False
            
            # Wait for startup
            await asyncio.sleep(3)
            
            # Step 3: Health check
            print("\n3️⃣ Health check...")
            if not self.check_orchestrator_health():
                print("❌ Orchestrator health check failed")
                return False
            
            # Step 4: Register mock services
            print("\n4️⃣ Registering mock services...")
            await self.register_mock_services()
            
            # Step 5: Show status
            print("\n5️⃣ Service status...")
            self.show_service_status()
            
            # Step 6: Test client connection
            print("\n6️⃣ Testing client connection...")
            await self.test_client_connection()
            
            print("\n✅ All tests completed successfully!")
            print("\n📝 Next steps:")
            print("   1. The orchestrator is running on:")
            print("      - Client WebSocket: ws://localhost:9000/client")  
            print("      - Service WebSocket: ws://localhost:9001/service")
            print("      - HTTP API: http://localhost:8080")
            print("   2. Run 'python ws_headset_client.py' to test interactive client")
            print("   3. Implement real STT/LLM/TTS service clients")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return False
    
    async def run_interactive_test(self):
        """Run interactive test mode"""
        print("\n========================================")
        print("🎮 Interactive Test Mode")
        print("========================================")
        self.show_test_menu()
        
        while True:
            try:
                command = input("\n📱 Select option (0-6): ").strip()
                
                if command == '0':
                    print("👋 Exiting interactive test mode...")
                    
                    # Check if orchestrator is running and inform user
                    if self.check_orchestrator_health(quiet=True):
                        print("✅ Leaving orchestrator running")
                        self.cleanup_requested = False
                    break
                elif command == '1':
                    print("\n🚀 Starting orchestrator...")
                    self.start_orchestrator()
                elif command == '2':
                    print("\n🏥 Checking orchestrator health...")
                    self.check_orchestrator_health()
                elif command == '3':
                    print("\n📝 Registering mock services...")
                    await self.register_mock_services()
                elif command == '4':
                    print("\n📊 Current service status...")
                    self.show_service_status()
                elif command == '5':
                    print("\n🔌 Testing client connection...")
                    await self.test_client_connection()
                elif command == '6':
                    print("\n🧹 Cleaning up services...")
                    self.cleanup()
                    self.cleanup_requested = True
                else:
                    print(f"❌ Invalid option '{command}'. Please choose 0-6.")
                
                # Always show menu after each action (except exit)
                if command != '0':
                    self.show_test_menu()
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                self.show_test_menu()
    
    def show_test_menu(self):
        """Display the test menu"""
        print("\n📋 Test Options:")
        print("-----------------------------------")
        print("  1. Start orchestrator")
        print("  2. Check health")
        print("  3. Register mock services")
        print("  4. Show service status")
        print("  5. Test client connection")
        print("  6. Cleanup/stop services")
        print("  0. Exit")
        print("-----------------------------------")

async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WebSocket Test Configuration")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Run quick connection test only")
    
    args = parser.parse_args()
    
    tester = WebSocketTester()
    
    try:
        if args.interactive:
            await tester.run_interactive_test()
        elif args.quick:
            await tester.test_client_connection()
        else:
            await tester.run_full_test()
            
            # Keep running for testing
            print("\n⏸️  Press Ctrl+C to stop...")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                pass
                
    finally:
        # Only cleanup if we started something ourselves AND user didn't explicitly choose to keep it running
        if (hasattr(tester, 'orchestrator_process') and tester.orchestrator_process and 
            not getattr(tester, 'cleanup_requested', None) == False):
            # If interactive mode and user chose to keep running, don't cleanup
            if hasattr(tester, 'cleanup_requested') and not tester.cleanup_requested:
                print("\n👋 Test completed! (Orchestrator left running as requested)")
            else:
                tester.cleanup()
                print("\n👋 Test completed!")
        else:
            print("\n👋 Test completed!")

if __name__ == "__main__":
    asyncio.run(main())
