#!/usr/bin/env python3
"""
🔧 PORT CONFLICT RESOLVER & ZONOS TTS MANAGER
============================================
Ensures Zonos TTS runs without port conflicts by:
1. Checking and cleaning up existing services on port 8014
2. Providing alternative port options
3. Managing service startup/shutdown properly
"""

import subprocess
import sys
import time
import requests
import psutil
import socket
from typing import List, Optional, Dict

class PortManager:
    """Manages ports and resolves conflicts for TTS services"""
    
    def __init__(self):
        self.tts_ports = {
            "kokoro_tts": 8011,
            "hira_dia_tts": 8012, 
            "dia_4bit_tts": 8013,
            "zonos_tts": 8014
        }
        
        self.alternative_ports = {
            "zonos_tts": [8014, 8015, 8016, 8017, 8018]
        }
    
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is currently in use"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                return result == 0
        except Exception:
            return False
    
    def find_processes_using_port(self, port: int) -> List[Dict]:
        """Find all processes using a specific port"""
        processes = []
        
        try:
            # Use netstat to find processes
            result = subprocess.run(
                ['netstat', '-ano'], 
                capture_output=True, 
                text=True, 
                shell=True
            )
            
            lines = result.stdout.split('\n')
            for line in lines:
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            pid = int(parts[-1])
                            try:
                                proc = psutil.Process(pid)
                                processes.append({
                                    'pid': pid,
                                    'name': proc.name(),
                                    'cmdline': ' '.join(proc.cmdline()),
                                    'create_time': proc.create_time()
                                })
                            except psutil.NoSuchProcess:
                                processes.append({
                                    'pid': pid,
                                    'name': 'Unknown',
                                    'cmdline': 'Process not accessible',
                                    'create_time': 0
                                })
                        except (ValueError, IndexError):
                            continue
                            
        except Exception as e:
            print(f"⚠️ Error finding processes: {e}")
            
        return processes
    
    def terminate_process_on_port(self, port: int, force: bool = False) -> bool:
        """Terminate processes using a specific port"""
        processes = self.find_processes_using_port(port)
        
        if not processes:
            print(f"✅ No processes found using port {port}")
            return True
        
        print(f"🔍 Found {len(processes)} process(es) using port {port}:")
        
        for proc in processes:
            print(f"   PID {proc['pid']}: {proc['name']} - {proc['cmdline'][:100]}")
            
            if force or self._should_terminate_process(proc):
                try:
                    psutil.Process(proc['pid']).terminate()
                    print(f"✅ Terminated process {proc['pid']}")
                    
                    # Wait for graceful shutdown
                    time.sleep(2)
                    
                    # Force kill if still running
                    try:
                        if psutil.Process(proc['pid']).is_running():
                            psutil.Process(proc['pid']).kill()
                            print(f"💀 Force killed process {proc['pid']}")
                    except psutil.NoSuchProcess:
                        pass
                        
                except Exception as e:
                    print(f"❌ Failed to terminate process {proc['pid']}: {e}")
                    return False
        
        # Verify port is now free
        time.sleep(1)
        return not self.is_port_in_use(port)
    
    def _should_terminate_process(self, proc: Dict) -> bool:
        """Determine if a process should be terminated"""
        cmdline = proc['cmdline'].lower()
        
        # Terminate if it's a TTS service we recognize
        tts_indicators = [
            'tts_zonos_service.py',
            'zonos_tts',
            'tts_service',
            'fastapi',
            'uvicorn'
        ]
        
        return any(indicator in cmdline for indicator in tts_indicators)
    
    def find_available_port(self, service: str) -> int:
        """Find the first available port for a service"""
        ports_to_try = self.alternative_ports.get(service, [self.tts_ports.get(service, 8014)])
        
        for port in ports_to_try:
            if not self.is_port_in_use(port):
                print(f"✅ Port {port} is available for {service}")
                return port
        
        # If all predefined ports are taken, find any available port
        for port in range(8019, 8030):
            if not self.is_port_in_use(port):
                print(f"✅ Using alternative port {port} for {service}")
                return port
        
        raise Exception("No available ports found!")
    
    def check_service_health(self, port: int) -> bool:
        """Check if a service on a port is healthy"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=3)
            return response.status_code == 200
        except:
            try:
                response = requests.get(f"http://localhost:{port}/", timeout=3)
                return response.status_code in [200, 404]  # 404 is ok, means server is responding
            except:
                return False
    
    def cleanup_zonos_conflicts(self) -> int:
        """Clean up any conflicts for Zonos TTS and return available port"""
        print("🧹 CLEANING UP ZONOS TTS PORT CONFLICTS")
        print("=" * 50)
        
        preferred_port = self.tts_ports["zonos_tts"]
        
        # Check if preferred port is in use
        if self.is_port_in_use(preferred_port):
            print(f"⚠️ Port {preferred_port} is in use")
            
            # Check if it's a healthy Zonos service
            if self.check_service_health(preferred_port):
                print(f"✅ Found healthy service on port {preferred_port}")
                
                # Test if it's actually Zonos TTS
                try:
                    response = requests.get(f"http://localhost:{preferred_port}/", timeout=3)
                    if "zonos" in response.text.lower() or "tts" in response.text.lower():
                        print(f"✅ Confirmed Zonos TTS already running on port {preferred_port}")
                        return preferred_port
                except:
                    pass
            
            # If not a healthy Zonos service, clean it up
            print(f"🔄 Cleaning up processes on port {preferred_port}...")
            if self.terminate_process_on_port(preferred_port):
                print(f"✅ Port {preferred_port} is now free")
                return preferred_port
            else:
                print(f"❌ Could not free port {preferred_port}, finding alternative...")
        else:
            print(f"✅ Port {preferred_port} is available")
            return preferred_port
        
        # Find alternative port
        return self.find_available_port("zonos_tts")

def start_zonos_tts_service(port: int = None) -> bool:
    """Start Zonos TTS service on specified or available port"""
    
    port_manager = PortManager()
    
    # Clean up conflicts and get available port
    if port is None:
        port = port_manager.cleanup_zonos_conflicts()
    
    print(f"\n🚀 STARTING ZONOS TTS SERVICE ON PORT {port}")
    print("=" * 50)
    
    # Update the service file if port is different from default
    if port != 8014:
        update_tts_service_port(port)
    
    # Start the service
    try:
        import subprocess
        import os
        
        service_path = os.path.join(os.getcwd(), "aws_microservices", "tts_zonos_service.py")
        
        print(f"📂 Starting service: {service_path}")
        print(f"🌐 Port: {port}")
        
        # Start service in background
        process = subprocess.Popen([
            sys.executable, service_path
        ], cwd=os.getcwd())
        
        # Wait and verify startup
        print("⏳ Waiting for service to start...")
        for i in range(10):
            time.sleep(1)
            if port_manager.check_service_health(port):
                print(f"✅ Zonos TTS service started successfully on port {port}")
                return True
            print(f"   Attempt {i+1}/10...")
        
        print(f"❌ Service did not start properly on port {port}")
        return False
        
    except Exception as e:
        print(f"❌ Error starting service: {e}")
        return False

def update_tts_service_port(new_port: int):
    """Update the TTS service file to use a different port"""
    service_file = "aws_microservices/tts_zonos_service.py"
    
    try:
        with open(service_file, 'r') as f:
            content = f.read()
        
        # Replace port references
        content = content.replace('port=8014', f'port={new_port}')
        content = content.replace('"port": 8014', f'"port": {new_port}')
        content = content.replace('Port: 8014', f'Port: {new_port}')
        
        with open(service_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated service to use port {new_port}")
        
    except Exception as e:
        print(f"⚠️ Could not update service port: {e}")

def main():
    """Main function to resolve port conflicts and start Zonos TTS"""
    
    print("🎙️ ZONOS TTS PORT CONFLICT RESOLVER")
    print("=" * 60)
    
    port_manager = PortManager()
    
    # Show current port status
    print("\n📊 CURRENT PORT STATUS:")
    for service, port in port_manager.tts_ports.items():
        in_use = port_manager.is_port_in_use(port)
        status = "🔴 IN USE" if in_use else "🟢 AVAILABLE"
        print(f"   {service}: {port} - {status}")
        
        if in_use:
            processes = port_manager.find_processes_using_port(port)
            for proc in processes:
                print(f"      PID {proc['pid']}: {proc['name']}")
    
    # Clean up and start Zonos TTS
    success = start_zonos_tts_service()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("   Zonos TTS is now running without port conflicts")
        print("   You can now test your TTS without issues")
    else:
        print("\n❌ FAILED!")
        print("   Could not start Zonos TTS service")
        print("   Check the error messages above")

if __name__ == "__main__":
    main()
