#!/usr/bin/env python3
"""
Fix Launcher Status Display
==========================
Fix the launcher to properly detect running services after restart
"""

import re
from pathlib import Path

def fix_status_detection():
    """Fix the service status detection logic"""
    
    launcher_file = Path("comprehensive_ws_launcher.py")
    if not launcher_file.exists():
        print("❌ comprehensive_ws_launcher.py not found")
        return False
    
    print("🔧 Fixing service status detection logic...")
    
    with open(launcher_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add a new method to check if a service is actually running
    new_method = '''
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
        
        # Check if registered in orchestrator
        orchestrator_registered = False
        try:
            response = requests.get(f"{self.orchestrator_url}/services", timeout=2)
            if response.status_code == 200:
                services_data = response.json()
                registered_services = [s.get('service_id', '') for s in services_data.get('value', [])]
                orchestrator_registered = service_id in registered_services
        except:
            pass
        
        # Check if process is running by looking for the service script
        process_running = False
        service_script = service.get('file', '')
        if service_script:
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                        cmdline = ' '.join(proc.info['cmdline'])
                        if service_script.replace('\\\\', '/').replace('\\\\', '/') in cmdline.replace('\\\\', '/'):
                            process_running = True
                            # Update the process object if we found it
                            try:
                                service["process"] = psutil.Process(proc.info['pid'])
                            except:
                                pass
                            break
            except:
                pass
        
        return process_running, orchestrator_registered'''
    
    # Find a good place to insert this method (after __init__ and before start_orchestrator)
    init_end_pattern = r'(        self\.setup_signal_handlers\(\)\n\n    def setup_signal_handlers)'
    content = re.sub(init_end_pattern, r'\\1' + new_method + '\\n\\n    def setup_signal_handlers', content)
    
    # Now fix the status checking logic to use the new method
    # Replace the STT status check
    old_stt_status = r'                status = "✅" if service\["process"\] and service\["process"\]\.poll\(\) is None else "❌"'
    new_stt_status = '''                process_running, orchestrator_registered = self.is_service_actually_running(service_id)
                if process_running and orchestrator_registered:
                    status = "✅ Running | ✅ Registered"
                elif process_running:
                    status = "✅ Running | ⏳ Connecting"
                elif orchestrator_registered:
                    status = "❌ Stopped | 🔄 Registered"
                else:
                    status = "❌ Stopped | ❌ Not Registered"'''
    
    content = re.sub(old_stt_status, new_stt_status, content)
    
    # Replace the LLM status check
    old_llm_status = r'                status = "✅" if service\["process"\] and service\["process"\]\.poll\(\) is None else "❌"'
    new_llm_status = '''                process_running, orchestrator_registered = self.is_service_actually_running(service_id)
                if process_running and orchestrator_registered:
                    status = "✅ Running | ✅ Registered"
                elif process_running:
                    status = "✅ Running | ⏳ Connecting"
                elif orchestrator_registered:
                    status = "❌ Stopped | 🔄 Registered"
                else:
                    status = "❌ Stopped | ❌ Not Registered"'''
    
    content = re.sub(old_llm_status, new_llm_status, content)
    
    # Replace the TTS status check
    old_tts_status = r'                status = "✅" if service\["process"\] and service\["process"\]\.poll\(\) is None else "❌"'
    new_tts_status = '''                process_running, orchestrator_registered = self.is_service_actually_running(service_id)
                if process_running and orchestrator_registered:
                    status = "✅ Running | ✅ Registered"
                elif process_running:
                    status = "✅ Running | ⏳ Connecting"
                elif orchestrator_registered:
                    status = "❌ Stopped | 🔄 Registered"
                else:
                    status = "❌ Stopped | ❌ Not Registered"'''
    
    content = re.sub(old_tts_status, new_tts_status, content)
    
    # Add psutil import if not present
    if 'import psutil' not in content:
        content = content.replace('import subprocess', 'import subprocess\\nimport psutil')
    
    # Write the updated content
    with open(launcher_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed service status detection logic")
    print("📋 Changes made:")
    print("   - Added is_service_actually_running() method")
    print("   - Fixed status display to check actual processes and orchestrator")
    print("   - Added detailed status information (Running/Stopped | Registered/Not Registered)")
    return True

def main():
    """Main function"""
    print("🔧 Fixing Launcher Status Display...")
    print("=" * 50)
    
    success = fix_status_detection()
    
    if success:
        print("\\n✅ STATUS DISPLAY FIXED!")
        print("🎯 Launcher will now properly show running services")
        print("📋 Status will show:")
        print("   ✅ Running | ✅ Registered - Service is working perfectly")
        print("   ✅ Running | ⏳ Connecting - Service is starting up") 
        print("   ❌ Stopped | 🔄 Registered - Service crashed but orchestrator hasn't cleaned up")
        print("   ❌ Stopped | ❌ Not Registered - Service is completely stopped")
        print("\\n💡 Note: You may need to install psutil: pip install psutil")
    else:
        print("\\n❌ Fix failed - please check manually")

if __name__ == "__main__":
    main()
