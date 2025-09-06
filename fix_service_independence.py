#!/usr/bin/env python3
"""
Fix Service Independence
======================
Make services truly independent from the launcher so they survive Ctrl+C
"""

import re
from pathlib import Path

def fix_service_launching():
    """Fix the comprehensive launcher to use detached processes"""
    
    launcher_file = Path("comprehensive_ws_launcher.py")
    if not launcher_file.exists():
        print("❌ comprehensive_ws_launcher.py not found")
        return False
    
    print("🔧 Fixing service independence in comprehensive launcher...")
    
    with open(launcher_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the subprocess.Popen call for services and modify it
    old_popen_pattern = r'''            # Start service process
            service\["process"\] = subprocess\.Popen\(\[
                sys\.executable, str\(script_path\)
            \], cwd=os\.getcwd\(\), stdout=subprocess\.PIPE, stderr=subprocess\.PIPE, text=True\)'''
    
    # New pattern that creates detached processes on Windows
    new_popen_pattern = '''            # Start service process (detached from parent)
            import subprocess
            import os
            import sys
            
            # Create detached process on Windows
            if os.name == 'nt':  # Windows
                # Use CREATE_NEW_PROCESS_GROUP to detach from parent signal handling
                service["process"] = subprocess.Popen([
                    sys.executable, str(script_path)
                ], 
                cwd=os.getcwd(), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                )
            else:  # Unix-like systems
                # Use setsid to create new session
                service["process"] = subprocess.Popen([
                    sys.executable, str(script_path)
                ], 
                cwd=os.getcwd(), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                preexec_fn=os.setsid
                )'''
    
    if old_popen_pattern.replace(' ', '').replace('\n', '') in content.replace(' ', '').replace('\n', ''):
        content = re.sub(old_popen_pattern, new_popen_pattern, content, flags=re.MULTILINE)
        print("✅ Found and updated service launching code")
    else:
        # Try a simpler pattern match
        simple_pattern = r'service\["process"\] = subprocess\.Popen\(\[\s*sys\.executable, str\(script_path\)\s*\], cwd=os\.getcwd\(\), stdout=subprocess\.PIPE, stderr=subprocess\.PIPE, text=True\)'
        
        simple_replacement = '''service["process"] = subprocess.Popen([
                sys.executable, str(script_path)
            ], 
            cwd=os.getcwd(), 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == 'nt' else 0,
            preexec_fn=None if os.name == 'nt' else os.setsid
            )'''
        
        if re.search(simple_pattern, content):
            content = re.sub(simple_pattern, simple_replacement, content)
            print("✅ Found and updated service launching code (simple pattern)")
        else:
            print("❌ Could not find subprocess.Popen pattern to replace")
            return False
    
    # Also need to import os at the top if not already imported
    if 'import os' not in content[:500]:  # Check first 500 chars for imports
        content = content.replace('import subprocess', 'import subprocess\nimport os')
        print("✅ Added 'import os' for process flags")
    
    # Write the updated content
    with open(launcher_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed service independence - services will now survive launcher shutdown")
    return True

def add_signal_handler():
    """Add proper signal handling to the launcher"""
    
    launcher_file = Path("comprehensive_ws_launcher.py")
    
    with open(launcher_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add signal import if not present
    if 'import signal' not in content:
        content = content.replace('import subprocess', 'import subprocess\nimport signal')
        print("✅ Added signal import")
    
    # Add signal handler for graceful shutdown
    signal_handler = '''
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print("\\n🛑 Launcher shutting down gracefully...")
            print("💡 Note: Services will continue running in background")
            print("   Use option 5 to stop all services when needed")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
'''
    
    # Add the method to the class
    if 'def setup_signal_handlers(self):' not in content:
        # Find the class definition and add the method
        class_pattern = r'(class WebSocketServicesLauncher:.*?\n    def __init__\(self\):)'
        replacement = r'\1' + signal_handler + '\n'
        content = re.sub(class_pattern, replacement, content, flags=re.DOTALL)
        print("✅ Added signal handler method")
    
    # Call setup_signal_handlers in __init__
    init_pattern = r'(def __init__\(self\):.*?)(def )'
    if 'self.setup_signal_handlers()' not in content:
        def add_signal_setup(match):
            init_content = match.group(1)
            next_method = match.group(2)
            # Add the setup call before the next method
            return init_content + '        self.setup_signal_handlers()\n\n    ' + next_method
        
        content = re.sub(init_pattern, add_signal_setup, content, flags=re.DOTALL)
        print("✅ Added signal handler setup to __init__")
    
    # Write the updated content
    with open(launcher_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Added proper signal handling")

def main():
    """Main function"""
    print("🔧 Fixing Service Independence Issues...")
    print("=" * 50)
    
    success1 = fix_service_launching()
    if success1:
        add_signal_handler()
    
    if success1:
        print("\n✅ ALL FIXES APPLIED!")
        print("🎯 Services will now survive launcher Ctrl+C shutdown")
        print("📋 Test by:")
        print("   1. Start the launcher")
        print("   2. Start some services") 
        print("   3. Press Ctrl+C to close launcher")
        print("   4. Check services are still running")
    else:
        print("\n❌ Some fixes failed - please check manually")

if __name__ == "__main__":
    main()
