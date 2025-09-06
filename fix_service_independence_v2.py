#!/usr/bin/env python3
"""
Fix Service Independence - Alternative Approach
==============================================
Use a different method to create truly independent services
"""

import re
from pathlib import Path

def fix_service_independence_v2():
    """Fix service independence using a better approach for Windows"""
    
    launcher_file = Path("comprehensive_ws_launcher.py")
    if not launcher_file.exists():
        print("❌ comprehensive_ws_launcher.py not found")
        return False
    
    print("🔧 Applying better service independence fix...")
    
    with open(launcher_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the current subprocess.Popen call and replace it with a better version
    old_pattern = r'''            # Start service process
            service\["process"\] = subprocess\.Popen\(\[
                sys\.executable, str\(script_path\)
            \], 
            cwd=os\.getcwd\(\), 
            stdout=subprocess\.PIPE, 
            stderr=subprocess\.PIPE, 
            text=True,
            creationflags=subprocess\.CREATE_NEW_PROCESS_GROUP \| subprocess\.DETACHED_PROCESS if os\.name == 'nt' else 0,
            preexec_fn=None if os\.name == 'nt' else os\.setsid
            \)'''
    
    # Better approach: Use CREATE_NEW_PROCESS_GROUP without DETACHED_PROCESS
    new_pattern = '''            # Start service process as independent background process
            if os.name == 'nt':  # Windows
                # Use CREATE_NEW_PROCESS_GROUP to separate from parent's signal handling
                # Don't use DETACHED_PROCESS as it makes tracking difficult
                service["process"] = subprocess.Popen([
                    sys.executable, str(script_path)
                ], 
                cwd=os.getcwd(), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
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
    
    # Apply the replacement
    content = re.sub(old_pattern, new_pattern, content, flags=re.MULTILINE | re.DOTALL)
    
    # Also modify the signal handler to NOT terminate child processes
    signal_handler_pattern = r'''        def signal_handler\(signum, frame\):
            print\("\\n🛑 Launcher shutting down gracefully..."\)
            print\("💡 Note: Services will continue running in background"\)
            print\("   Use option 5 to stop all services when needed"\)
            sys\.exit\(0\)'''
    
    better_signal_handler = '''        def signal_handler(signum, frame):
            print("\\n🛑 Launcher shutting down gracefully...")
            print("💡 Note: Services will continue running in background")
            print("   Use option 5 to stop all services when needed")
            # Don't terminate child processes - let them run independently
            os._exit(0)  # Exit immediately without cleanup that might kill children'''
    
    content = re.sub(signal_handler_pattern, better_signal_handler, content, flags=re.MULTILINE)
    
    # Write the updated content
    with open(launcher_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Applied better service independence fix")
    print("📋 Changes made:")
    print("   - Removed DETACHED_PROCESS flag (was causing issues)")
    print("   - Kept CREATE_NEW_PROCESS_GROUP for signal isolation")
    print("   - Changed signal handler to use os._exit(0)")
    return True

def main():
    """Main function"""
    print("🔧 Fixing Service Independence Issues (Version 2)...")
    print("=" * 60)
    
    success = fix_service_independence_v2()
    
    if success:
        print("\n✅ BETTER FIX APPLIED!")
        print("🎯 Services should now truly survive launcher shutdown")
        print("📋 Key changes:")
        print("   - Better process group separation")
        print("   - Immediate launcher exit without child process cleanup")
        print("   - Services run in separate process groups")
    else:
        print("\n❌ Fix failed - please check manually")

if __name__ == "__main__":
    main()
