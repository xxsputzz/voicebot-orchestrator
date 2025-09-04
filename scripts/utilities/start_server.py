#!/usr/bin/env python3
"""
Startup script for Voicebot Orchestrator.
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def launch_in_new_terminal():
    """Launch the orchestrator in a new terminal window."""
    import subprocess
    import sys
    import os
    
    # Get the current script path
    script_path = os.path.abspath(__file__)
    python_exe = sys.executable
    
    try:
        # Launch in new PowerShell window that stays open
        cmd = f"Start-Process -FilePath 'powershell.exe' -ArgumentList '-NoExit', '-Command', 'cd \"{os.path.dirname(script_path)}\"; python \"{script_path}\" --direct' -WindowStyle Normal"
        subprocess.Popen([
            "powershell.exe", 
            "-Command", 
            cmd
        ], shell=True)
        
        print(">>> Launching Original Orchestrator in new terminal...")
        print(">>> Service will be available at: http://localhost:8000")
        print(">>> WebSocket endpoint: ws://localhost:8000/ws")
        print(">>> Original terminal is now free for other commands")
        
    except Exception as e:
        print(f"XXX Failed to launch in new terminal: {e}")
        print(">>> Falling back to current terminal...")
        return False
    
    return True

def main():
    """Main entry point for the application."""
    import sys
    
    # Check if this is a direct launch (from new terminal)
    if "--direct" in sys.argv:
        print(">>> Starting Voicebot Orchestrator in dedicated terminal...")
        print(">>> Port: 8000")
        print(">>> API: http://localhost:8000")
        print(">>> WebSocket: ws://localhost:8000/ws")
        print(">>> Close this window to stop the orchestrator")
        print("-" * 50)
        
        try:
            import uvicorn
            from voicebot_orchestrator.config import settings
            
            uvicorn.run(
                "voicebot_orchestrator.main:app",
                host=settings.host,
                port=settings.port,
                log_level=settings.log_level.lower(),
                reload=False,
                access_log=True
            )
            
        except KeyboardInterrupt:
            print("\n👋 Shutting down Voicebot Orchestrator")
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("   Please install requirements: pip install -r requirements.txt")
            sys.exit(1)
        except Exception as e:
            print(f"💥 Startup failed: {e}")
            sys.exit(1)
    else:
        # Try to launch in new terminal, fallback to current if it fails
        if not launch_in_new_terminal():
            # Fallback to current terminal
            try:
                import uvicorn
                from voicebot_orchestrator.config import settings
                
                print("🚀 Starting Voicebot Orchestrator")
                print(f"   Server: {settings.host}:{settings.port}")
                print(f"   Log Level: {settings.log_level}")
                print("   Press Ctrl+C to stop")
                print("-" * 40)
                
                uvicorn.run(
                    "voicebot_orchestrator.main:app",
                    host=settings.host,
                    port=settings.port,
                    log_level=settings.log_level.lower(),
                    reload=False,
                    access_log=True
                )
                
            except KeyboardInterrupt:
                print("\n👋 Shutting down Voicebot Orchestrator")
            except ImportError as e:
                print(f"❌ Missing dependency: {e}")
                print("   Please install requirements: pip install -r requirements.txt")
                sys.exit(1)
            except Exception as e:
                print(f"💥 Startup failed: {e}")
                sys.exit(1)

if __name__ == "__main__":
    main()
