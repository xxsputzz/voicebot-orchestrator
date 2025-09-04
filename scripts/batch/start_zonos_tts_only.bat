@echo off
echo ========================================
echo 🎙️ ZONOS TTS DEDICATED STARTUP MANAGER
echo ========================================
echo.
echo This ensures Zonos TTS runs without port conflicts
echo.

echo 📋 Step 1: Checking for port conflicts...
python -c "
import socket
import subprocess
import sys

def is_port_in_use(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            return result == 0
    except:
        return False

port = 8014
if is_port_in_use(port):
    print(f'⚠️ Port {port} is in use - cleaning up...')
    # Kill any existing processes on this port
    try:
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, shell=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        pid = int(parts[-1])
                        subprocess.run(['taskkill', '/F', '/PID', str(pid)], capture_output=True)
                        print(f'✅ Terminated process {pid}')
                    except:
                        pass
    except Exception as e:
        print(f'Error during cleanup: {e}')
else:
    print(f'✅ Port {port} is available')
"

echo.
echo 📋 Step 2: Starting Zonos TTS service...
echo 🌐 URL: http://localhost:8014
echo 🎙️ Enhanced real speech synthesis active
echo.

cd /d "%~dp0"
python aws_microservices/tts_zonos_service.py

echo.
echo 📋 Service stopped.
pause
