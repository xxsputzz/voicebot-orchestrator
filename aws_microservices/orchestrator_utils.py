#!/usr/bin/env python3
"""
Orchestrator Utility Functions
Shared utilities for WebSocket services to manage orchestrator auto-start
"""

import os
import sys
import time
import socket
import logging
import subprocess
from pathlib import Path

def check_port_available(host: str, port: int) -> bool:
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0  # Port is available if connection failed
    except Exception:
        return False

def is_orchestrator_running() -> bool:
    """Check if orchestrator is already running on expected ports"""
    return (
        not check_port_available('localhost', 9000) and 
        not check_port_available('localhost', 9001) and
        not check_port_available('localhost', 8080)
    )

def start_orchestrator_if_needed() -> bool:
    """
    Start orchestrator if it's not already running. 
    Returns True if orchestrator is available.
    This function ensures only one orchestrator instance runs.
    """
    if is_orchestrator_running():
        logging.info("✅ Orchestrator is already running")
        return True
    
    logging.info("🚀 Starting WebSocket Orchestrator (single instance)...")
    
    try:
        # Find orchestrator script
        current_dir = Path(__file__).parent.parent  # Go up from aws_microservices to root
        orchestrator_script = current_dir / "ws_orchestrator_service.py"
        
        if not orchestrator_script.exists():
            logging.error(f"❌ Orchestrator script not found: {orchestrator_script}")
            return False
        
        # Start orchestrator in background
        process = subprocess.Popen([
            sys.executable, str(orchestrator_script)
        ], cwd=str(current_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for orchestrator to start (check ports become unavailable)
        max_wait = 10  # 10 seconds max wait
        wait_interval = 0.5
        waited = 0
        
        while waited < max_wait:
            if is_orchestrator_running():
                logging.info("✅ Orchestrator started successfully (shared instance)")
                return True
            time.sleep(wait_interval)
            waited += wait_interval
        
        logging.error("❌ Orchestrator failed to start within timeout")
        return False
        
    except Exception as e:
        logging.error(f"❌ Error starting orchestrator: {e}")
        return False

def get_orchestrator_status() -> dict:
    """Get orchestrator status information"""
    return {
        'running': is_orchestrator_running(),
        'ports': {
            'client_ws': 9000,
            'service_ws': 9001, 
            'http_api': 8080
        },
        'endpoints': {
            'client_websocket': 'ws://localhost:9000',
            'service_websocket': 'ws://localhost:9001',
            'http_api': 'http://localhost:8080'
        }
    }
