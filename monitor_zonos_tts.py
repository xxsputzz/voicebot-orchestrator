#!/usr/bin/env python3
"""
🔍 ZONOS TTS SERVICE MONITOR
===========================
Monitors the Zonos TTS service and provides status information
"""

import requests
import time
import socket
import subprocess
import sys
from datetime import datetime

class ZonosTTSMonitor:
    def __init__(self, port=8014):
        self.port = port
        self.base_url = f"http://localhost:{port}"
        
    def check_port_available(self):
        """Check if port is in use"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', self.port))
                return result == 0
        except:
            return False
    
    def check_service_health(self):
        """Check if service is responding"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=3)
            return response.status_code == 200
        except:
            try:
                response = requests.get(f"{self.base_url}/", timeout=3)
                return response.status_code in [200, 404]
            except:
                return False
    
    def test_synthesis(self):
        """Test actual TTS synthesis"""
        try:
            test_data = {
                'text': 'Quick test',
                'voice': 'default',
                'emotion': 'neutral',
                'seed': 999
            }
            
            response = requests.post(f"{self.base_url}/synthesize", json=test_data, timeout=5)
            return response.status_code == 200 and len(response.content) > 1000
        except:
            return False
    
    def get_status_report(self):
        """Get comprehensive status report"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = {
            'timestamp': now,
            'port_in_use': self.check_port_available(),
            'service_responding': False,
            'synthesis_working': False,
            'status': 'UNKNOWN'
        }
        
        if report['port_in_use']:
            report['service_responding'] = self.check_service_health()
            
            if report['service_responding']:
                report['synthesis_working'] = self.test_synthesis()
                
                if report['synthesis_working']:
                    report['status'] = 'HEALTHY'
                else:
                    report['status'] = 'RESPONDING_BUT_SYNTHESIS_FAILED'
            else:
                report['status'] = 'PORT_IN_USE_BUT_NOT_RESPONDING'
        else:
            report['status'] = 'SERVICE_NOT_RUNNING'
            
        return report
    
    def print_status(self):
        """Print formatted status"""
        report = self.get_status_report()
        
        print(f"🔍 ZONOS TTS SERVICE STATUS - {report['timestamp']}")
        print("=" * 60)
        
        # Port status
        port_emoji = "🟢" if report['port_in_use'] else "🔴"
        print(f"{port_emoji} Port {self.port}: {'IN USE' if report['port_in_use'] else 'AVAILABLE'}")
        
        # Service health
        if report['service_responding']:
            health_emoji = "🟢"
            health_status = "RESPONDING"
        else:
            health_emoji = "🔴" if report['port_in_use'] else "⚫"
            health_status = "NOT RESPONDING" if report['port_in_use'] else "N/A"
        print(f"{health_emoji} Service Health: {health_status}")
        
        # Synthesis capability
        if report['service_responding']:
            synth_emoji = "🟢" if report['synthesis_working'] else "🔴"
            synth_status = "WORKING" if report['synthesis_working'] else "FAILED"
        else:
            synth_emoji = "⚫"
            synth_status = "N/A"
        print(f"{synth_emoji} TTS Synthesis: {synth_status}")
        
        # Overall status
        status_colors = {
            'HEALTHY': '🟢',
            'RESPONDING_BUT_SYNTHESIS_FAILED': '🟡',
            'PORT_IN_USE_BUT_NOT_RESPONDING': '🔴', 
            'SERVICE_NOT_RUNNING': '⚫'
        }
        
        status_emoji = status_colors.get(report['status'], '❓')
        print(f"\n{status_emoji} OVERALL STATUS: {report['status']}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if report['status'] == 'HEALTHY':
            print("   ✅ Service is working perfectly!")
            print(f"   🌐 Ready to use: {self.base_url}")
        elif report['status'] == 'SERVICE_NOT_RUNNING':
            print("   🚀 Start service: start_zonos_tts_only.bat")
            print("   🔧 Or use: python fix_zonos_port_conflicts.py")
        elif report['status'] == 'PORT_IN_USE_BUT_NOT_RESPONDING':
            print("   🔄 Restart service: stop current process and restart")
            print("   🧹 Clean conflicts: python fix_zonos_port_conflicts.py")
        elif report['status'] == 'RESPONDING_BUT_SYNTHESIS_FAILED':
            print("   🔧 Check TTS engine installation")
            print("   📥 Reinstall: install_real_tts.bat")
        
        return report

def monitor_continuous():
    """Continuous monitoring mode"""
    monitor = ZonosTTSMonitor()
    
    print("🔄 CONTINUOUS MONITORING MODE")
    print("Press Ctrl+C to stop")
    print("=" * 40)
    
    try:
        while True:
            report = monitor.get_status_report()
            status_emoji = {
                'HEALTHY': '🟢',
                'RESPONDING_BUT_SYNTHESIS_FAILED': '🟡',
                'PORT_IN_USE_BUT_NOT_RESPONDING': '🔴',
                'SERVICE_NOT_RUNNING': '⚫'
            }.get(report['status'], '❓')
            
            print(f"{report['timestamp']} {status_emoji} {report['status']}")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        monitor_continuous()
    else:
        monitor = ZonosTTSMonitor()
        monitor.print_status()

if __name__ == "__main__":
    main()
