"""
Simple Sprint 3 CLI tests that verify functionality.
"""
import sys
import subprocess
from pathlib import Path

def test_sprint3_cli_help():
    """Test Sprint 3 CLI help command."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'voicebot_orchestrator.sprint3_cli', '--help'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert 'orchestrator-log' in result.stdout
        assert 'monitor-session' in result.stdout
        assert 'analytics-report' in result.stdout
        print("✅ CLI help test passed")
        return True
    except Exception as e:
        print(f"❌ CLI help test failed: {e}")
        return False

def test_orchestrator_log_metrics():
    """Test orchestrator log metrics command."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'voicebot_orchestrator.sprint3_cli', 
            'orchestrator-log', '--metrics'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert 'METRICS SNAPSHOT' in result.stdout
        assert 'Average Handle Time' in result.stdout
        print("✅ Orchestrator log metrics test passed")
        return True
    except Exception as e:
        print(f"❌ Orchestrator log metrics test failed: {e}")
        return False

def test_monitor_session_stats():
    """Test monitor session stats command."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'voicebot_orchestrator.sprint3_cli',
            'monitor-session', '--stats'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert 'SESSION STATISTICS' in result.stdout
        assert 'Global Statistics' in result.stdout or 'Active Sessions' in result.stdout
        print("✅ Monitor session stats test passed")
        return True
    except Exception as e:
        print(f"❌ Monitor session stats test failed: {e}")
        return False

def test_analytics_report():
    """Test analytics report command."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'voicebot_orchestrator.sprint3_cli',
            'analytics-report', '--export=txt'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert 'ANALYTICS REPORT' in result.stdout
        print("✅ Analytics report test passed")
        return True
    except Exception as e:
        print(f"❌ Analytics report test failed: {e}")
        return False

def test_business_dashboard():
    """Test business dashboard command."""
    try:
        # Test the dashboard with unbuffered output
        process = subprocess.Popen([
            sys.executable, '-u', '-m', 'voicebot_orchestrator.sprint3_cli',
            'dashboard', '--refresh=1'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
        text=True, cwd=Path(__file__).parent.parent, bufsize=0)
        
        # Let it run for 3 seconds to capture initial output
        try:
            stdout, stderr = process.communicate(timeout=3)
            # If it finishes quickly, that's unexpected for a dashboard
            if stdout and 'LIVE METRICS DASHBOARD' in stdout:
                print("✅ Business dashboard test passed")
                return True
            else:
                print(f"❌ Business dashboard test failed: Dashboard exited unexpectedly")
                return False
        except subprocess.TimeoutExpired:
            # This is expected - dashboard should be running indefinitely
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            
            # Check if it started properly by looking for dashboard output
            if stdout and ('LIVE METRICS DASHBOARD' in stdout or 'CURRENT METRICS' in stdout):
                print("✅ Business dashboard test passed")
                return True
            else:
                print(f"❌ Business dashboard test failed: Expected dashboard output not found")
                print(f"STDOUT length: {len(stdout) if stdout else 0}")
                print(f"STDERR length: {len(stderr) if stderr else 0}")
                if stdout:
                    print(f"STDOUT preview: {stdout[:200]}...")
                if stderr:
                    print(f"STDERR preview: {stderr[:200]}...")
                return False
                
    except Exception as e:
        print(f"❌ Business dashboard test failed: {e}")
        return False

def run_sprint3_tests():
    """Run all Sprint 3 tests."""
    print("🧪 Running Sprint 3 CLI Tests")
    print("=" * 35)
    
    tests = [
        test_sprint3_cli_help,
        test_orchestrator_log_metrics,
        test_monitor_session_stats,
        test_analytics_report,
        test_business_dashboard
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 35)
    print(f"Sprint 3 Tests: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All Sprint 3 CLI tests passed!")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = run_sprint3_tests()
    sys.exit(0 if success else 1)
