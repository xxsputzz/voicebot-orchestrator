"""
Simple Sprint 4 CLI tests that verify functionality.
"""
import sys
import subprocess
import json
from pathlib import Path


def test_loan_calc_basic():
    """Test basic loan calculation."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'voicebot_orchestrator.sprint4_cli',
            'loan-calc', '--amount=10000', '--interest=0.06', '--months=12'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert 'Monthly Payment' in result.stdout
        assert 'Total Paid' in result.stdout
        print("✅ Loan calculation test passed")
        return True
    except Exception as e:
        print(f"❌ Loan calculation test failed: {e}")
        return False


def test_loan_calc_json():
    """Test loan calculation with JSON output."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'voicebot_orchestrator.sprint4_cli',
            '--json', 'loan-calc', '--amount=1000', '--interest=0.05', '--months=12'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert 'monthly_payment' in output
        assert 'total_paid' in output
        assert output['monthly_payment'] > 0
        
        print("✅ Loan calculation JSON test passed")
        return True
    except Exception as e:
        print(f"❌ Loan calculation JSON test failed: {e}")
        return False


def test_payment_plan_basic():
    """Test basic payment plan generation."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'voicebot_orchestrator.sprint4_cli',
            'payment-plan', '--amount=5000', '--interest=0.04', '--months=24'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert 'Payment Plan Results' in result.stdout
        assert 'Monthly Payment' in result.stdout
        print("✅ Payment plan test passed")
        return True
    except Exception as e:
        print(f"❌ Payment plan test failed: {e}")
        return False


def test_payment_plan_with_extra():
    """Test payment plan with extra payments."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'voicebot_orchestrator.sprint4_cli',
            '--json', 'payment-plan', '--amount=5000', '--interest=0.04', '--months=24', '--extra=100'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert 'monthly' in output
        assert 'total_paid' in output
        assert 'months_saved' in output
        
        print("✅ Payment plan with extra payments test passed")
        return True
    except Exception as e:
        print(f"❌ Payment plan with extra payments test failed: {e}")
        return False


def test_payment_plan_scenarios():
    """Test payment plan scenarios."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'voicebot_orchestrator.sprint4_cli',
            '--json', 'payment-plan', '--amount=10000', '--scenarios', 
            '--rates=0.04,0.06', '--terms=12,24'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert 'scenarios' in output
        assert len(output['scenarios']) == 4  # 2 rates × 2 terms
        
        print("✅ Payment plan scenarios test passed")
        return True
    except Exception as e:
        print(f"❌ Payment plan scenarios test failed: {e}")
        return False


def test_compliance_basic():
    """Test basic compliance prompts."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'voicebot_orchestrator.sprint4_cli',
            'compliance-test', '--type=KYC'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert 'Compliance Prompt' in result.stdout
        assert 'recorded for compliance' in result.stdout
        print("✅ Compliance basic test passed")
        return True
    except Exception as e:
        print(f"❌ Compliance basic test failed: {e}")
        return False


def test_compliance_flow():
    """Test compliance flow prompts."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'voicebot_orchestrator.sprint4_cli',
            '--json', 'compliance-test', '--flow=onboarding', '--stage=welcome'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert 'prompt_text' in output
        assert 'character_count' in output
        
        print("✅ Compliance flow test passed")
        return True
    except Exception as e:
        print(f"❌ Compliance flow test failed: {e}")
        return False


def test_compliance_validation():
    """Test compliance validation."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'voicebot_orchestrator.sprint4_cli',
            '--json', 'compliance-test', '--validate', '--conversation-type=transaction', 
            '--verified', '--consent'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert 'compliant' in output
        assert output['compliant'] == True
        
        print("✅ Compliance validation test passed")
        return True
    except Exception as e:
        print(f"❌ Compliance validation test failed: {e}")
        return False


def test_analytics():
    """Test analytics command."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'voicebot_orchestrator.sprint4_cli',
            'analytics'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert 'Sprint 4 Domain Logic Analytics' in result.stdout
        assert 'Loan Calculator' in result.stdout
        assert 'Payment Planner' in result.stdout
        assert 'Compliance Prompts' in result.stdout
        
        print("✅ Analytics test passed")
        return True
    except Exception as e:
        print(f"❌ Analytics test failed: {e}")
        return False


def test_cli_help():
    """Test CLI help command."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'voicebot_orchestrator.sprint4_cli', '--help'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert 'loan-calc' in result.stdout
        assert 'payment-plan' in result.stdout
        assert 'compliance-test' in result.stdout
        assert 'analytics' in result.stdout
        
        print("✅ CLI help test passed")
        return True
    except Exception as e:
        print(f"❌ CLI help test failed: {e}")
        return False


def run_sprint4_simple_tests():
    """Run all simple Sprint 4 tests."""
    print("🧪 Running Sprint 4 Simple CLI Tests")
    print("=" * 45)
    
    tests = [
        test_cli_help,
        test_loan_calc_basic,
        test_loan_calc_json,
        test_payment_plan_basic,
        test_payment_plan_with_extra,
        test_payment_plan_scenarios,
        test_compliance_basic,
        test_compliance_flow,
        test_compliance_validation,
        test_analytics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 45)
    print(f"Sprint 4 Tests: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All Sprint 4 CLI tests passed!")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_sprint4_simple_tests()
    sys.exit(0 if success else 1)
