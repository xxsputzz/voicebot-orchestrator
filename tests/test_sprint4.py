"""
Tests for Sprint 4 Domain Logic functionality.

Tests loan calculations, payment plans, compliance prompts, and CLI commands.
"""

import unittest
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from voicebot_orchestrator.loan import (
        calculate_payment, calculate_total_interest, 
        calculate_remaining_balance, get_loan_analytics
    )
    from voicebot_orchestrator.payment_plan import (
        generate_payment_plan, generate_multiple_scenarios,
        generate_step_down_plan, get_payment_plan_analytics
    )
    from voicebot_orchestrator.compliance import (
        get_prompt, get_parameterized_prompt, get_flow_prompt,
        validate_compliance_requirements, get_compliance_analytics
    )
    from voicebot_orchestrator.sprint4_cli import (
        loan_calc_command, payment_plan_command, compliance_test_command
    )
    REAL_IMPORTS = True
except ImportError:
    REAL_IMPORTS = False


@unittest.skipUnless(REAL_IMPORTS, "Real imports not available")
class TestLoanCalculator(unittest.TestCase):
    """Test cases for loan calculation functionality."""
    
    async def asyncSetUp(self):
        """Set up test fixtures."""
        pass
    
    def test_calculate_payment_basic(self):
        """Test basic loan payment calculation."""
        async def run_test():
            monthly, schedule = await calculate_payment(10000, 0.06, 12)
            
            self.assertIsInstance(monthly, float)
            self.assertGreater(monthly, 0)
            self.assertEqual(len(schedule), 12)
            
            # Check schedule structure
            for entry in schedule:
                self.assertIn("month", entry)
                self.assertIn("interest", entry)
                self.assertIn("principal", entry)
                self.assertIn("balance", entry)
                self.assertGreaterEqual(entry["month"], 1)
                self.assertGreaterEqual(entry["interest"], 0)
                self.assertGreaterEqual(entry["principal"], 0)
                self.assertGreaterEqual(entry["balance"], 0)
        
        asyncio.run(run_test())
    
    def test_calculate_payment_zero_interest(self):
        """Test loan calculation with zero interest rate."""
        async def run_test():
            monthly, schedule = await calculate_payment(1200, 0.0, 12)
            
            self.assertEqual(monthly, 100.0)  # 1200 / 12
            self.assertEqual(len(schedule), 12)
            
            # All interest should be zero
            for entry in schedule:
                self.assertEqual(entry["interest"], 0.0)
        
        asyncio.run(run_test())
    
    def test_calculate_payment_invalid_inputs(self):
        """Test loan calculation with invalid inputs."""
        async def run_test():
            # Negative principal
            with self.assertRaises(ValueError):
                await calculate_payment(-1000, 0.05, 12)
            
            # Invalid interest rate
            with self.assertRaises(ValueError):
                await calculate_payment(1000, 1.5, 12)  # > 1
            
            with self.assertRaises(ValueError):
                await calculate_payment(1000, -0.01, 12)  # < 0
            
            # Invalid months
            with self.assertRaises(ValueError):
                await calculate_payment(1000, 0.05, -12)
            
            with self.assertRaises(ValueError):
                await calculate_payment(1000, 0.05, 0)
        
        asyncio.run(run_test())
    
    def test_calculate_total_interest(self):
        """Test total interest calculation."""
        async def run_test():
            _, schedule = await calculate_payment(10000, 0.06, 12)
            total_interest = calculate_total_interest(schedule)
            
            self.assertIsInstance(total_interest, float)
            self.assertGreater(total_interest, 0)
            
            # Verify it matches sum of schedule
            manual_total = sum(entry["interest"] for entry in schedule)
            self.assertAlmostEqual(total_interest, manual_total, places=2)
        
        asyncio.run(run_test())
    
    def test_calculate_remaining_balance(self):
        """Test remaining balance calculation."""
        balance = calculate_remaining_balance(10000, 0.06, 12, 6)
        
        self.assertIsInstance(balance, float)
        self.assertGreaterEqual(balance, 0)
        self.assertLess(balance, 10000)  # Should be less than original
        
        # Test edge cases
        self.assertEqual(calculate_remaining_balance(10000, 0.06, 12, 12), 0.0)
        self.assertEqual(calculate_remaining_balance(10000, 0.06, 12, 15), 0.0)  # Over-payment


@unittest.skipUnless(REAL_IMPORTS, "Real imports not available")
class TestPaymentPlan(unittest.TestCase):
    """Test cases for payment plan generation."""
    
    def test_generate_payment_plan_basic(self):
        """Test basic payment plan generation."""
        async def run_test():
            plan = await generate_payment_plan(10000, 0.06, 12, 0)
            
            self.assertIn("monthly", plan)
            self.assertIn("schedule", plan)
            self.assertIn("total_paid", plan)
            self.assertIn("total_interest", plan)
            
            self.assertIsInstance(plan["monthly"], float)
            self.assertIsInstance(plan["schedule"], list)
            self.assertGreater(plan["monthly"], 0)
            self.assertEqual(len(plan["schedule"]), 12)
        
        asyncio.run(run_test())
    
    def test_generate_payment_plan_with_extra(self):
        """Test payment plan with extra payments."""
        async def run_test():
            plan = await generate_payment_plan(10000, 0.06, 12, 100)
            
            self.assertIn("months_saved", plan)
            self.assertIn("actual_months", plan)
            
            # With extra payments, should pay off faster
            self.assertLessEqual(plan["actual_months"], 12)
            
            # Total interest should be less than without extra payments
            plan_no_extra = await generate_payment_plan(10000, 0.06, 12, 0)
            self.assertLess(plan["total_interest"], plan_no_extra["total_interest"])
        
        asyncio.run(run_test())
    
    def test_generate_multiple_scenarios(self):
        """Test multiple scenario generation."""
        async def run_test():
            rates = [0.04, 0.06, 0.08]
            terms = [12, 24, 36]
            
            scenarios = await generate_multiple_scenarios(10000, rates, terms)
            
            self.assertEqual(len(scenarios), 9)  # 3 rates × 3 terms
            
            for scenario in scenarios:
                self.assertIn("scenario_id", scenario)
                self.assertIn("rate_percent", scenario)
                self.assertIn("term_months", scenario)
                self.assertIn("monthly_payment", scenario)
                self.assertIn("total_paid", scenario)
        
        asyncio.run(run_test())
    
    def test_generate_step_down_plan(self):
        """Test step-down rate payment plan."""
        async def run_test():
            rate_reductions = [
                {"month": 6, "new_rate": 0.05},
                {"month": 12, "new_rate": 0.04}
            ]
            
            plan = await generate_step_down_plan(10000, 0.06, rate_reductions, 24)
            
            self.assertIn("schedule", plan)
            self.assertIn("total_paid", plan)
            self.assertIn("total_interest", plan)
            
            # Check that rates change at specified months
            schedule = plan["schedule"]
            self.assertEqual(schedule[5]["rate"], 5.0)  # Month 6, rate 5%
            self.assertEqual(schedule[11]["rate"], 4.0)  # Month 12, rate 4%
        
        asyncio.run(run_test())


@unittest.skipUnless(REAL_IMPORTS, "Real imports not available")
class TestCompliance(unittest.TestCase):
    """Test cases for compliance prompts."""
    
    def test_get_prompt_valid_types(self):
        """Test getting valid compliance prompts."""
        valid_types = ["KYC", "opt-out", "legal", "privacy", "recording", "data-retention"]
        
        for prompt_type in valid_types:
            prompt = get_prompt(prompt_type)
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 0)
    
    def test_get_prompt_invalid_type(self):
        """Test getting prompt with invalid type."""
        with self.assertRaises(ValueError):
            get_prompt("INVALID_TYPE")
    
    def test_get_parameterized_prompt(self):
        """Test parameterized prompt generation."""
        params = {
            "customer_name": "John Doe",
            "account_suffix": "1234"
        }
        
        prompt = get_parameterized_prompt("account-verification", params)
        self.assertIn("John Doe", prompt)
        
        # Test missing parameter
        with self.assertRaises(ValueError):
            get_parameterized_prompt("account-verification", {})
    
    def test_get_flow_prompt(self):
        """Test flow-based prompt generation."""
        prompt = get_flow_prompt("onboarding", "welcome")
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)
        
        # Test with parameters
        params = {"auth_method": "SMS code"}
        prompt = get_flow_prompt("authentication", "challenge", params)
        self.assertIn("SMS code", prompt)
        
        # Test invalid flow
        with self.assertRaises(ValueError):
            get_flow_prompt("invalid_flow", "start")
        
        # Test invalid stage
        with self.assertRaises(ValueError):
            get_flow_prompt("onboarding", "invalid_stage")
    
    def test_validate_compliance_requirements(self):
        """Test compliance requirements validation."""
        # Valid transaction
        result = validate_compliance_requirements(
            "transaction", 
            customer_verified=True, 
            recording_consent=True
        )
        self.assertTrue(result["compliant"])
        self.assertEqual(len(result["errors"]), 0)
        
        # Invalid transaction (missing verification)
        result = validate_compliance_requirements(
            "transaction",
            customer_verified=False,
            recording_consent=True
        )
        self.assertFalse(result["compliant"])
        self.assertGreater(len(result["errors"]), 0)
        
        # General info (no requirements)
        result = validate_compliance_requirements("general_info")
        self.assertTrue(result["compliant"])


class TestSprint4CLI(unittest.TestCase):
    """Test cases for Sprint 4 CLI commands."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_args = MagicMock()
    
    @patch('sys.stdout')
    def test_loan_calc_command_json(self, mock_stdout):
        """Test loan calculation command with JSON output."""
        if not REAL_IMPORTS:
            self.skipTest("Real imports not available")
        
        async def run_test():
            self.mock_args.amount = 10000
            self.mock_args.interest = 0.06
            self.mock_args.months = 12
            self.mock_args.json = True
            self.mock_args.verbose = False
            
            await loan_calc_command(self.mock_args)
        
        asyncio.run(run_test())
    
    @patch('sys.stdout')
    def test_payment_plan_command_scenarios(self, mock_stdout):
        """Test payment plan command with scenarios."""
        if not REAL_IMPORTS:
            self.skipTest("Real imports not available")
        
        async def run_test():
            self.mock_args.amount = 10000
            self.mock_args.scenarios = True
            self.mock_args.rates = "0.04,0.06,0.08"
            self.mock_args.terms = "12,24,36"
            self.mock_args.extra = 0
            self.mock_args.json = True
            self.mock_args.verbose = False
            
            await payment_plan_command(self.mock_args)
        
        asyncio.run(run_test())
    
    @patch('sys.stdout')
    def test_compliance_test_command(self, mock_stdout):
        """Test compliance test command."""
        if not REAL_IMPORTS:
            self.skipTest("Real imports not available")
        
        self.mock_args.type = "KYC"
        self.mock_args.flow = None
        self.mock_args.stage = None
        self.mock_args.parameterized = False
        self.mock_args.params = None
        self.mock_args.validate = False
        self.mock_args.json = False
        
        compliance_test_command(self.mock_args)


class TestAnalytics(unittest.TestCase):
    """Test cases for analytics functionality."""
    
    def test_loan_analytics(self):
        """Test loan calculation analytics."""
        if not REAL_IMPORTS:
            self.skipTest("Real imports not available")
        
        analytics = get_loan_analytics()
        self.assertIn("total_calculations", analytics)
        self.assertIn("service_name", analytics)
        self.assertEqual(analytics["service_name"], "loan_calculator")
    
    def test_payment_plan_analytics(self):
        """Test payment plan analytics."""
        if not REAL_IMPORTS:
            self.skipTest("Real imports not available")
        
        analytics = get_payment_plan_analytics()
        self.assertIn("total_plans_generated", analytics)
        self.assertIn("service_name", analytics)
        self.assertEqual(analytics["service_name"], "payment_plan_generator")
    
    def test_compliance_analytics(self):
        """Test compliance analytics."""
        if not REAL_IMPORTS:
            self.skipTest("Real imports not available")
        
        analytics = get_compliance_analytics()
        self.assertIn("total_prompts_served", analytics)
        self.assertIn("service_name", analytics)
        self.assertEqual(analytics["service_name"], "compliance_prompts")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_loan_calculator_edge_cases(self):
        """Test loan calculator edge cases."""
        if not REAL_IMPORTS:
            self.skipTest("Real imports not available")
        
        async def run_test():
            # Very small loan
            monthly, schedule = await calculate_payment(1, 0.01, 1)
            self.assertGreater(monthly, 0)
            self.assertEqual(len(schedule), 1)
            
            # Very high interest (but valid)
            monthly, schedule = await calculate_payment(1000, 0.99, 12)
            self.assertGreater(monthly, 0)
            self.assertEqual(len(schedule), 12)
        
        asyncio.run(run_test())
    
    def test_payment_plan_edge_cases(self):
        """Test payment plan edge cases."""
        if not REAL_IMPORTS:
            self.skipTest("Real imports not available")
        
        async def run_test():
            # Extra payment larger than monthly payment
            plan = await generate_payment_plan(1000, 0.06, 12, 200)
            self.assertIn("actual_months", plan)
            self.assertLess(plan["actual_months"], 12)
            
            # Zero rate with extra payment
            plan = await generate_payment_plan(1200, 0.0, 12, 50)
            self.assertGreater(plan["months_saved"], 0)
        
        asyncio.run(run_test())


def run_sprint4_tests():
    """Run all Sprint 4 tests."""
    print("🧪 Running Sprint 4 Domain Logic Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestLoanCalculator,
        TestPaymentPlan, 
        TestCompliance,
        TestSprint4CLI,
        TestAnalytics,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_sprint4_tests()
    sys.exit(0 if success else 1)
