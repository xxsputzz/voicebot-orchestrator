"""
Sprint 4: Domain Logic CLI

Command-line interface for loan calculations, payment plans, and compliance testing.
Provides JSON and human-readable output formats.
"""

import asyncio
import argparse
import json
import sys
from typing import Dict, Any, List, Union

try:
    from .loan import calculate_payment_with_analytics, get_loan_analytics
    from .payment_plan import (
        generate_payment_plan_with_analytics, 
        generate_multiple_scenarios,
        get_payment_plan_analytics
    )
    from .compliance import (
        get_prompt_with_analytics, 
        get_parameterized_prompt,
        get_flow_prompt,
        validate_compliance_requirements,
        get_compliance_analytics
    )
    REAL_IMPORTS = True
except ImportError:
    # Mock implementations for testing
    REAL_IMPORTS = False
    
    class MockModule:
        @staticmethod
        async def calculate_payment_with_analytics(principal, rate, months):
            return 100.0, [{"month": 1, "interest": 5.0, "principal": 95.0, "balance": 0.0}]
        
        @staticmethod
        def get_loan_analytics():
            return {"total_calculations": 1}
        
        @staticmethod
        async def generate_payment_plan_with_analytics(principal, rate, months, extra=0):
            return {"monthly": 100.0, "schedule": [], "total_paid": 1200.0}
        
        @staticmethod
        async def generate_multiple_scenarios(principal, rates, terms, extra=None):
            return [{"scenario_id": 1, "monthly_payment": 100.0}]
        
        @staticmethod
        def get_payment_plan_analytics():
            return {"total_plans_generated": 1}
        
        @staticmethod
        def get_prompt_with_analytics(prompt_type):
            return f"Mock compliance prompt for {prompt_type}"
        
        @staticmethod
        def get_parameterized_prompt(prompt_type, params=None):
            return f"Mock parameterized prompt for {prompt_type}"
        
        @staticmethod
        def get_flow_prompt(flow_type, stage, params=None):
            return f"Mock flow prompt for {flow_type} at {stage}"
        
        @staticmethod
        def validate_compliance_requirements(conv_type, verified=False, consent=False):
            return {"compliant": True, "errors": [], "required_actions": []}
        
        @staticmethod
        def get_compliance_analytics():
            return {"total_prompts_served": 1}
    
    mock = MockModule()
    calculate_payment_with_analytics = mock.calculate_payment_with_analytics
    get_loan_analytics = mock.get_loan_analytics
    generate_payment_plan_with_analytics = mock.generate_payment_plan_with_analytics
    generate_multiple_scenarios = mock.generate_multiple_scenarios
    get_payment_plan_analytics = mock.get_payment_plan_analytics
    get_prompt_with_analytics = mock.get_prompt_with_analytics
    get_parameterized_prompt = mock.get_parameterized_prompt
    get_flow_prompt = mock.get_flow_prompt
    validate_compliance_requirements = mock.validate_compliance_requirements
    get_compliance_analytics = mock.get_compliance_analytics


def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    return f"${amount:,.2f}"


def format_percentage(rate: float) -> str:
    """Format decimal rate as percentage string."""
    return f"{rate * 100:.2f}%"


async def loan_calc_command(args: argparse.Namespace) -> None:
    """Execute loan calculation command."""
    try:
        monthly, schedule = await calculate_payment_with_analytics(
            args.amount, args.interest, args.months
        )
        
        if args.json:
            result = {
                "monthly_payment": monthly,
                "total_payments": len(schedule),
                "total_paid": sum(entry["interest"] + entry["principal"] for entry in schedule),
                "total_interest": sum(entry["interest"] for entry in schedule),
                "schedule": schedule if args.verbose else schedule[:3]  # First 3 months only
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"Loan Calculator Results")
            print(f"=" * 40)
            print(f"Loan Amount: {format_currency(args.amount)}")
            print(f"Interest Rate: {format_percentage(args.interest)}")
            print(f"Term: {args.months} months")
            print()
            print(f"Monthly Payment: {format_currency(monthly)}")
            
            total_paid = monthly * args.months
            total_interest = total_paid - args.amount
            print(f"Total Paid: {format_currency(total_paid)}")
            print(f"Total Interest: {format_currency(total_interest)}")
            
            if args.verbose:
                print(f"\nAmortization Schedule:")
                print(f"{'Month':<6} {'Payment':<10} {'Interest':<10} {'Principal':<10} {'Balance':<12}")
                print("-" * 50)
                for entry in schedule[:12]:  # Show first year
                    payment = entry["interest"] + entry["principal"]
                    print(f"{entry['month']:<6} {format_currency(payment):<10} "
                          f"{format_currency(entry['interest']):<10} "
                          f"{format_currency(entry['principal']):<10} "
                          f"{format_currency(entry['balance']):<12}")
                
                if len(schedule) > 12:
                    print(f"... and {len(schedule) - 12} more payments")
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


async def payment_plan_command(args: argparse.Namespace) -> None:
    """Execute payment plan command."""
    try:
        if args.scenarios:
            # Multiple scenario comparison
            rates = [float(r) for r in args.rates.split(",")]
            terms = [int(t) for t in args.terms.split(",")]
            extra_payments = [0.0] if not args.extra else [0.0, args.extra]
            
            scenarios = await generate_multiple_scenarios(
                args.amount, rates, terms, extra_payments
            )
            
            if args.json:
                print(json.dumps({"scenarios": scenarios}, indent=2))
            else:
                print("Payment Plan Scenarios")
                print("=" * 60)
                print(f"{'ID':<3} {'Rate':<6} {'Term':<5} {'Extra':<8} {'Monthly':<10} {'Total':<12}")
                print("-" * 60)
                for scenario in scenarios:
                    print(f"{scenario['scenario_id']:<3} "
                          f"{scenario['rate_percent']:<6.2f}% "
                          f"{scenario['term_months']:<5} "
                          f"{format_currency(scenario['extra_payment']):<8} "
                          f"{format_currency(scenario['monthly_payment']):<10} "
                          f"{format_currency(scenario['total_paid']):<12}")
        else:
            # Single payment plan
            plan = await generate_payment_plan_with_analytics(
                args.amount, args.interest, args.months, args.extra
            )
            
            if args.json:
                if not args.verbose:
                    # Remove detailed schedule for concise output
                    plan_summary = {k: v for k, v in plan.items() if k != "schedule"}
                    plan_summary["schedule_length"] = len(plan["schedule"])
                    print(json.dumps(plan_summary, indent=2))
                else:
                    print(json.dumps(plan, indent=2))
            else:
                print("Payment Plan Results")
                print("=" * 40)
                print(f"Loan Amount: {format_currency(args.amount)}")
                print(f"Interest Rate: {format_percentage(args.interest)}")
                print(f"Term: {args.months} months")
                if args.extra > 0:
                    print(f"Extra Payment: {format_currency(args.extra)}")
                print()
                print(f"Monthly Payment: {format_currency(plan['monthly'])}")
                print(f"Total Paid: {format_currency(plan['total_paid'])}")
                print(f"Total Interest: {format_currency(plan['total_interest'])}")
                
                if "months_saved" in plan and plan["months_saved"] > 0:
                    print(f"Months Saved: {plan['months_saved']}")
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def compliance_test_command(args: argparse.Namespace) -> None:
    """Execute compliance test command."""
    try:
        if args.flow:
            # Flow-based prompt
            params = {}
            if args.params:
                # Parse key=value pairs
                for param in args.params.split(","):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        params[key.strip()] = value.strip()
            
            prompt = get_flow_prompt(args.flow, args.stage, params)
        
        elif args.parameterized:
            # Parameterized prompt
            params = {}
            if args.params:
                for param in args.params.split(","):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        params[key.strip()] = value.strip()
            
            prompt = get_parameterized_prompt(args.type, params)
        
        elif args.validate:
            # Compliance validation
            validation = validate_compliance_requirements(
                args.conversation_type,
                customer_verified=args.verified,
                recording_consent=args.consent
            )
            
            if args.json:
                print(json.dumps(validation, indent=2))
            else:
                print("Compliance Validation Results")
                print("=" * 40)
                print(f"Conversation Type: {args.conversation_type}")
                print(f"Compliant: {'Yes' if validation['compliant'] else 'No'}")
                
                if validation["errors"]:
                    print("\nErrors:")
                    for error in validation["errors"]:
                        print(f"  - {error}")
                
                if validation["required_actions"]:
                    print("\nRequired Actions:")
                    for action in validation["required_actions"]:
                        print(f"  - {action}")
            return
        
        else:
            # Standard prompt
            prompt = get_prompt_with_analytics(args.type)
        
        if args.json:
            result = {
                "prompt_type": args.type,
                "prompt_text": prompt,
                "character_count": len(prompt)
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"Compliance Prompt ({args.type})")
            print("=" * 50)
            print(prompt)
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def analytics_command(args: argparse.Namespace) -> None:
    """Execute analytics command."""
    analytics_data = {
        "loan_calculator": get_loan_analytics(),
        "payment_planner": get_payment_plan_analytics(),
        "compliance_prompts": get_compliance_analytics()
    }
    
    if args.json:
        print(json.dumps(analytics_data, indent=2))
    else:
        print("Sprint 4 Domain Logic Analytics")
        print("=" * 40)
        for service, data in analytics_data.items():
            print(f"\n{service.replace('_', ' ').title()}:")
            for key, value in data.items():
                if key != "service_name":
                    print(f"  {key.replace('_', ' ').title()}: {value}")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for Sprint 4 CLI."""
    parser = argparse.ArgumentParser(
        description="Sprint 4 Domain Logic CLI - Loan calculations, payment plans, compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Loan calculator
    loan_parser = subparsers.add_parser("loan-calc", help="Calculate loan payments")
    loan_parser.add_argument("--amount", type=float, required=True, help="Loan amount")
    loan_parser.add_argument("--interest", type=float, required=True, help="Annual interest rate (decimal)")
    loan_parser.add_argument("--months", type=int, required=True, help="Loan term in months")
    
    # Payment plan
    plan_parser = subparsers.add_parser("payment-plan", help="Generate payment plans")
    plan_parser.add_argument("--amount", type=float, required=True, help="Loan amount")
    plan_parser.add_argument("--interest", type=float, help="Annual interest rate (decimal)")
    plan_parser.add_argument("--months", type=int, help="Loan term in months")
    plan_parser.add_argument("--extra", type=float, default=0.0, help="Extra monthly payment")
    plan_parser.add_argument("--scenarios", action="store_true", help="Generate multiple scenarios")
    plan_parser.add_argument("--rates", type=str, help="Comma-separated interest rates for scenarios")
    plan_parser.add_argument("--terms", type=str, help="Comma-separated terms for scenarios")
    
    # Compliance testing
    comp_parser = subparsers.add_parser("compliance-test", help="Test compliance prompts")
    comp_parser.add_argument("--type", choices=["KYC", "opt-out", "legal", "privacy", "recording", "data-retention"], help="Prompt type")
    comp_parser.add_argument("--flow", choices=["onboarding", "authentication", "transaction", "support"], help="Flow type")
    comp_parser.add_argument("--stage", type=str, help="Flow stage")
    comp_parser.add_argument("--parameterized", action="store_true", help="Use parameterized prompt")
    comp_parser.add_argument("--params", type=str, help="Parameters as key=value,key=value")
    comp_parser.add_argument("--validate", action="store_true", help="Validate compliance requirements")
    comp_parser.add_argument("--conversation-type", type=str, help="Type of conversation for validation")
    comp_parser.add_argument("--verified", action="store_true", help="Customer is verified")
    comp_parser.add_argument("--consent", action="store_true", help="Recording consent given")
    
    # Analytics
    analytics_parser = subparsers.add_parser("analytics", help="Show domain logic analytics")
    
    return parser


async def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "loan-calc":
            await loan_calc_command(args)
        elif args.command == "payment-plan":
            await payment_plan_command(args)
        elif args.command == "compliance-test":
            compliance_test_command(args)
        elif args.command == "analytics":
            analytics_command(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
