"""
Sprint 4 Integration Demo

Demonstrates all Sprint 4 domain logic functionality working together.
"""

import asyncio
import json
from voicebot_orchestrator.loan import calculate_payment_with_analytics
from voicebot_orchestrator.payment_plan import generate_payment_plan_with_analytics
from voicebot_orchestrator.compliance import get_prompt_with_analytics, get_flow_prompt


async def demo_loan_calculation():
    """Demo loan calculation functionality."""
    print("🏦 LOAN CALCULATION DEMO")
    print("=" * 40)
    
    # Calculate a $50,000 mortgage at 4.5% for 30 years
    monthly, schedule = await calculate_payment_with_analytics(50000, 0.045, 360)
    
    print(f"Loan Amount: $50,000")
    print(f"Interest Rate: 4.5%")
    print(f"Term: 30 years (360 months)")
    print(f"Monthly Payment: ${monthly:,.2f}")
    
    total_paid = monthly * 360
    total_interest = total_paid - 50000
    print(f"Total Paid: ${total_paid:,.2f}")
    print(f"Total Interest: ${total_interest:,.2f}")
    print()


async def demo_payment_plan():
    """Demo payment plan with extra payments."""
    print("💰 PAYMENT PLAN DEMO")
    print("=" * 40)
    
    # Same loan with $200 extra monthly payment
    plan = await generate_payment_plan_with_analytics(50000, 0.045, 360, 200)
    
    print(f"Base Monthly Payment: ${plan['monthly'] - 200:,.2f}")
    print(f"Extra Payment: $200.00")
    print(f"Total Monthly Payment: ${plan['monthly']:,.2f}")
    print(f"Total Paid: ${plan['total_paid']:,.2f}")
    print(f"Total Interest: ${plan['total_interest']:,.2f}")
    print(f"Months Saved: {plan['months_saved']}")
    print(f"Years Saved: {plan['months_saved'] / 12:.1f}")
    print()


def demo_compliance():
    """Demo compliance prompts."""
    print("⚖️  COMPLIANCE DEMO")
    print("=" * 40)
    
    # KYC prompt
    kyc_prompt = get_prompt_with_analytics("KYC")
    print("KYC Prompt:")
    print(f"  {kyc_prompt}")
    print()
    
    # Flow-based prompt
    auth_prompt = get_flow_prompt("authentication", "start")
    print("Authentication Flow:")
    print(f"  {auth_prompt}")
    print()
    
    # Transaction flow with parameters
    transaction_prompt = get_flow_prompt("transaction", "confirmation", {
        "transaction_type": "wire transfer",
        "amount": 5000,
        "recipient": "John Smith"
    })
    print("Transaction Confirmation:")
    print(f"  {transaction_prompt}")
    print()


async def demo_banking_scenario():
    """Demo complete banking scenario."""
    print("🎯 COMPLETE BANKING SCENARIO")
    print("=" * 50)
    print("Customer calls about refinancing their mortgage...")
    print()
    
    # Step 1: Compliance check
    print("Step 1: Initial Compliance")
    initial_prompt = get_flow_prompt("onboarding", "welcome")
    print(f"Agent: {initial_prompt}")
    print()
    
    # Step 2: Current loan analysis
    print("Step 2: Current Loan Analysis")
    print("Customer's current loan: $250,000 at 6.5% with 25 years remaining")
    current_monthly, _ = await calculate_payment_with_analytics(250000, 0.065, 300)
    print(f"Current monthly payment: ${current_monthly:,.2f}")
    print()
    
    # Step 3: Refinancing options
    print("Step 3: Refinancing Options")
    print("New loan: $250,000 at 4.0% for 30 years")
    refi_plan = await generate_payment_plan_with_analytics(250000, 0.04, 360)
    print(f"New monthly payment: ${refi_plan['monthly']:,.2f}")
    
    savings = current_monthly - refi_plan['monthly']
    print(f"Monthly savings: ${savings:,.2f}")
    print(f"Annual savings: ${savings * 12:,.2f}")
    print()
    
    # Step 4: With extra payments
    print("Step 4: Accelerated Payoff Option")
    print(f"If customer pays extra ${savings:,.0f}/month toward principal:")
    accelerated = await generate_payment_plan_with_analytics(250000, 0.04, 360, savings)
    print(f"Payoff time: {accelerated['actual_months']} months ({accelerated['actual_months']/12:.1f} years)")
    print(f"Interest saved: ${refi_plan['total_interest'] - accelerated['total_interest']:,.2f}")
    print()
    
    # Step 5: Final compliance
    print("Step 5: Final Disclosure")
    legal_prompt = get_prompt_with_analytics("legal")
    print(f"Agent: {legal_prompt}")


async def main():
    """Run all Sprint 4 demos."""
    print("🚀 Sprint 4 Domain Logic Integration Demo")
    print("=" * 60)
    print()
    
    await demo_loan_calculation()
    await demo_payment_plan()
    demo_compliance()
    await demo_banking_scenario()
    
    print("✅ Sprint 4 Integration Demo Complete!")
    print("All domain logic components working together successfully.")


if __name__ == "__main__":
    asyncio.run(main())
