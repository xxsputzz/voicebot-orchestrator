"""
Sprint 4: Payment Plan Generator

This module generates payment plans with multiple rate scenarios and extra payments.
Supports step-down rate plans and total interest calculations for banking voicebots.
"""

from __future__ import annotations
import asyncio
import math
from typing import List, Dict, Union, Optional
from .loan import calculate_payment, calculate_total_interest


async def generate_payment_plan(
    principal: float,
    rate: float,
    months: int,
    extra: float = 0.0
) -> Dict[str, Union[float, List[Dict[str, float]], int]]:
    """
    Generate a payment plan with optional extra monthly contribution.
    
    Args:
        principal: starting balance
        rate: annual rate as decimal
        months: planned months
        extra: extra payment toward principal each month
        
    Returns:
        dict with 'monthly', 'schedule', 'total_paid', and 'total_interest'
        
    Raises:
        ValueError: if any input is invalid
    """
    # Input validation
    if not isinstance(principal, (int, float)) or principal <= 0:
        raise ValueError("principal must be a positive number")
    
    if not isinstance(rate, (int, float)) or not (0 <= rate < 1):
        raise ValueError("rate must be a decimal between 0 and 1")
    
    if not isinstance(months, int) or months <= 0:
        raise ValueError("months must be a positive integer")
    
    if not isinstance(extra, (int, float)) or extra < 0:
        raise ValueError("extra payment must be non-negative")
    
    # Get base payment calculation
    monthly_payment, base_schedule = await calculate_payment(principal, rate, months)
    
    # If no extra payment, return base calculation
    if extra == 0:
        total_interest = calculate_total_interest(base_schedule)
        total_paid = monthly_payment * months
        
        return {
            "monthly": monthly_payment,
            "schedule": base_schedule,
            "total_paid": round(total_paid, 2),
            "total_interest": total_interest,
            "months_saved": 0
        }
    
    # Calculate with extra payments
    schedule: List[Dict[str, float]] = []
    balance = principal
    monthly_rate = rate / 12
    total_paid = 0.0
    month = 0
    
    while balance > 0.01 and month < months:  # 1 cent threshold
        month += 1
        interest = balance * monthly_rate
        
        # Calculate principal payment (base + extra, but not more than balance)
        total_payment = monthly_payment + extra
        principal_payment = min(total_payment - interest, balance)
        actual_payment = interest + principal_payment
        
        balance -= principal_payment
        total_paid += actual_payment
        
        schedule.append({
            "month": month,
            "interest": round(interest, 2),
            "principal": round(principal_payment, 2),
            "balance": round(max(balance, 0.0), 2),
            "payment": round(actual_payment, 2)
        })
        
        if balance <= 0.01:
            break
    
    total_interest = sum(entry["interest"] for entry in schedule)
    months_saved = months - month
    
    return {
        "monthly": round(monthly_payment + extra, 2),
        "schedule": schedule,
        "total_paid": round(total_paid, 2),
        "total_interest": round(total_interest, 2),
        "months_saved": months_saved,
        "actual_months": month
    }


async def generate_multiple_scenarios(
    principal: float,
    rates: List[float],
    terms: List[int],
    extra_payments: Optional[List[float]] = None
) -> List[Dict[str, Union[float, int, str]]]:
    """
    Generate multiple payment scenarios for comparison.
    
    Args:
        principal: loan amount
        rates: list of annual interest rates as decimals
        terms: list of loan terms in months
        extra_payments: optional list of extra monthly payments
        
    Returns:
        list of scenario dictionaries with comparison data
    """
    if not rates:
        raise ValueError("rates list cannot be empty")
    
    if not terms:
        raise ValueError("terms list cannot be empty")
    
    if extra_payments is None:
        extra_payments = [0.0]
    
    scenarios = []
    scenario_id = 1
    
    for rate in rates:
        for term in terms:
            for extra in extra_payments:
                try:
                    plan = await generate_payment_plan(principal, rate, term, extra)
                    
                    scenarios.append({
                        "scenario_id": scenario_id,
                        "rate_percent": round(rate * 100, 2),
                        "term_months": term,
                        "extra_payment": extra,
                        "monthly_payment": plan["monthly"],
                        "total_paid": plan["total_paid"],
                        "total_interest": plan["total_interest"],
                        "months_saved": plan.get("months_saved", 0)
                    })
                    
                    scenario_id += 1
                    
                except ValueError as e:
                    # Skip invalid scenarios
                    continue
    
    return scenarios


async def generate_step_down_plan(
    principal: float,
    initial_rate: float,
    rate_reductions: List[Dict[str, Union[float, int]]],
    months: int
) -> Dict[str, Union[float, List[Dict[str, float]]]]:
    """
    Generate a step-down rate payment plan.
    
    Args:
        principal: starting loan amount
        initial_rate: starting annual interest rate as decimal
        rate_reductions: list of {"month": int, "new_rate": float} dictionaries
        months: total loan term
        
    Returns:
        dict with payment plan including rate changes
    """
    # Input validation
    if not isinstance(principal, (int, float)) or principal <= 0:
        raise ValueError("principal must be positive")
    
    if not isinstance(initial_rate, (int, float)) or not (0 <= initial_rate < 1):
        raise ValueError("initial_rate must be between 0 and 1")
    
    if not isinstance(months, int) or months <= 0:
        raise ValueError("months must be positive")
    
    # Sort rate reductions by month
    rate_changes = sorted(rate_reductions, key=lambda x: x["month"])
    
    # Validate rate changes
    for change in rate_changes:
        if change["month"] <= 0 or change["month"] > months:
            raise ValueError(f"Rate change month {change['month']} out of range")
        
        if not (0 <= change["new_rate"] < 1):
            raise ValueError(f"New rate {change['new_rate']} out of valid range")
    
    schedule: List[Dict[str, float]] = []
    balance = principal
    current_rate = initial_rate
    total_paid = 0.0
    
    # Create rate schedule
    rate_schedule = {}
    for change in rate_changes:
        rate_schedule[change["month"]] = change["new_rate"]
    
    for month in range(1, months + 1):
        # Check for rate change
        if month in rate_schedule:
            current_rate = rate_schedule[month]
        
        monthly_rate = current_rate / 12
        
        # Calculate payment for remaining balance and term
        remaining_months = months - month + 1
        if current_rate == 0:
            monthly_payment = balance / remaining_months
        else:
            monthly_payment = (
                balance * monthly_rate / 
                (1 - math.pow(1 + monthly_rate, -remaining_months))
            )
        
        interest = balance * monthly_rate
        principal_payment = monthly_payment - interest
        balance -= principal_payment
        total_paid += monthly_payment
        
        schedule.append({
            "month": month,
            "interest": round(interest, 2),
            "principal": round(principal_payment, 2),
            "balance": round(max(balance, 0.0), 2),
            "payment": round(monthly_payment, 2),
            "rate": round(current_rate * 100, 3)
        })
        
        if balance <= 0.01:
            break
    
    total_interest = sum(entry["interest"] for entry in schedule)
    
    return {
        "schedule": schedule,
        "total_paid": round(total_paid, 2),
        "total_interest": round(total_interest, 2),
        "final_month": len(schedule)
    }


# Function calling schema for LLM integration
PAYMENT_PLAN_SCHEMA = {
    "name": "generate_payment_plan",
    "description": "Generate payment plan with optional extra payments",
    "parameters": {
        "type": "object",
        "properties": {
            "principal": {
                "type": "number",
                "description": "Loan principal amount",
                "minimum": 0.01
            },
            "rate": {
                "type": "number",
                "description": "Annual interest rate as decimal",
                "minimum": 0,
                "maximum": 0.99
            },
            "months": {
                "type": "integer", 
                "description": "Loan term in months",
                "minimum": 1
            },
            "extra": {
                "type": "number",
                "description": "Extra monthly payment toward principal",
                "minimum": 0,
                "default": 0
            }
        },
        "required": ["principal", "rate", "months"]
    }
}


# Analytics counter
_payment_plan_counter = 0


def get_payment_plan_analytics() -> Dict[str, Union[int, str]]:
    """Get analytics data for payment plan generation."""
    return {
        "total_plans_generated": _payment_plan_counter,
        "service_name": "payment_plan_generator"
    }


def _increment_plan_counter() -> None:
    """Increment payment plan counter for analytics."""
    global _payment_plan_counter
    _payment_plan_counter += 1


# Wrapper with analytics
async def generate_payment_plan_with_analytics(
    principal: float,
    rate: float,
    months: int,
    extra: float = 0.0
) -> Dict[str, Union[float, List[Dict[str, float]], int]]:
    """Generate payment plan with analytics tracking."""
    _increment_plan_counter()
    return await generate_payment_plan(principal, rate, months, extra)
