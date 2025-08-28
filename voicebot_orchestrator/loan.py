"""
Sprint 4: Loan Balancing Calculator

This module provides async loan calculation functionality for enterprise voicebot systems.
Implements amortization schedules with proper validation and edge case handling.
"""

from __future__ import annotations
import math
from typing import Tuple, List, Dict, Union


async def calculate_payment(
    principal: float, 
    annual_rate: float, 
    months: int
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Calculate monthly payment and full amortization breakdown.
    
    Args:
        principal: loan principal amount (>0)
        annual_rate: annual interest rate as decimal (0 <= rate < 1)
        months: total number of monthly payments (>0)
        
    Returns:
        tuple of (monthly_payment, amortization_schedule)
        
    Raises:
        ValueError: if any input is invalid
    """
    # Input validation
    if not isinstance(principal, (int, float)) or principal <= 0:
        raise ValueError("principal must be a positive number")
    
    if not isinstance(annual_rate, (int, float)) or not (0 <= annual_rate < 1):
        raise ValueError("annual_rate must be a decimal between 0 and 1")
    
    if not isinstance(months, int) or months <= 0:
        raise ValueError("months must be a positive integer")
    
    # Handle zero interest rate case
    if annual_rate == 0:
        monthly_payment = principal / months
        schedule: List[Dict[str, float]] = []
        balance = principal
        
        for m in range(1, months + 1):
            principal_paid = monthly_payment
            balance -= principal_paid
            schedule.append({
                "month": m,
                "interest": 0.0,
                "principal": round(principal_paid, 2),
                "balance": round(max(balance, 0.0), 2)
            })
        
        return round(monthly_payment, 2), schedule
    
    # Standard amortization calculation
    monthly_rate = annual_rate / 12
    
    # Calculate monthly payment using amortization formula
    try:
        monthly_payment = (
            principal * monthly_rate / 
            (1 - math.pow(1 + monthly_rate, -months))
        )
    except (OverflowError, ZeroDivisionError):
        raise ValueError("Invalid calculation parameters - rate or term too large")
    
    # Generate amortization schedule
    schedule: List[Dict[str, float]] = []
    balance = principal
    
    for m in range(1, months + 1):
        interest = balance * monthly_rate
        principal_paid = monthly_payment - interest
        balance -= principal_paid
        
        schedule.append({
            "month": m,
            "interest": round(interest, 2),
            "principal": round(principal_paid, 2),
            "balance": round(max(balance, 0.0), 2)
        })
    
    return round(monthly_payment, 2), schedule


def calculate_total_interest(schedule: List[Dict[str, float]]) -> float:
    """
    Calculate total interest from an amortization schedule.
    
    Args:
        schedule: amortization schedule from calculate_payment
        
    Returns:
        total interest amount
    """
    return round(sum(entry["interest"] for entry in schedule), 2)


def calculate_remaining_balance(
    principal: float,
    annual_rate: float, 
    months: int,
    payments_made: int
) -> float:
    """
    Calculate remaining balance after specified number of payments.
    
    Args:
        principal: original loan amount
        annual_rate: annual interest rate as decimal
        months: total loan term in months
        payments_made: number of payments already made
        
    Returns:
        remaining balance
        
    Raises:
        ValueError: if payments_made exceeds months or other invalid inputs
    """
    if payments_made < 0:
        raise ValueError("payments_made cannot be negative")
    
    if payments_made >= months:
        return 0.0
    
    # Calculate using standard remaining balance formula
    if annual_rate == 0:
        monthly_payment = principal / months
        return principal - (monthly_payment * payments_made)
    
    monthly_rate = annual_rate / 12
    monthly_payment = (
        principal * monthly_rate / 
        (1 - math.pow(1 + monthly_rate, -months))
    )
    
    remaining_balance = (
        principal * math.pow(1 + monthly_rate, payments_made) -
        monthly_payment * (math.pow(1 + monthly_rate, payments_made) - 1) / monthly_rate
    )
    
    return round(max(remaining_balance, 0.0), 2)


# Function calling schema for LLM integration
LOAN_CALCULATOR_SCHEMA = {
    "name": "calculate_payment",
    "description": "Calculate monthly loan payment and amortization schedule",
    "parameters": {
        "type": "object",
        "properties": {
            "principal": {
                "type": "number",
                "description": "Loan principal amount in dollars",
                "minimum": 0.01
            },
            "annual_rate": {
                "type": "number", 
                "description": "Annual interest rate as decimal (e.g., 0.06 for 6%)",
                "minimum": 0,
                "maximum": 0.99
            },
            "months": {
                "type": "integer",
                "description": "Loan term in months",
                "minimum": 1
            }
        },
        "required": ["principal", "annual_rate", "months"]
    }
}


# Analytics counter for orchestrator integration
_loan_calculation_counter = 0


def get_loan_analytics() -> Dict[str, Union[int, float]]:
    """
    Get analytics data for loan calculations.
    
    Returns:
        dictionary with calculation statistics
    """
    return {
        "total_calculations": _loan_calculation_counter,
        "service_name": "loan_calculator"
    }


def _increment_counter() -> None:
    """Increment the loan calculation counter for analytics."""
    global _loan_calculation_counter
    _loan_calculation_counter += 1


# Wrapper for analytics integration
async def calculate_payment_with_analytics(
    principal: float,
    annual_rate: float, 
    months: int
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Calculate payment with analytics tracking.
    
    Same as calculate_payment but increments usage counter.
    """
    _increment_counter()
    return await calculate_payment(principal, annual_rate, months)
