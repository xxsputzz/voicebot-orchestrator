"""
Sprint 4: Compliance Prompts

This module provides parameterized compliance prompts for banking voicebot systems.
Includes KYC, opt-out, legal notifications with fallback messages.
"""

from __future__ import annotations
from typing import Dict, Union, Literal, Optional, List


def get_prompt(
    prompt_type: Literal["KYC", "opt-out", "legal", "privacy", "recording", "data-retention"]
) -> str:
    """
    Return compliance prompt text for banking voicebot.
    
    Args:
        prompt_type: type of compliance prompt needed
        
    Returns:
        formatted string prompt
        
    Raises:
        ValueError: if prompt_type is not recognized
    """
    texts: Dict[str, str] = {
        "KYC": "This call will be recorded for compliance and quality purposes. "
               "Please confirm you are the account holder and provide your date of birth "
               "and the last four digits of your Social Security number for verification.",
        
        "opt-out": "You may opt out of call recording at any time by saying 'I opt out'. "
                   "Please note that opting out may limit our ability to assist you with "
                   "certain account services and transactions.",
        
        "legal": "All communications are subject to legal monitoring and retention policies. "
                 "Information shared during this call may be used for regulatory compliance, "
                 "fraud prevention, and quality assurance purposes.",
        
        "privacy": "Your privacy is important to us. We collect and use your personal information "
                   "in accordance with our Privacy Policy. You have the right to know what "
                   "information we collect and how it is used.",
        
        "recording": "For quality assurance and training purposes, this call may be monitored "
                     "or recorded. If you do not consent to recording, please inform us now.",
        
        "data-retention": "Your call data and transaction information will be retained according "
                          "to federal banking regulations and our data retention policy. "
                          "Most records are kept for a minimum of seven years."
    }
    
    if prompt_type not in texts:
        raise ValueError(f"Unknown prompt_type: {prompt_type}. "
                        f"Valid types: {', '.join(texts.keys())}")
    
    return texts[prompt_type]


def get_parameterized_prompt(
    prompt_type: str, 
    params: Optional[Dict[str, Union[str, int, float]]] = None
) -> str:
    """
    Get compliance prompt with parameter substitution.
    
    Args:
        prompt_type: type of compliance prompt
        params: dictionary of parameters for template substitution
        
    Returns:
        formatted prompt with parameters substituted
        
    Raises:
        ValueError: if prompt_type invalid or required parameters missing
    """
    if params is None:
        params = {}
    
    templates: Dict[str, str] = {
        "account-verification": (
            "Hello {customer_name}, I need to verify your identity before we proceed. "
            "Please provide your date of birth and the last four digits of your "
            "Social Security number. This information is required for account security."
        ),
        
        "transaction-limit": (
            "For security purposes, {transaction_type} transactions are limited to "
            "${max_amount:,.2f} per {time_period}. Your current transaction amount "
            "of ${amount:,.2f} is within this limit."
        ),
        
        "fraud-alert": (
            "We have detected unusual activity on your account ending in {account_suffix}. "
            "As a security measure, we have temporarily {action_taken}. "
            "Please verify recent transactions totaling ${amount:,.2f}."
        ),
        
        "opt-out-confirmation": (
            "You have chosen to opt out of call recording. Please note that this may "
            "limit our ability to assist with {restricted_services}. "
            "You can change this preference at any time."
        ),
        
        "regulatory-disclosure": (
            "This {product_type} is subject to {regulation_name} regulations. "
            "Interest rates, fees, and terms may vary based on your creditworthiness. "
            "Annual percentage rate (APR) ranges from {min_apr:.2f}% to {max_apr:.2f}%."
        )
    }
    
    if prompt_type not in templates:
        # Fall back to standard prompts
        if prompt_type in ["KYC", "opt-out", "legal", "privacy", "recording", "data-retention"]:
            return get_prompt(prompt_type)  # type: ignore
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")
    
    template = templates[prompt_type]
    
    try:
        return template.format(**params)
    except KeyError as e:
        missing_param = str(e).strip("'")
        raise ValueError(f"Missing required parameter: {missing_param}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Parameter formatting error: {e}")


def get_flow_prompt(
    flow_type: Literal["onboarding", "authentication", "transaction", "support"],
    stage: str,
    params: Optional[Dict[str, Union[str, int, float]]] = None
) -> str:
    """
    Get compliance prompt for specific conversation flow and stage.
    
    Args:
        flow_type: type of conversation flow
        stage: current stage in the flow
        params: optional parameters for customization
        
    Returns:
        appropriate compliance prompt for the flow stage
    """
    if params is None:
        params = {}
    
    flow_prompts: Dict[str, Dict[str, str]] = {
        "onboarding": {
            "welcome": "Welcome to our secure banking system. To protect your account, "
                      "we need to verify your identity before proceeding.",
            "identity": "Please provide your full legal name as it appears on your account, "
                       "date of birth, and Social Security number.",
            "recording": "This onboarding session will be recorded for compliance purposes. "
                        "Do you consent to this recording?",
            "complete": "Thank you for completing identity verification. Your account "
                       "setup will be finalized within 24 hours."
        },
        
        "authentication": {
            "start": "For your security, I need to verify your identity before accessing "
                    "account information.",
            "challenge": "Please provide {auth_method} for additional verification.",
            "success": "Identity verified successfully. How may I assist you today?",
            "failure": "I was unable to verify your identity. For security reasons, "
                      "you will need to visit a branch or call our secure line."
        },
        
        "transaction": {
            "pre-auth": "Before processing this {transaction_type}, I need to confirm "
                       "the transaction details and verify your authorization.",
            "limits": "This transaction is subject to daily limits and fraud monitoring.",
            "confirmation": "Please confirm you authorize this {transaction_type} "
                           "of ${amount:,.2f} to {recipient}.",
            "complete": "Transaction completed successfully. Reference number: {ref_number}"
        },
        
        "support": {
            "start": "I'm here to help with your banking needs. This call may be "
                    "recorded for quality and training purposes.",
            "escalation": "I'm transferring you to a specialist who can better assist "
                         "with your {issue_type} inquiry.",
            "resolution": "Is there anything else I can help you with today?",
            "feedback": "We value your feedback. You may receive a brief survey "
                       "about your experience."
        }
    }
    
    if flow_type not in flow_prompts:
        raise ValueError(f"Unknown flow_type: {flow_type}")
    
    if stage not in flow_prompts[flow_type]:
        available_stages = ", ".join(flow_prompts[flow_type].keys())
        raise ValueError(f"Unknown stage '{stage}' for flow '{flow_type}'. "
                        f"Available stages: {available_stages}")
    
    template = flow_prompts[flow_type][stage]
    
    try:
        return template.format(**params)
    except KeyError as e:
        # Return template without substitution if optional parameters missing
        return template


def validate_compliance_requirements(
    conversation_type: str,
    customer_verified: bool = False,
    recording_consent: bool = False
) -> Dict[str, Union[bool, List[str]]]:
    """
    Validate compliance requirements for a conversation type.
    
    Args:
        conversation_type: type of customer interaction
        customer_verified: whether customer identity is verified
        recording_consent: whether customer consented to recording
        
    Returns:
        dict with compliance status and required actions
    """
    requirements = {
        "account_inquiry": {
            "identity_required": True,
            "recording_required": False,
            "privacy_notice": True
        },
        "transaction": {
            "identity_required": True,
            "recording_required": True,
            "privacy_notice": True
        },
        "general_info": {
            "identity_required": False,
            "recording_required": False,
            "privacy_notice": False
        },
        "complaint": {
            "identity_required": True,
            "recording_required": True,
            "privacy_notice": True
        }
    }
    
    if conversation_type not in requirements:
        return {
            "compliant": False,
            "errors": [f"Unknown conversation type: {conversation_type}"],
            "required_actions": []
        }
    
    reqs = requirements[conversation_type]
    errors = []
    required_actions = []
    
    if reqs["identity_required"] and not customer_verified:
        errors.append("Customer identity verification required")
        required_actions.append("verify_identity")
    
    if reqs["recording_required"] and not recording_consent:
        errors.append("Recording consent required")
        required_actions.append("obtain_recording_consent")
    
    if reqs["privacy_notice"]:
        required_actions.append("provide_privacy_notice")
    
    return {
        "compliant": len(errors) == 0,
        "errors": errors,
        "required_actions": required_actions
    }


# Function calling schema for LLM integration
COMPLIANCE_SCHEMA = {
    "name": "get_compliance_prompt",
    "description": "Get appropriate compliance prompt for banking interactions",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt_type": {
                "type": "string",
                "enum": ["KYC", "opt-out", "legal", "privacy", "recording", "data-retention"],
                "description": "Type of compliance prompt needed"
            },
            "flow_type": {
                "type": "string", 
                "enum": ["onboarding", "authentication", "transaction", "support"],
                "description": "Conversation flow type"
            },
            "stage": {
                "type": "string",
                "description": "Current stage in the conversation flow"
            }
        },
        "required": ["prompt_type"]
    }
}


# Analytics counter
_compliance_prompt_counter = 0


def get_compliance_analytics() -> Dict[str, Union[int, str]]:
    """Get analytics data for compliance prompt usage."""
    return {
        "total_prompts_served": _compliance_prompt_counter,
        "service_name": "compliance_prompts"
    }


def _increment_compliance_counter() -> None:
    """Increment compliance prompt counter for analytics."""
    global _compliance_prompt_counter
    _compliance_prompt_counter += 1


def get_prompt_with_analytics(prompt_type: str) -> str:
    """Get compliance prompt with analytics tracking."""
    _increment_compliance_counter()
    return get_prompt(prompt_type)  # type: ignore
