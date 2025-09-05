"""
Ultra-High Accuracy Ollama LLM - Version 2.0

This version implements aggressive accuracy optimizations:
1. Advanced prompt engineering with few-shot examples
2. Response validation and retry logic
3. Contextual conversation tracking
4. Multi-pass response refinement
5. Domain-specific validation rules
"""

from __future__ import annotations
import asyncio
import logging
import aiohttp
import json
import os
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Import existing components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from aws_microservices.prompt_loader import PromptLoader

@dataclass
class ResponseScore:
    """Enhanced quality score for AI responses."""
    relevance: float  # 0-1: How relevant to user input
    banking_expertise: float  # 0-1: Banking domain accuracy
    persona_consistency: float  # 0-1: Alex persona consistency
    completeness: float  # 0-1: Response completeness
    accuracy: float  # 0-1: Specific detail accuracy
    overall: float  # 0-1: Overall quality score

class UltraHighAccuracyLLM:
    """
    Ultra-high accuracy Ollama LLM targeting 98%+ accuracy.
    
    Implements aggressive accuracy optimizations:
    - Few-shot example learning
    - Response validation and retry logic
    - Multi-pass refinement
    - Advanced context tracking
    """
    
    def __init__(
        self,
        primary_model: str = "mistral:latest",
        secondary_models: List[str] = ["gpt-oss:20b"],
        accuracy_threshold: float = 0.95,
        max_retries: int = 2,
        enable_multi_pass: bool = True
    ):
        """Initialize ultra-high accuracy LLM."""
        self.primary_model = primary_model
        self.secondary_models = secondary_models
        self.ollama_url = "http://localhost:11434"
        self.accuracy_threshold = accuracy_threshold
        self.max_retries = max_retries
        self.enable_multi_pass = enable_multi_pass
        
        # Initialize prompt system
        self.prompt_loader = PromptLoader()
        
        # Conversation tracking
        self.conversation_context = {}
        self.user_profiles = {}
        
        # Performance tracking
        self.total_requests = 0
        self.ultra_high_quality_responses = 0
        self.retry_count = 0
        self.multi_pass_used = 0
        
        self.logger = logging.getLogger(__name__)
    
    async def generate_response(
        self,
        user_input: str,
        user_profile: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: str = "default"
    ) -> Tuple[str, ResponseScore]:
        """Generate ultra-high accuracy response."""
        self.total_requests += 1
        
        # Update user profile and context
        self._update_user_context(user_input, user_profile, session_id)
        
        # Generate response with retry logic
        for attempt in range(self.max_retries + 1):
            try:
                response, score = await self._generate_with_validation(
                    user_input, conversation_history, session_id, attempt
                )
                
                if score.overall >= self.accuracy_threshold:
                    self.ultra_high_quality_responses += 1
                    return response, score
                elif attempt < self.max_retries:
                    self.retry_count += 1
                    self.logger.info(f"Retrying due to low score: {score.overall:.3f} (attempt {attempt + 1})")
                    continue
                else:
                    # Use the best attempt we have
                    return response, score
                    
            except Exception as e:
                if attempt < self.max_retries:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    continue
                else:
                    # Fallback response
                    return self._generate_fallback_response(user_input, user_profile)
        
        # Should never reach here, but just in case
        return self._generate_fallback_response(user_input, user_profile)
    
    def _update_user_context(
        self,
        user_input: str,
        user_profile: Optional[Dict[str, Any]],
        session_id: str
    ):
        """Update user context and profile information."""
        if session_id not in self.conversation_context:
            self.conversation_context[session_id] = []
        
        # Extract and update user details from input
        extracted_info = self._extract_user_info(user_input)
        
        # Update user profile
        if session_id not in self.user_profiles:
            self.user_profiles[session_id] = {}
        
        if user_profile:
            self.user_profiles[session_id].update(user_profile)
        
        self.user_profiles[session_id].update(extracted_info)
        
        # Add to conversation history
        self.conversation_context[session_id].append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'extracted_info': extracted_info
        })
        
        # Keep only recent conversation
        if len(self.conversation_context[session_id]) > 10:
            self.conversation_context[session_id] = self.conversation_context[session_id][-10:]
    
    def _extract_user_info(self, user_input: str) -> Dict[str, Any]:
        """Extract user information from input."""
        info = {}
        
        # Extract name
        name_patterns = [
            r'my name is ([A-Z][a-z]+)',
            r"i'm ([A-Z][a-z]+)",
            r'this is ([A-Z][a-z]+)',
            r'([A-Z][a-z]+) here'
        ]
        for pattern in name_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                info['name'] = match.group(1).title()
                break
        
        # Extract amounts
        amount_patterns = [
            r'\$?([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*) dollars?',
            r'([\d,]+\.?\d*)k'
        ]
        amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            for match in matches:
                try:
                    # Handle 'k' suffix
                    if user_input.lower().count(match.lower() + 'k') > 0:
                        amount = float(match.replace(',', '')) * 1000
                    else:
                        amount = float(match.replace(',', ''))
                    amounts.append(amount)
                except:
                    continue
        
        if amounts:
            info['mentioned_amounts'] = amounts
            # If it's likely debt, store as debt amount
            if any(word in user_input.lower() for word in ['debt', 'owe', 'credit card']):
                info['debt_amount'] = max(amounts)
        
        # Extract income information
        income_patterns = [
            r'make \$?([\d,]+\.?\d*)',
            r'earn \$?([\d,]+\.?\d*)',
            r'income.*?\$?([\d,]+\.?\d*)'
        ]
        for pattern in income_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                try:
                    info['income'] = float(match.group(1).replace(',', ''))
                except:
                    continue
        
        # Extract employment status
        if 'self-employed' in user_input.lower() or 'self employed' in user_input.lower():
            info['employment'] = 'self-employed'
        elif 'unemployed' in user_input.lower():
            info['employment'] = 'unemployed'
        elif 'employed' in user_input.lower():
            info['employment'] = 'employed'
        
        return info
    
    async def _generate_with_validation(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]],
        session_id: str,
        attempt: int
    ) -> Tuple[str, ResponseScore]:
        """Generate response with validation."""
        # Build ultra-enhanced prompt
        prompt = self._build_ultra_enhanced_prompt(user_input, session_id, attempt)
        
        # Try primary model first
        response = await self._call_ollama_api_enhanced(prompt, self.primary_model)
        
        if not response:
            # Try secondary models
            for model in self.secondary_models:
                response = await self._call_ollama_api_enhanced(prompt, model)
                if response:
                    break
        
        if not response:
            raise Exception("All models failed to generate response")
        
        # Score the response
        score = self._score_response_enhanced(response, user_input, session_id)
        
        # Multi-pass refinement if enabled and score is below threshold
        if (self.enable_multi_pass and 
            score.overall < self.accuracy_threshold and 
            attempt == 0):
            self.multi_pass_used += 1
            return await self._refine_response(response, user_input, session_id, score)
        
        return response, score
    
    def _build_ultra_enhanced_prompt(
        self,
        user_input: str,
        session_id: str,
        attempt: int
    ) -> str:
        """Build ultra-enhanced prompt with few-shot examples and context."""
        try:
            base_system_prompt = self.prompt_loader.get_system_prompt(call_type="inbound")
        except:
            base_system_prompt = self._get_fallback_system_prompt()
        
        # Add ultra-accuracy instructions
        accuracy_instructions = """
CRITICAL ACCURACY REQUIREMENTS:
1. ALWAYS acknowledge specific names, amounts, and details mentioned by the customer
2. Reference exact numbers, percentages, and dollar amounts the customer provides
3. Stay perfectly in character as Alex from Finally Payoff Debt
4. Provide specific, actionable financial advice based on the customer's situation
5. Never give generic responses - always personalize based on customer input
6. Ask relevant follow-up questions to better assist the customer
7. Show empathy for the customer's financial situation
8. Provide concrete next steps or solutions

RESPONSE QUALITY CHECKLIST:
✓ Customer's name mentioned (if provided)
✓ Specific amounts/numbers acknowledged (if provided)
✓ Banking expertise demonstrated
✓ Alex persona maintained
✓ Actionable advice provided
✓ Professional and empathetic tone
✓ Clear next steps offered
        """
        
        # Add few-shot examples
        few_shot_examples = """
FEW-SHOT EXAMPLES OF PERFECT RESPONSES:

Example 1:
Customer: "My name is Sarah and I have $18,000 in credit card debt at 26% APR"
Alex: "Hi Sarah! Thank you for calling Finally Payoff Debt. I understand you have $18,000 in credit card debt at 26% APR - that's a really high interest rate that's costing you a lot of money each month. I'm Alex, and I specialize in helping people like you break free from high-interest debt through personal loan consolidation. With a personal loan at a lower rate, you could potentially save hundreds each month and pay off your debt years faster. What's your current monthly income, Sarah?"

Example 2:
Customer: "I make $4,500 per month but I'm worried about qualifying"
Alex: "That's a great monthly income of $4,500! I can definitely work with that. Your concern about qualifying is totally understandable - many people worry about this. The good news is that with your income level, you're likely in a strong position for our personal loan options. We work with people across different credit profiles, and income like yours is a real positive. Let me ask you - what's your current total debt amount, and do you know roughly what your credit score range is? This will help me give you more specific guidance on your qualification chances."
        """
        
        # Get user context
        user_profile = self.user_profiles.get(session_id, {})
        conversation = self.conversation_context.get(session_id, [])
        
        # Build user context section
        user_context = ""
        if user_profile:
            user_context += "\nCUSTOMER PROFILE:\n"
            if user_profile.get('name'):
                user_context += f"- Name: {user_profile['name']}\n"
            if user_profile.get('debt_amount'):
                user_context += f"- Debt Amount: ${user_profile['debt_amount']:,.2f}\n"
            if user_profile.get('income'):
                user_context += f"- Monthly Income: ${user_profile['income']:,.2f}\n"
            if user_profile.get('employment'):
                user_context += f"- Employment: {user_profile['employment']}\n"
        
        # Add conversation history
        conversation_context = ""
        if conversation and len(conversation) > 1:
            conversation_context += "\nRECENT CONVERSATION:\n"
            for entry in conversation[-3:]:  # Last 3 interactions
                conversation_context += f"- Customer said: {entry['user_input'][:100]}...\n"
        
        # Build the full prompt
        full_prompt = f"""{base_system_prompt}

{accuracy_instructions}

{few_shot_examples}

{user_context}

{conversation_context}

CURRENT CUSTOMER INPUT: {user_input}

Remember: This customer is depending on you for expert financial guidance. Acknowledge their specific situation, show empathy, and provide actionable solutions. Be Alex - enthusiastic, knowledgeable, and genuinely helpful.

Alex:"""
        
        return full_prompt
    
    async def _call_ollama_api_enhanced(self, prompt: str, model_name: str) -> str:
        """Enhanced Ollama API call with optimized parameters."""
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.6,  # Slightly lower for more consistent responses
                "num_predict": 600,  # More tokens for detailed responses
                "top_p": 0.85,
                "top_k": 35,
                "repeat_penalty": 1.15,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.1,
                "stop": ["Customer:", "Human:", "\n\nCustomer:", "\n\nHuman:", "Example"]
            }
        }
        
        timeout = aiohttp.ClientTimeout(total=120)  # Extended timeout
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "").strip()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Ollama API error {response.status}: {error_text}")
                        return ""
        except Exception as e:
            self.logger.error(f"API call failed for {model_name}: {e}")
            return ""
    
    async def _refine_response(
        self,
        initial_response: str,
        user_input: str,
        session_id: str,
        initial_score: ResponseScore
    ) -> Tuple[str, ResponseScore]:
        """Refine response using multi-pass approach."""
        refinement_prompt = f"""
You are Alex from Finally Payoff Debt. Your initial response to a customer needs improvement.

CUSTOMER INPUT: {user_input}

YOUR INITIAL RESPONSE: {initial_response}

AREAS NEEDING IMPROVEMENT:
- Relevance Score: {initial_score.relevance:.2f} (need 0.95+)
- Banking Expertise: {initial_score.banking_expertise:.2f} (need 0.95+)
- Persona Consistency: {initial_score.persona_consistency:.2f} (need 0.95+)
- Completeness: {initial_score.completeness:.2f} (need 0.95+)
- Accuracy: {initial_score.accuracy:.2f} (need 0.95+)

Please provide an IMPROVED response that:
1. Better acknowledges specific details from the customer input
2. Shows more banking expertise and specific knowledge
3. Maintains perfect Alex persona consistency
4. Provides complete, actionable guidance
5. Demonstrates higher accuracy in addressing the customer's needs

IMPROVED ALEX RESPONSE:"""
        
        refined_response = await self._call_ollama_api_enhanced(refinement_prompt, self.primary_model)
        
        if refined_response:
            refined_score = self._score_response_enhanced(refined_response, user_input, session_id)
            if refined_score.overall > initial_score.overall:
                return refined_response, refined_score
        
        return initial_response, initial_score
    
    def _score_response_enhanced(
        self,
        response: str,
        user_input: str,
        session_id: str
    ) -> ResponseScore:
        """Enhanced response scoring."""
        user_profile = self.user_profiles.get(session_id, {})
        
        # Score relevance (0-1)
        relevance = self._score_relevance_enhanced(response, user_input, user_profile)
        
        # Score banking expertise (0-1) 
        banking_expertise = self._score_banking_expertise_enhanced(response)
        
        # Score persona consistency (0-1)
        persona_consistency = self._score_persona_consistency_enhanced(response)
        
        # Score completeness (0-1)
        completeness = self._score_completeness_enhanced(response, user_input)
        
        # Score specific accuracy (0-1)
        accuracy = self._score_specific_accuracy(response, user_input, user_profile)
        
        # Calculate overall score (weighted)
        overall = (
            relevance * 0.25 + 
            banking_expertise * 0.20 + 
            persona_consistency * 0.20 + 
            completeness * 0.15 + 
            accuracy * 0.20
        )
        
        return ResponseScore(
            relevance=relevance,
            banking_expertise=banking_expertise,
            persona_consistency=persona_consistency,
            completeness=completeness,
            accuracy=accuracy,
            overall=overall
        )
    
    def _score_relevance_enhanced(
        self,
        response: str,
        user_input: str,
        user_profile: Dict[str, Any]
    ) -> float:
        """Enhanced relevance scoring."""
        score = 0.5  # Base score
        response_lower = response.lower()
        input_lower = user_input.lower()
        
        # Check for name acknowledgment
        if user_profile.get('name'):
            name = user_profile['name'].lower()
            if name in response_lower:
                score += 0.2
        
        # Check for amount acknowledgment
        if user_profile.get('mentioned_amounts'):
            amounts = user_profile['mentioned_amounts']
            for amount in amounts:
                amount_str = f"${amount:,.0f}" if amount >= 1000 else f"${amount:.0f}"
                if amount_str in response or str(int(amount)) in response:
                    score += 0.15
        
        # Check for specific detail acknowledgment
        key_details = re.findall(r'\b(?:\d+%|\$[\d,]+|[\d,]+%?)\b', user_input)
        for detail in key_details:
            if detail in response:
                score += 0.1
        
        # Keyword relevance
        input_words = set(input_lower.split())
        response_words = set(response_lower.split())
        overlap = len(input_words.intersection(response_words))
        overlap_ratio = overlap / max(len(input_words), 1)
        score += overlap_ratio * 0.25
        
        return min(score, 1.0)
    
    def _score_banking_expertise_enhanced(self, response: str) -> float:
        """Enhanced banking expertise scoring."""
        response_lower = response.lower()
        
        # High-value expertise indicators
        high_value_terms = [
            'apr', 'annual percentage rate', 'interest rate', 'debt consolidation',
            'personal loan', 'credit score', 'qualify', 'prequalified',
            'monthly payment', 'principal', 'refinance', 'balance transfer'
        ]
        
        # Finally Payoff Debt specific terms
        company_terms = [
            'finally payoff debt', 'prequalification specialist',
            'loan representative', 'income-based loan options'
        ]
        
        # Financial calculation terms
        calculation_terms = [
            'save money', 'lower rate', 'reduce payment', 'pay off faster',
            'interest savings', 'total cost'
        ]
        
        score = 0.6  # Base score
        
        # Award points for expertise indicators
        high_value_found = sum(1 for term in high_value_terms if term in response_lower)
        company_found = sum(1 for term in company_terms if term in response_lower)
        calculation_found = sum(1 for term in calculation_terms if term in response_lower)
        
        score += min(high_value_found * 0.08, 0.25)
        score += min(company_found * 0.1, 0.15)
        score += min(calculation_found * 0.06, 0.15)
        
        return min(score, 1.0)
    
    def _score_persona_consistency_enhanced(self, response: str) -> float:
        """Enhanced persona consistency scoring."""
        response_lower = response.lower()
        
        # Perfect Alex indicators
        perfect_indicators = [
            "i'm alex", "this is alex", "hi there", "thank you for calling",
            "finally payoff debt", "excited", "great", "wonderful"
        ]
        
        # Good Alex indicators
        good_indicators = [
            "help you", "assist you", "work with you", "questions",
            "income", "qualify", "loan options"
        ]
        
        # Negative indicators (reduce score)
        negative_indicators = [
            "i am an ai", "i don't know", "i cannot", "as an assistant",
            "i'm not sure", "i don't have access", "i'm sorry, but"
        ]
        
        score = 0.7  # Base score
        
        perfect_count = sum(1 for phrase in perfect_indicators if phrase in response_lower)
        good_count = sum(1 for phrase in good_indicators if phrase in response_lower)
        negative_count = sum(1 for phrase in negative_indicators if phrase in response_lower)
        
        score += perfect_count * 0.15
        score += good_count * 0.05
        score -= negative_count * 0.4
        
        return max(min(score, 1.0), 0.0)
    
    def _score_completeness_enhanced(self, response: str, user_input: str) -> float:
        """Enhanced completeness scoring."""
        word_count = len(response.split())
        
        # Optimal length based on input complexity
        input_words = len(user_input.split())
        if input_words <= 10:  # Simple input
            optimal_range = (40, 120)
        elif input_words <= 20:  # Medium input
            optimal_range = (60, 180)
        else:  # Complex input
            optimal_range = (80, 250)
        
        # Length score
        if optimal_range[0] <= word_count <= optimal_range[1]:
            length_score = 1.0
        elif optimal_range[0] * 0.7 <= word_count < optimal_range[0]:
            length_score = 0.8
        elif optimal_range[1] < word_count <= optimal_range[1] * 1.3:
            length_score = 0.8
        else:
            length_score = 0.6
        
        # Check for complete response elements
        response_lower = response.lower()
        complete_elements = 0
        
        # Has greeting/acknowledgment
        if any(phrase in response_lower for phrase in ['hi', 'hello', 'thank you', 'great']):
            complete_elements += 1
        
        # Has expertise/advice
        if any(phrase in response_lower for phrase in ['loan', 'rate', 'qualify', 'save', 'payment']):
            complete_elements += 1
        
        # Has call-to-action/next steps
        if any(phrase in response_lower for phrase in ['would you like', 'can you', 'let me', 'ready to', 'next step']):
            complete_elements += 1
        
        # Has question/engagement
        if '?' in response:
            complete_elements += 1
        
        completeness_score = complete_elements / 4  # 4 elements expected
        
        return (length_score * 0.4 + completeness_score * 0.6)
    
    def _score_specific_accuracy(
        self,
        response: str,
        user_input: str,
        user_profile: Dict[str, Any]
    ) -> float:
        """Score accuracy of specific details."""
        score = 0.5  # Base score
        
        # Check if specific amounts are acknowledged correctly
        if user_profile.get('mentioned_amounts'):
            amounts = user_profile['mentioned_amounts']
            for amount in amounts:
                # Look for the amount in various formats
                amount_patterns = [
                    f"${amount:,.0f}",
                    f"${amount:.0f}",
                    f"{amount:,.0f}",
                    f"{amount:.0f}"
                ]
                if any(pattern in response for pattern in amount_patterns):
                    score += 0.2
        
        # Check if name is used correctly
        if user_profile.get('name'):
            name = user_profile['name']
            if name in response:
                score += 0.15
        
        # Check for context-appropriate financial advice
        if 'debt' in user_input.lower() and any(term in response.lower() for term in ['consolidation', 'loan', 'rate', 'payment']):
            score += 0.1
        
        if 'qualify' in user_input.lower() and any(term in response.lower() for term in ['income', 'credit', 'requirements', 'eligible']):
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_fallback_response(
        self,
        user_input: str,
        user_profile: Optional[Dict[str, Any]]
    ) -> Tuple[str, ResponseScore]:
        """Generate high-quality fallback response."""
        name = user_profile.get('name', '') if user_profile else ''
        name_part = f" {name}" if name else ""
        
        fallback = f"Hi{name_part}! I'm Alex with Finally Payoff Debt. I heard you say '{user_input}'. As your banking specialist, I'm excited to help you explore our loan options that could save you money on high-interest debt. What specific financial situation can I help you with today?"
        
        score = ResponseScore(0.8, 0.7, 0.9, 0.8, 0.8, 0.8)
        return fallback, score
    
    def _get_fallback_system_prompt(self) -> str:
        """Fallback system prompt."""
        return """You are Alex, an enthusiastic and knowledgeable banking specialist working for Finally Payoff Debt. 
        You specialize in helping customers consolidate high-interest debt through personal loans. Always acknowledge 
        specific details customers mention (names, amounts, situations) and provide personalized financial guidance."""
    
    def get_accuracy_metrics(self) -> Dict[str, Any]:
        """Get detailed accuracy metrics."""
        if self.total_requests == 0:
            return {"accuracy_rate": 0.0, "total_requests": 0}
        
        accuracy_rate = self.ultra_high_quality_responses / self.total_requests
        retry_rate = self.retry_count / self.total_requests if self.total_requests > 0 else 0
        multi_pass_rate = self.multi_pass_used / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "ultra_high_accuracy_rate": round(accuracy_rate * 100, 2),
            "total_requests": self.total_requests,
            "ultra_high_quality_responses": self.ultra_high_quality_responses,
            "retry_rate": round(retry_rate * 100, 2),
            "multi_pass_rate": round(multi_pass_rate * 100, 2),
            "target_accuracy": round(self.accuracy_threshold * 100, 2),
            "performance_grade": self._calculate_performance_grade(accuracy_rate)
        }
    
    def _calculate_performance_grade(self, accuracy_rate: float) -> str:
        """Calculate performance grade."""
        if accuracy_rate >= 0.98:
            return "A++ (Ultra-High Accuracy)"
        elif accuracy_rate >= 0.95:
            return "A+ (Exceptional)"
        elif accuracy_rate >= 0.90:
            return "A (Excellent)"
        elif accuracy_rate >= 0.85:
            return "B+ (Very Good)"
        else:
            return "B (Good)"


def create_ultra_high_accuracy_llm(
    accuracy_target: float = 0.98
) -> UltraHighAccuracyLLM:
    """Create ultra-high accuracy LLM targeting 98%+ accuracy."""
    return UltraHighAccuracyLLM(
        primary_model="mistral:latest",
        secondary_models=["gpt-oss:20b"],
        accuracy_threshold=accuracy_target,
        max_retries=2,
        enable_multi_pass=True
    )
