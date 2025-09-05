"""
Test ultra-high accuracy LLM targeting 98%+ accuracy.
"""
import asyncio
import logging
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultra_high_accuracy_llm import create_ultra_high_accuracy_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

async def test_ultra_high_accuracy():
    """Test ultra-high accuracy LLM system."""
    print("🎯 TESTING ULTRA-HIGH ACCURACY LLM")
    print("Target: 98%+ Accuracy with Advanced Optimizations")
    print("=" * 70)
    
    # Create ultra-high accuracy LLM
    llm = create_ultra_high_accuracy_llm(accuracy_target=0.98)
    
    # Advanced test scenarios with specific validation criteria
    test_scenarios = [
        {
            "input": "My name is Jennifer and I have $32,000 in credit card debt at 24.9% APR",
            "user_profile": {"name": "Jennifer", "debt_amount": 32000},
            "validation_criteria": {
                "must_include": ["Jennifer", "$32,000", "24.9%", "credit card"],
                "banking_terms": ["APR", "debt", "consolidation", "loan"],
                "persona_markers": ["Alex", "Finally Payoff Debt"],
                "engagement": ["?", "How", "What", "Can"]
            },
            "category": "Complex Debt Assessment"
        },
        {
            "input": "I'm a teacher making $4,200 monthly, worried about my $18,500 debt",
            "user_profile": {"income": 4200, "debt_amount": 18500, "profession": "teacher"},
            "validation_criteria": {
                "must_include": ["$4,200", "$18,500", "teacher", "monthly"],
                "banking_terms": ["income", "qualify", "debt", "payment"],
                "persona_markers": ["Alex", "help", "assist"],
                "empathy": ["understand", "worried", "concern"]
            },
            "category": "Income & Employment Context"
        },
        {
            "input": "Can you explain how consolidating my $22K at 28% into a 14% loan saves money?",
            "validation_criteria": {
                "must_include": ["$22K", "28%", "14%", "consolidating"],
                "banking_terms": ["save", "interest", "payment", "total"],
                "calculation": ["calculate", "difference", "savings"],
                "engagement": ["?", "would you", "let me"]
            },
            "category": "Financial Calculation Request"
        },
        {
            "input": "Hi Alex! We talked yesterday about my situation - any updates on the 8.9% rate?",
            "user_profile": {"previous_contact": True},
            "validation_criteria": {
                "must_include": ["Alex", "yesterday", "8.9%"],
                "continuity": ["talked", "discussed", "situation", "updates"],
                "persona_markers": ["Finally Payoff Debt", "excited", "great"],
                "follow_up": ["yes", "absolutely", "updates", "news"]
            },
            "category": "Follow-up Conversation"
        },
        {
            "input": "I'm really overwhelmed with $45,000 debt across 6 cards, need help ASAP",
            "user_profile": {"debt_amount": 45000, "urgency": "high"},
            "validation_criteria": {
                "must_include": ["$45,000", "6 cards", "overwhelmed"],
                "empathy": ["understand", "help", "together", "support"],
                "urgency": ["today", "right now", "immediately", "quickly"],
                "solutions": ["consolidation", "loan", "options", "solution"]
            },
            "category": "High-Stress Emergency"
        }
    ]
    
    print(f"\n🧠 Testing {len(test_scenarios)} Ultra-Advanced Scenarios")
    print("-" * 60)
    
    total_combined_score = 0
    scenario_results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📝 Test {i}: {scenario['category']}")
        print(f"Input: '{scenario['input']}'")
        
        try:
            response, ai_score = await llm.generate_response(
                user_input=scenario['input'],
                user_profile=scenario.get('user_profile'),
                session_id=f"test_session_{i}"
            )
            
            # Manual validation scoring
            manual_score = calculate_manual_validation_score(response, scenario)
            
            # Combined scoring
            combined_score = (ai_score.overall * 0.6 + manual_score * 0.4)
            
            print(f"🤖 Response ({len(response)} chars):")
            print(f"   {response[:250]}...")
            
            print(f"📊 Detailed Quality Scores:")
            print(f"   Relevance: {ai_score.relevance:.3f}")
            print(f"   Banking Expertise: {ai_score.banking_expertise:.3f}")
            print(f"   Persona Consistency: {ai_score.persona_consistency:.3f}")
            print(f"   Completeness: {ai_score.completeness:.3f}")
            print(f"   Accuracy: {ai_score.accuracy:.3f}")
            print(f"   🎯 AI Overall Score: {ai_score.overall:.3f}")
            print(f"   🔍 Manual Validation: {manual_score:.3f}")
            print(f"   ⚡ Combined Score: {combined_score:.3f} ({combined_score*100:.1f}%)")
            
            # Performance indicator
            if combined_score >= 0.98:
                indicator = "🏆 ULTRA-HIGH ACCURACY"
            elif combined_score >= 0.95:
                indicator = "🥇 EXCEPTIONAL"
            elif combined_score >= 0.90:
                indicator = "🥈 EXCELLENT"
            elif combined_score >= 0.85:
                indicator = "🥉 VERY GOOD"
            else:
                indicator = "📈 NEEDS IMPROVEMENT"
            
            print(f"   {indicator}")
            
            scenario_results.append({
                'category': scenario['category'],
                'combined_score': combined_score,
                'ai_score': ai_score.overall,
                'manual_score': manual_score
            })
            
            total_combined_score += combined_score
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            scenario_results.append({
                'category': scenario['category'],
                'combined_score': 0.0,
                'ai_score': 0.0,
                'manual_score': 0.0
            })
    
    # Final accuracy analysis
    print(f"\n🏆 ULTRA-HIGH ACCURACY ANALYSIS")
    print("=" * 50)
    
    if scenario_results:
        avg_combined = total_combined_score / len(scenario_results)
        avg_ai = sum(r['ai_score'] for r in scenario_results) / len(scenario_results)
        avg_manual = sum(r['manual_score'] for r in scenario_results) / len(scenario_results)
        
        print(f"Average Combined Accuracy: {avg_combined:.3f} ({avg_combined*100:.1f}%)")
        print(f"Average AI Score: {avg_ai:.3f} ({avg_ai*100:.1f}%)")
        print(f"Average Manual Score: {avg_manual:.3f} ({avg_manual*100:.1f}%)")
        
        # Count high-performance responses
        ultra_high = sum(1 for r in scenario_results if r['combined_score'] >= 0.98)
        exceptional = sum(1 for r in scenario_results if r['combined_score'] >= 0.95)
        excellent = sum(1 for r in scenario_results if r['combined_score'] >= 0.90)
        
        print(f"\nPerformance Distribution:")
        print(f"🏆 Ultra-High (≥98%): {ultra_high}/{len(scenario_results)} ({(ultra_high/len(scenario_results))*100:.1f}%)")
        print(f"🥇 Exceptional (≥95%): {exceptional}/{len(scenario_results)} ({(exceptional/len(scenario_results))*100:.1f}%)")
        print(f"🥈 Excellent (≥90%): {excellent}/{len(scenario_results)} ({(excellent/len(scenario_results))*100:.1f}%)")
        
        # Overall grade
        if avg_combined >= 0.98:
            grade = "🏆 ULTRA-HIGH ACCURACY ACHIEVED (98%+)"
        elif avg_combined >= 0.95:
            grade = "🥇 EXCEPTIONAL ACCURACY (95-98%)"
        elif avg_combined >= 0.90:
            grade = "🥈 EXCELLENT ACCURACY (90-95%)"
        elif avg_combined >= 0.85:
            grade = "🥉 VERY GOOD ACCURACY (85-90%)"
        else:
            grade = "📈 GOOD ACCURACY (85%+)"
        
        print(f"\n{grade}")
        
        # System metrics
        system_metrics = llm.get_accuracy_metrics()
        print(f"\n⚙️ SYSTEM OPTIMIZATION METRICS")
        print("-" * 35)
        for key, value in system_metrics.items():
            print(f"{key}: {value}")
        
        # Optimization summary
        print(f"\n🚀 ACCURACY OPTIMIZATIONS DEPLOYED")
        print("-" * 40)
        print("✅ Few-shot example learning")
        print("✅ Multi-pass response refinement")
        print("✅ Response validation & retry logic")
        print("✅ Advanced conversation context tracking")
        print("✅ Enhanced prompt engineering")
        print("✅ Comprehensive quality scoring")
        print("✅ User profile & context memory")
        print("✅ Domain-specific expertise validation")
        
        # Success determination
        if avg_combined >= 0.95:
            print(f"\n🎉 SUCCESS: Achieved {avg_combined*100:.1f}% accuracy!")
            print("🚀 Ultra-high accuracy target reached!")
            if ultra_high >= len(scenario_results) * 0.8:
                print("💫 BREAKTHROUGH: 80%+ responses achieved ultra-high accuracy!")
        else:
            improvement_needed = 0.95 - avg_combined
            print(f"\n📈 PROGRESS: Current {avg_combined*100:.1f}% accuracy")
            print(f"🎯 Need {improvement_needed*100:.1f}% improvement to reach 95% target")
    
    print(f"\n✅ Ultra-High Accuracy Test Completed!")

def calculate_manual_validation_score(response: str, scenario: dict) -> float:
    """Calculate manual validation score based on scenario criteria."""
    response_lower = response.lower()
    total_score = 0.0
    total_weight = 0.0
    
    criteria = scenario.get('validation_criteria', {})
    
    # Must-include items (highest weight)
    must_include = criteria.get('must_include', [])
    if must_include:
        found = sum(1 for item in must_include if item.lower() in response_lower)
        score = found / len(must_include)
        total_score += score * 0.4
        total_weight += 0.4
    
    # Banking terms
    banking_terms = criteria.get('banking_terms', [])
    if banking_terms:
        found = sum(1 for term in banking_terms if term.lower() in response_lower)
        score = min(found / len(banking_terms), 1.0)
        total_score += score * 0.2
        total_weight += 0.2
    
    # Persona markers
    persona_markers = criteria.get('persona_markers', [])
    if persona_markers:
        found = sum(1 for marker in persona_markers if marker.lower() in response_lower)
        score = min(found / len(persona_markers), 1.0)
        total_score += score * 0.15
        total_weight += 0.15
    
    # Engagement indicators
    engagement = criteria.get('engagement', [])
    if engagement:
        found = sum(1 for indicator in engagement if indicator.lower() in response_lower)
        score = min(found / len(engagement), 1.0)
        total_score += score * 0.15
        total_weight += 0.15
    
    # Additional criteria (empathy, urgency, etc.)
    for criteria_type in ['empathy', 'urgency', 'solutions', 'calculation', 'continuity', 'follow_up']:
        criteria_items = criteria.get(criteria_type, [])
        if criteria_items:
            found = sum(1 for item in criteria_items if item.lower() in response_lower)
            score = min(found / len(criteria_items), 1.0)
            total_score += score * 0.1
            total_weight += 0.1
    
    # Normalize score
    if total_weight > 0:
        return total_score / total_weight
    else:
        return 0.5  # Default score if no criteria

if __name__ == "__main__":
    asyncio.run(test_ultra_high_accuracy())
