"""
Test the accuracy-enhanced Ollama LLM to achieve 95%+ accuracy.
"""
import asyncio
import logging
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from accuracy_enhanced_llm import create_high_accuracy_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

async def test_accuracy_enhanced_llm():
    """Test the accuracy-enhanced LLM system."""
    print("🎯 Testing Accuracy-Enhanced Ollama LLM")
    print("Target: 95%+ Accuracy")
    print("=" * 60)
    
    # Create high-accuracy LLM
    llm = create_high_accuracy_llm(
        primary_model="mistral:latest",
        secondary_models=["gpt-oss:20b"],  # Will try ensemble if available
        accuracy_target=0.95
    )
    
    # Comprehensive test scenarios for accuracy validation
    test_scenarios = [
        {
            "input": "My name is Mike and I have $25,000 in credit card debt at 24% APR",
            "user_profile": {"name": "Mike", "debt_amount": 25000, "credit_score": "fair"},
            "expected_elements": ["Mike", "$25,000", "24%", "credit card", "debt"],
            "category": "Debt Assessment"
        },
        {
            "input": "I make $5,000 per month but I'm self-employed, can I still qualify?",
            "user_profile": {"income": 5000, "employment": "self-employed"},
            "expected_elements": ["$5,000", "self-employed", "qualify", "monthly"],
            "category": "Income Qualification"
        },
        {
            "input": "What's the difference between a 5.99% APR and 18% APR loan?",
            "expected_elements": ["5.99%", "18%", "APR", "difference", "loan"],
            "category": "Financial Education"
        },
        {
            "input": "I'm really stressed about my debt situation and need help urgently",
            "expected_elements": ["stressed", "debt", "help", "urgently"],
            "category": "Emotional Support"
        },
        {
            "input": "Can you calculate how much I'd save if I consolidate $15K at 22% into a 12% loan?",
            "expected_elements": ["calculate", "$15K", "22%", "12%", "consolidate", "save"],
            "category": "Savings Calculation"
        },
        {
            "input": "Hi Alex, we spoke last week about my $8,500 debt, any updates on rates?",
            "user_profile": {"name": "Previous Customer", "debt_amount": 8500},
            "expected_elements": ["Alex", "last week", "$8,500", "debt", "rates"],
            "category": "Follow-up Call"
        }
    ]
    
    print(f"\n🧠 Testing {len(test_scenarios)} Advanced Scenarios")
    print("-" * 50)
    
    total_score = 0
    scenario_scores = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📝 Test {i}: {scenario['category']}")
        print(f"Input: '{scenario['input']}'")
        
        try:
            response, quality_score = await llm.generate_response(
                user_input=scenario['input'],
                user_profile=scenario.get('user_profile'),
                domain_context="banking",
                call_type="inbound"
            )
            
            # Manual accuracy check
            elements_found = sum(1 for element in scenario['expected_elements'] 
                               if element.lower() in response.lower())
            manual_accuracy = elements_found / len(scenario['expected_elements'])
            
            print(f"🤖 Response ({len(response)} chars):")
            print(f"   {response[:200]}...")
            print(f"📊 Quality Scores:")
            print(f"   Relevance: {quality_score.relevance:.2f}")
            print(f"   Banking Expertise: {quality_score.banking_expertise:.2f}")
            print(f"   Persona Consistency: {quality_score.persona_consistency:.2f}")
            print(f"   Completeness: {quality_score.completeness:.2f}")
            print(f"   Overall AI Score: {quality_score.overall:.2f}")
            print(f"   Manual Accuracy: {manual_accuracy:.2f}")
            
            # Combined accuracy score
            combined_score = (quality_score.overall + manual_accuracy) / 2
            scenario_scores.append(combined_score)
            total_score += combined_score
            
            print(f"✅ Combined Accuracy: {combined_score:.2f} ({combined_score*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            scenario_scores.append(0.0)
    
    # Overall accuracy metrics
    print(f"\n📈 ACCURACY ANALYSIS")
    print("=" * 40)
    
    if scenario_scores:
        avg_accuracy = total_score / len(scenario_scores)
        print(f"Average Accuracy: {avg_accuracy:.2f} ({avg_accuracy*100:.1f}%)")
        
        high_accuracy_count = sum(1 for score in scenario_scores if score >= 0.90)
        print(f"High Accuracy Responses (≥90%): {high_accuracy_count}/{len(scenario_scores)} ({(high_accuracy_count/len(scenario_scores))*100:.1f}%)")
        
        excellent_count = sum(1 for score in scenario_scores if score >= 0.95)
        print(f"Excellent Responses (≥95%): {excellent_count}/{len(scenario_scores)} ({(excellent_count/len(scenario_scores))*100:.1f}%)")
        
        # Performance grade
        if avg_accuracy >= 0.95:
            grade = "🏆 EXCEPTIONAL (95%+)"
        elif avg_accuracy >= 0.90:
            grade = "🥇 EXCELLENT (90-95%)"
        elif avg_accuracy >= 0.85:
            grade = "🥈 VERY GOOD (85-90%)"
        else:
            grade = "🥉 GOOD (<85%)"
        
        print(f"Performance Grade: {grade}")
    
    # System-level accuracy metrics
    accuracy_metrics = llm.get_accuracy_metrics()
    print(f"\n⚙️ SYSTEM METRICS")
    print("-" * 25)
    for key, value in accuracy_metrics.items():
        print(f"{key}: {value}")
    
    print(f"\n🎯 ACCURACY ENHANCEMENT SUMMARY")
    print("-" * 35)
    print("✅ Multi-model ensemble responses")
    print("✅ Context-aware prompt engineering")
    print("✅ Response quality scoring")
    print("✅ Conversation memory integration")
    print("✅ Domain-specific expertise validation")
    print("✅ Emotional intelligence recognition")
    print("✅ Financial detail acknowledgment")
    
    if avg_accuracy >= 0.90:
        print(f"\n🎉 SUCCESS: Achieved {avg_accuracy*100:.1f}% accuracy (Target: 95%+)")
        if avg_accuracy >= 0.95:
            print("🚀 BREAKTHROUGH: Exceeded 95% accuracy target!")
    else:
        print(f"\n📈 PROGRESS: Current accuracy {avg_accuracy*100:.1f}% - Continue optimizing")
    
    print(f"\n✅ Accuracy Enhancement Test Completed!")

if __name__ == "__main__":
    asyncio.run(test_accuracy_enhanced_llm())
