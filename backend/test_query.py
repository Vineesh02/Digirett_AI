"""
Test Script for Intent Detection Accuracy
Tests 20 queries (10 CASUAL, 10 LEGAL) in Norwegian and English
"""

import asyncio
import json
from app.services.fireworks_service import FireworksService
from app.config import settings

# Initialize service
llm_service = FireworksService(
    api_key=settings.FIREWORKS_API_KEY,
    model=settings.FIREWORKS_MODEL
)

# Test queries with expected results
TEST_QUERIES = [
    # ==================== CASUAL QUERIES ====================
    {
        "id": 1,
        "query": "Hei, hvordan har du det?",
        "language": "norwegian",
        "expected_intent": "CASUAL",
        "description": "Greeting"
    },
    {
        "id": 2,
        "query": "Jeg er trist i dag, trenger motivasjon",
        "language": "norwegian",
        "expected_intent": "CASUAL",
        "description": "Emotional support"
    },
    {
        "id": 3,
        "query": "Please give some jokes",
        "language": "English",
        "expected_intent": "CASUAL",
        "description": "Small talk - joke"
    },
    {
        "id": 4,
        "query": "Kan du snakke norsk?",
        "language": "norwegian",
        "expected_intent": "CASUAL",
        "description": "Language request"
    },
    {
        "id": 5,
        "query": "Hva synes du jeg burde gjøre med livet mitt?",
        "language": "norwegian",
        "expected_intent": "CASUAL",
        "description": "Life advice"
    },
    {
        "id": 6,
        "query": "Can you please motivate me",
        "language": "english",
        "expected_intent": "CASUAL",
        "description": "Motivation"
    },
    {
        "id": 7,
        "query": "I love you so much",
        "language": "english",
        "expected_intent": "CASUAL",
        "description": "Emotional expression"
    },
    {
        "id": 8,
        "query": "Tell me a joke",
        "language": "english",
        "expected_intent": "CASUAL",
        "description": "Small talk"
    },
    {
        "id": 9,
        "query": "How are you today?",
        "language": "english",
        "expected_intent": "CASUAL",
        "description": "Chitchat"
    },
    {
        "id": 10,
        "query": "Give me some life advice",
        "language": "english",
        "expected_intent": "CASUAL",
        "description": "Personal advice"
    },
    
    # ==================== LEGAL QUERIES ====================
    {
        "id": 11,
        "query": "Hva er aksjeloven?",
        "language": "norwegian",
        "expected_intent": "LEGAL",
        "description": "Specific law inquiry"
    },
    {
        "id": 12,
        "query": "Diskuter selskapets forpliktelser i henhold til norsk lov",
        "language": "norwegian",
        "expected_intent": "LEGAL",
        "description": "Vague legal discussion"
    },
    {
        "id": 13,
        "query": "Hvilke krav stilles til styremøter?",
        "language": "norwegian",
        "expected_intent": "LEGAL",
        "description": "Legal requirements"
    },
    {
        "id": 14,
        "query": "Discuss about child law",
        "language": "English",
        "expected_intent": "LEGAL",
        "description": "Legal question"
    },
    {
        "id": 15,
        "query": "Hva sier arbeidsmiljøloven om arbeidstid?",
        "language": "norwegian",
        "expected_intent": "LEGAL",
        "description": "Specific law reference"
    },
    {
        "id": 16,
        "query": "What is the company law in Norway?",
        "language": "english",
        "expected_intent": "LEGAL",
        "description": "General law inquiry"
    },
    {
        "id": 17,
        "query": "Discuss company obligation law",
        "language": "english",
        "expected_intent": "LEGAL",
        "description": "Vague legal discussion"
    },
    {
        "id": 18,
        "query": "What are the board meeting requirements?",
        "language": "english",
        "expected_intent": "LEGAL",
        "description": "Legal requirements"
    },
    {
        "id": 19,
        "query": "Discuss education law",
        "language": "english",
        "expected_intent": "LEGAL",
        "description": "Legal question"
    },
    {
        "id": 20,
        "query": "Tell me about Norwegian employment law",
        "language": "english",
        "expected_intent": "LEGAL",
        "description": "General law inquiry"
    }
]


async def test_intent_detection():
    """Test all 20 queries and report accuracy"""
    
    print("=" * 80)
    print("INTENT DETECTION TEST - 20 QUERIES")
    print("=" * 80)
    
    results = {
        "total": len(TEST_QUERIES),
        "correct": 0,
        "incorrect": 0,
        "errors": 0,
        "details": []
    }
    
    for test in TEST_QUERIES:
        print(f"\n[Test {test['id']}/20] {test['description']}")
        print(f"Query: \"{test['query']}\"")
        print(f"Expected: {test['expected_intent']} ({test['language']})")
        
        try:
            # FIXED: detect_intent is synchronous, no await needed
            intent_result = llm_service.detect_intent(test['query'])
            detected_intent = intent_result['intent']
            detected_language = intent_result['language']
            
            # Check if correct
            is_correct = (
                detected_intent == test['expected_intent'] and
                detected_language == test['language']
            )
            
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            print(f"Detected: {detected_intent} ({detected_language}) - {status}")
            
            if is_correct:
                results['correct'] += 1
            else:
                results['incorrect'] += 1
            
            results['details'].append({
                "id": test['id'],
                "query": test['query'],
                "expected": test['expected_intent'],
                "detected": detected_intent,
                "correct": is_correct
            })
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results['errors'] += 1
            results['details'].append({
                "id": test['id'],
                "query": test['query'],
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Queries: {results['total']}")
    print(f"✅ Correct: {results['correct']}")
    print(f"❌ Incorrect: {results['incorrect']}")
    print(f"⚠️  Errors: {results['errors']}")
    print(f"Accuracy: {(results['correct'] / results['total'] * 100):.1f}%")
    
    # Show incorrect ones
    if results['incorrect'] > 0:
        print("\n" + "=" * 80)
        print("INCORRECT CLASSIFICATIONS:")
        print("=" * 80)
        for detail in results['details']:
            if 'correct' in detail and not detail['correct']:
                print(f"Query {detail['id']}: \"{detail['query']}\"")
                print(f"  Expected: {detail['expected']}, Got: {detail['detected']}")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    asyncio.run(test_intent_detection())