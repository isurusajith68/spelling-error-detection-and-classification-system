#!/usr/bin/env python3
"""
Test script to verify JSON serialization works properly for check_words endpoint
"""

import requests
import json
import sys

def test_check_words_serialization():
    """Test that check_words endpoint returns properly serializable JSON"""
    
    # Test data
    test_data = {
        "words": ["nife", "rite", "hello"]  # Mix of incorrect and correct words
    }
    
    try:
        # Make request
        response = requests.post("http://localhost:5000/check_words", json=test_data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            # Try to parse JSON
            result = response.json()
            
            # Verify JSON serialization worked
            json_str = json.dumps(result, indent=2)
            print("✅ JSON serialization successful!")
            
            # Print basic info
            print(f"\nSummary:")
            summary = result.get('summary', {})
            print(f"  Total words: {summary.get('total_words')}")
            print(f"  Correct words: {summary.get('correct_words')}")
            print(f"  Accuracy: {summary.get('accuracy', 0):.2%}")
            print(f"  Final score: {summary.get('final_score')}")
            
            # Check individual results
            individual_results = result.get('individual_results', [])
            print(f"\nIndividual Results:")
            for i, word_result in enumerate(individual_results[:3], 1):  # Show first 3
                print(f"  {i}. {word_result.get('word')} - {'✅' if word_result.get('correct') else '❌'}")
                if not word_result.get('correct'):
                    print(f"     Error Type: {word_result.get('max_confidence_type')}")
                    print(f"     Confidence: {word_result.get('confidence', 0):.2%}")
            
            # Check grouped analysis
            grouped_analysis = result.get('grouped_analysis', {})
            print(f"\nGrouped Analysis:")
            for error_type, analysis in grouped_analysis.items():
                print(f"  {error_type}: {analysis.get('count')} words, avg confidence: {analysis.get('avg_confidence', 0):.2%}")
                
            return True
            
        else:
            print(f"❌ Error response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Make sure the Flask app is running on http://localhost:5000")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"Response text: {response.text[:500]}...")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_health_check():
    """Test health endpoint first"""
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code == 200:
            health = response.json()
            print(f"Health check: {health.get('status')}")
            print(f"Model loaded: {health.get('model_loaded')}")
            return health.get('model_loaded', False)
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

if __name__ == "__main__":
    print("Testing JSON serialization for check_words endpoint...")
    print("=" * 60)
    
    print("\n1. Testing health endpoint...")
    if not test_health_check():
        print("❌ Health check failed. Make sure the Flask app is running and model is loaded.")
        sys.exit(1)
    
    print("\n2. Testing check_words endpoint...")
    if test_check_words_serialization():
        print("\n✅ All tests passed! The JSON serialization issue has been fixed.")
    else:
        print("\n❌ Test failed. There may still be serialization issues.")
        sys.exit(1)
