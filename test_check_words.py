#!/usr/bin/env python3
"""
Test script for the check_words API endpoint
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:5000"

def test_check_words():
    """Test the check_words endpoint with sample misspelled words"""
    
    # Test words from your notebook
    test_words = [
        'nife',
        'rite', 
        'mountan',
        'naw',
        'freind',
        'fone',
        'sykology',
        'aline',
        'safary',
        'kitcehn'
    ]
    
    # Prepare request data
    data = {
        "words": test_words
    }
    
    try:
        # Send POST request
        response = requests.post(f"{API_URL}/check_words", json=data)
        
        if response.status_code == 200:
            result = response.json()
            
            print("=== API Response ===")
            print(f"Success: {result['success']}")
            print(f"Timestamp: {result['timestamp']}")
            
            print("\n=== Summary ===")
            summary = result['summary']
            print(f"Total Words: {summary['total_words']}")
            print(f"Correct Words: {summary['correct_words']}")
            print(f"Incorrect Words: {summary['incorrect_words']}")
            print(f"Accuracy: {summary['accuracy']:.2%}")
            print(f"Final Score: {summary['final_score']}")
            print(f"Final Streak: {summary['final_streak']}")
            print(f"Final Difficulty: {summary['final_difficulty']}")
            
            print("\n=== Individual Results ===")
            for i, word_result in enumerate(result['individual_results'], 1):
                status = "✓ Correct" if word_result['correct'] else "✗ Incorrect"
                print(f"{i:2d}. {word_result['word']:10s} - {status}")
                if not word_result['correct']:
                    print(f"    Error Types: {', '.join(word_result['error_types'])}")
                    print(f"    Max Confidence: {word_result['max_confidence_type']} ({word_result['confidence']:.2%})")
                    print(f"    Predicted Difficulty: {word_result['predicted_difficulty']}")
                print(f"    Score Change: {word_result['score_change']:+d} (Total: {word_result['total_score']})")
            
            print("\n=== Grouped Analysis by Error Type ===")
            grouped = result['grouped_analysis']
            for error_type, analysis in grouped.items():
                print(f"\nError Type: {error_type}")
                print(f"  Count: {analysis['count']}")
                print(f"  Average Confidence: {analysis['avg_confidence']:.2%}")
                print(f"  Average Difficulty: {analysis['avg_difficulty']:.2f}")
                print(f"  Words: {', '.join(analysis['words'])}")
                
                if analysis['suggested_words']:
                    print(f"  Suggested Words (Difficulty {analysis['rounded_difficulty_level']}):")
                    for j, suggestion in enumerate(analysis['suggested_words'], 1):
                        print(f"    {j}. {suggestion['incorrect_word']} (correct: {suggestion['correct_word']})")
            
            print(f"\n=== Final Game State ===")
            game_state = result['game_state']
            print(f"Score: {game_state['score']}")
            print(f"Level: {game_state['level']}")
            print(f"Streak: {game_state['streak']}")
            print(f"Difficulty: {game_state['difficulty']}")
            
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the Flask app is running on http://localhost:5000")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_single_word():
    """Test checking a single word"""
    
    data = {
        "words": ["speling"]
    }
    
    try:
        response = requests.post(f"{API_URL}/check_words", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== Single Word Test ===")
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {str(e)}")

def test_empty_request():
    """Test error handling with empty request"""
    
    try:
        response = requests.post(f"{API_URL}/check_words", json={})
        print(f"\n=== Empty Request Test ===")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Testing check_words API endpoint...")
    print("Make sure the Flask app is running first!")
    print("-" * 50)
    
    # Test main functionality
    test_check_words()
    
    # Test single word
    test_single_word()
    
    # Test error handling
    test_empty_request()
