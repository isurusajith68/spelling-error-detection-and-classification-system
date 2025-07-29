# Spelling Error Detection API - check_words Endpoint

## Overview

The `check_words` endpoint processes multiple words and returns detailed analysis including grouped results by error type, similar to the analysis performed in your Jupyter notebook.

## Usage

### 1. Start the Flask Application

```bash
python app.py
```

### 2. Test the check_words Endpoint

#### Using curl:

```bash
curl -X POST http://localhost:5000/check_words \
  -H "Content-Type: application/json" \
  -d '{
    "words": ["nife", "rite", "mountan", "naw", "freind", "fone", "sykology", "aline", "safary", "kitcehn"]
  }'
```

#### Using Python requests:

```python
import requests

data = {
    "words": ["nife", "rite", "mountan", "naw", "freind", "fone", "sykology", "aline", "safary", "kitcehn"]
}

response = requests.post("http://localhost:5000/check_words", json=data)
result = response.json()
print(result)
```

#### Using the test script:

```bash
python test_check_words.py
```

## API Response Format

The endpoint returns a JSON response with the following structure:

```json
{
  "success": true,
  "summary": {
    "total_words": 10,
    "correct_words": 0,
    "incorrect_words": 10,
    "accuracy": 0.0,
    "final_score": -20,
    "final_streak": 0,
    "final_difficulty": 1
  },
  "individual_results": [
    {
      "word": "nife",
      "correct": false,
      "error_types": ["Phoneme Mismatch"],
      "confidences": { "Phoneme Mismatch": 0.85 },
      "max_confidence_type": "Phoneme Mismatch",
      "confidence": 0.85,
      "predicted_difficulty": 2.3,
      "actual_difficulty": 2,
      "score_change": -2,
      "total_score": -2,
      "streak": 0,
      "difficulty": 1
    }
    // ... more individual results
  ],
  "grouped_analysis": {
    "Phoneme Mismatch": {
      "count": 3,
      "avg_confidence": 0.8234,
      "avg_difficulty": 2.45,
      "rounded_difficulty_level": 2,
      "words": ["nife", "fone", "sykology"],
      "total_score_change": -6,
      "suggested_words": [
        {
          "incorrect_word": "lite",
          "correct_word": "light",
          "difficulty": 2
        }
        // ... more suggestions
      ]
    }
    // ... more error types
  },
  "game_state": {
    "score": -20,
    "level": 1,
    "streak": 0,
    "difficulty": 1
  },
  "timestamp": "2025-07-25T10:30:45.123456"
}
```

## Features

### Individual Word Analysis

- Word correctness detection
- Error type classification with confidence scores
- Difficulty prediction
- Game mechanics (score, streak, level)

### Grouped Analysis by Error Type

- Count of words per error type
- Average confidence scores
- Average difficulty levels
- Word suggestions for practice

### Game Integration

- Score tracking with difficulty-based penalties/rewards
- Streak counting
- Dynamic difficulty adjustment
- Level progression

## Error Handling

The endpoint includes comprehensive error handling:

- **400 Bad Request**: Missing or invalid `words` parameter
- **500 Internal Server Error**: Model not initialized or processing errors
- **Rate Limiting**: Maximum 100 words per request

## Example Error Response

```json
{
  "error": "words list is required in request body"
}
```

## Related Endpoints

- `GET /health` - Check API health
- `POST /process_word` - Process single word
- `GET /error_types` - Get available error types
- `GET /statistics` - Get dataset statistics
- `GET /game_status` - Get current game status
- `POST /reset_game` - Reset game progress

## Requirements

Make sure the following files exist:

- `final.csv` - Training dataset
- `spelling_tutor_model.keras` - Trained model
- `spelling_tutor_metadata.json` - Model metadata

The Flask app will automatically load these files on startup.
