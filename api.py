from flask import Flask, request, jsonify
import numpy as np
import jellyfish
from collections import deque, defaultdict
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)

class DataProcessor:
    def __init__(self, metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.char_to_int = metadata['char_to_int']
        self.max_length = metadata['max_length']
        self.correct_words = set(metadata['correct_words'])
        self.label_encoder = metadata['label_encoder']
        self.inverse_label_encoder = {int(k): v for k, v in metadata['inverse_label_encoder'].items()}

    def preprocess_word(self, word):
        processed = word.lower().ljust(self.max_length)[:self.max_length]
        return [self.char_to_int.get(c, 0) for c in processed]

    def phonetic_features(self, word, correct_word=None):
        word = word.lower()
        correct_word = correct_word or word
        edits = jellyfish.levenshtein_distance(word, correct_word)
        transposition_flag = 1 if any(word[i:i+2] == correct_word[i+1:i-1:-1] for i in range(len(word)-1)) and len(word) == len(correct_word) else 0
        omission_flag = 1 if len(word) < len(correct_word) else 0
        phoneme_mismatch_flag = 1 if jellyfish.metaphone(word) != jellyfish.metaphone(correct_word) and not transposition_flag else 0
        silent_letter_flag = 1 if edits > 0 and jellyfish.soundex(word) == jellyfish.soundex(correct_word) and not omission_flag else 0
        reversal_flag = 1 if word[::-1] == correct_word or sum(a != b for a, b in zip(word[::-1], correct_word)) <= 2 else 0
        return np.array([edits, transposition_flag, omission_flag, phoneme_mismatch_flag, silent_letter_flag, reversal_flag], dtype=float)

class DifficultyManager:
    def __init__(self):
        self.difficulty = 1
        self.performance = deque(maxlen=10)

    def update(self, correct):
        self.performance.append(correct)
        success_rate = np.mean(self.performance) if len(self.performance) > 0 else 0
        if success_rate > 0.7:
            self.difficulty = min(5, self.difficulty + 1)
        elif success_rate < 0.3:
            self.difficulty = max(1, self.difficulty - 1)

class GameEngine:
    def __init__(self):
        self.score = 0
        self.streak = 0
        self.level = 1

    def update(self, correct, difficulty):
        if correct:
            self.streak += 1
            self.score += (10 * difficulty) + (5 * self.streak)
            if self.score >= self.level * 100:
                self.level = min(10, self.level + 1)
        else:
            self.streak = 0
            self.score = max(0, self.score - (2 * difficulty))
            if self.score < (self.level - 1) * 100:
                self.level = max(1, self.level - 1)

sessions = {}

class SpellingTutor:
    def __init__(self, metadata_path, model_path):
        self.processor = DataProcessor(metadata_path)
        self.model = load_model(model_path)

    def process_words(self, words, session_id):
        if session_id not in sessions:
            sessions[session_id] = {
                'difficulty': DifficultyManager(),
                'game': GameEngine(),
                'results': [],
                'previous_score': 0
            }
        
        difficulty = sessions[session_id]['difficulty']
        game = sessions[session_id]['game']
        results = []

        for word in words:
            previous_score = game.score
            correct = word.lower() in self.processor.correct_words
            if correct:
                error_type = None
                confidence = 1.0
                ph_input = self.processor.phonetic_features(word)
            else:
                seq_input = np.array([self.processor.preprocess_word(word)])
                ph_input = np.array([self.processor.phonetic_features(word)])
                proba = self.model.predict([seq_input, ph_input])[0]
                error_type_idx = np.argmax(proba)
                error_type = self.processor.inverse_label_encoder[error_type_idx]
                confidence = float(proba[error_type_idx])
            
            difficulty.update(correct)
            game.update(correct, difficulty.difficulty)
            
            score_change = game.score - previous_score
            result = {
                'word': word,
                'correct': correct,
                'error_type': error_type if not correct else 'Correct',
                'confidence': confidence,
                'score_change': score_change,
                'total_score': game.score,
                'difficulty': difficulty.difficulty,
                'streak': game.streak,
                'level': game.level
            }
            results.append(result)
            sessions[session_id]['results'].append(result)
            sessions[session_id]['previous_score'] = game.score

        grouped_results = defaultdict(lambda: {'count': 0, 'total_score_change': 0, 'words': []})
        for result in results:
            error_type = result['error_type']
            grouped_results[error_type]['count'] += 1
            grouped_results[error_type]['total_score_change'] += result['score_change']
            grouped_results[error_type]['words'].append(result['word'])

        return dict(grouped_results)

tutor = SpellingTutor("spelling_tutor_metadata.json", "spelling_tutor_model.h5")

@app.route('/process_words', methods=['POST'])
def process_words():
    data = request.get_json()
    words = data.get('words')
    session_id = data.get('session_id', 'default')

    if not words or not isinstance(words, list):
        return jsonify({'error': 'A list of words is required'}), 400
    
    if len(words) != 10:
        return jsonify({'error': 'Exactly 10 words are required'}), 400

    grouped_results = tutor.process_words(words, session_id)
    return jsonify(grouped_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)