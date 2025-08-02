from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import jellyfish
from collections import deque, defaultdict
from tensorflow.keras.models import load_model
import json
import os
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        self.df = self.clean_dataset(self.df)
        self.correct_words = set(self.df['Correct_Word'].str.lower())
        self.incorrect_words = set(self.df['Incorrect_Word'].str.lower())
        self.all_words = pd.concat([self.df['Incorrect_Word'], self.df['Correct_Word']]).unique()
        self.unique_words = len(set(self.all_words))
        self._create_char_encoding()
        self.df['Error_Type'] = self.df['Error_Type'].apply(lambda x: x.split(',') if ',' in x else [x])
        self.error_types = sorted(set([et for sublist in self.df['Error_Type'] for et in sublist]))
        self.label_encoder = {et: i for i, et in enumerate(self.error_types)}
        self.inverse_label_encoder = {i: et for et, i in self.label_encoder.items()}
        self.max_difficulty = self.df['Difficulty_Level'].max()
        self.error_type_counts = {et: sum(1 for errors in self.df['Error_Type'] for e in errors if e == et)
                                 for et in self.error_types}

    def _create_char_encoding(self):
        chars = set(''.join(self.all_words))
        self.char_to_int = {c: i+1 for i, c in enumerate(chars)}
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}
        self.max_length = int(np.percentile([len(word) for word in self.all_words], 95))

    def preprocess_word(self, word):
        processed = word.lower().ljust(self.max_length)[:self.max_length]
        return [self.char_to_int.get(c, 0) for c in processed]

    def decode_word(self, encoded_word):
        return ''.join([self.int_to_char.get(c, '') for c in encoded_word if c != 0])

    def phonetic_features(self, word):
        word = word.lower()
        correct_word = self.df[self.df['Incorrect_Word'] == word]['Correct_Word'].iloc[0] \
            if word in self.df['Incorrect_Word'].values else word
        
        edits = jellyfish.levenshtein_distance(word, correct_word)
        
        transposition_flag = any(word[i:i+2] == correct_word[i+1:i-1:-1] for i in range(len(word)-1)) and len(word) == len(correct_word)
        omission_flag = len(word) < len(correct_word)
        addition_flag = len(word) > len(correct_word)
        soundex_match = jellyfish.soundex(word) == jellyfish.soundex(correct_word)
        metaphone_match = jellyfish.metaphone(word) == jellyfish.metaphone(correct_word)
        nysiis_match = jellyfish.nysiis(word) == jellyfish.nysiis(correct_word)
        
        silent_letter_flag = False
        
        silent_letter_patterns = {
            'k': ['kn'], 'w': ['wr'], 'b': ['mb'], 'h': ['gh', 'rh', 'wh'],
            'l': ['lk', 'lm'], 'g': ['gn'], 'p': ['ps', 'pn'], 't': ['tch', 'tle'],
            'n': ['mn'], 'u': ['gu'], 'c': ['sc']
        }
        
        if omission_flag:
            for silent_letter, patterns in silent_letter_patterns.items():
                for pattern in patterns:
                    if any(pattern in correct_word and pattern not in word for pattern in patterns):
                        silent_letter_flag = True
                        break
        
        common_silent_combinations = [
            ('k', 'kn', 'n'),  
            ('g', 'gn', 'n'),  
            ('w', 'wr', 'r'),  
            ('p', 'ps', 's'),  
            ('b', 'mb', 'm'),  
            ('h', 'gh', 'g')   
        ]
        
        for silent_letter, full_pattern, shortened in common_silent_combinations:
            if shortened in word and full_pattern in correct_word:
                silent_letter_flag = True
                break
        
        if correct_word.endswith('e') and not word.endswith('e') and soundex_match:
            silent_letter_flag = True
        
        phoneme_mismatch_flag = not metaphone_match and not transposition_flag and not silent_letter_flag
        
        common_substitutions = [
            ('f', 'ph'), ('f', 'gh'), ('k', 'ch'), ('k', 'q'),
            ('s', 'c'), ('z', 's'), ('j', 'g'), ('ee', 'ea'),
            ('i', 'y'), ('ks', 'x'), ('shun', 'tion'), ('kw', 'qu')
        ]
        
        substitution_flag = False
        for sub_pair in common_substitutions:
            if (sub_pair[0] in word and sub_pair[1] in correct_word) or (sub_pair[1] in word and sub_pair[0] in correct_word):
                substitution_flag = True
                break
        
        doubled_letter_flag = False
        for i in range(len(word)-1):
            if word[i] == word[i+1]:
                if i+1 >= len(correct_word) or i >= len(correct_word) or word[i] != correct_word[i] or word[i+1] != correct_word[i+1]:
                    doubled_letter_flag = True
                    break
        
        vowels = 'aeiou'
        vowel_confusion = False
        word_vowels = [c for c in word if c in vowels]
        correct_vowels = [c for c in correct_word if c in vowels]
        if len(word_vowels) != len(correct_vowels) or ''.join(word_vowels) != ''.join(correct_vowels):
            vowel_confusion = True
        
        return np.array([
            edits,
            transposition_flag,
            omission_flag,
            addition_flag,
            phoneme_mismatch_flag,
            silent_letter_flag,
            substitution_flag,
            doubled_letter_flag,
            vowel_confusion,
            int(not soundex_match),
            int(not metaphone_match),
            int(not nysiis_match)
        ], dtype=float)

    def get_difficulty(self, word):
        return self.df[self.df['Incorrect_Word'] == word]['Difficulty_Level'].iloc[0] \
            if word in self.df['Incorrect_Word'].values else 0

    def suggest_words_by_difficulty(self, difficulty, count=5):
        return self.df[self.df['Difficulty_Level'] == difficulty]['Incorrect_Word'].sample(min(count, len(self.df[self.df['Difficulty_Level'] == difficulty]))).tolist()

    def clean_dataset(self, df):
        df.loc[df['Incorrect_Word'] == df['Correct_Word'], 'Error_Type'] = ['Correct']
        return df.drop_duplicates(subset=['Incorrect_Word', 'Correct_Word'], keep='first')

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

class SpellingTutor:
    def __init__(self, dataset_path, filepath, metadata_path):
        self.processor = DataProcessor(dataset_path)
        self.difficulty_manager = DifficultyManager()
        self.game = GameEngine()
        self.load_model(filepath, metadata_path, dataset_path)

    def load_model(self, filepath, metadata_path, dataset_path):
        self.model = load_model(filepath)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.processor.char_to_int = metadata['char_to_int']
            self.processor.max_length = metadata['max_length']
            self.processor.correct_words = set(metadata['correct_words'])
            self.processor.label_encoder = metadata['label_encoder']
            self.processor.inverse_label_encoder = {int(k): v for k, v in metadata['inverse_label_encoder'].items()}

    def process_word(self, word):
        correct = word.lower() in self.processor.correct_words
        seq_input = np.array([self.processor.preprocess_word(word)])
        ph_input = np.array([self.processor.phonetic_features(word)])
        error_proba, diff_pred = self.model.predict([seq_input, ph_input], verbose=0)

        if correct:
            error_types = ['Correct']
            confidence = 1.0
            max_confidence_type = 'Correct'
            confidences = {'Correct': 1.0}
        else:
            error_types = [self.processor.inverse_label_encoder[i] for i, prob in enumerate(error_proba[0]) if prob > 0.3]
            confidences = {self.processor.inverse_label_encoder[i]: float(prob) for i, prob in enumerate(error_proba[0]) if prob > 0.3}
            max_confidence_idx = np.argmax(error_proba[0])
            max_confidence_type = self.processor.inverse_label_encoder[max_confidence_idx]
            confidence = float(error_proba[0][max_confidence_idx])
            if not error_types:
                error_types = ['Unknown']
                confidences = {'Unknown': confidence}

        difficulty_pred = float(diff_pred[0][0])
        actual_difficulty = self.processor.get_difficulty(word) if not correct else 0

        self.difficulty_manager.update(correct)
        self.game.update(correct, self.difficulty_manager.difficulty)
        
        return {
            'word': word,
            'correct': correct,
            'error_types': error_types,
            'confidences': confidences,
            'max_confidence_type': max_confidence_type,
            'confidence': confidence,
            'predicted_difficulty': round(difficulty_pred, 2),
            'actual_difficulty': actual_difficulty,
            'game_difficulty': self.difficulty_manager.difficulty,
            'score': self.game.score,
            'level': self.game.level,
            'streak': self.game.streak
        }

    def get_suggestions(self, error_type, difficulty=None, count=5):
        """Get word suggestions based on error type and difficulty"""
        if difficulty is None:
            difficulty = self.difficulty_manager.difficulty
        
        filtered_df = self.processor.df[
            self.processor.df['Error_Type'].apply(lambda x: error_type in x) &
            (self.processor.df['Difficulty_Level'] == difficulty)
        ]
        
        if filtered_df.empty:
            for adj_diff in [difficulty-1, difficulty+1]:
                if 1 <= adj_diff <= self.processor.max_difficulty:
                    filtered_df = self.processor.df[
                        self.processor.df['Error_Type'].apply(lambda x: error_type in x) &
                        (self.processor.df['Difficulty_Level'] == adj_diff)
                    ]
                    if not filtered_df.empty:
                        break
        
        if not filtered_df.empty:
            suggestions = filtered_df.sample(min(count, len(filtered_df)))
            return [
                {
                    'incorrect_word': row['Incorrect_Word'],
                    'correct_word': row['Correct_Word'],
                    'difficulty': row['Difficulty_Level']
                }
                for _, row in suggestions.iterrows()
            ]
        return []

tutor = None

def initialize_tutor():
    """Initialize the spelling tutor with model files"""
    global tutor
    try:
        dataset_path = "final.csv"
        model_path = "spelling_tutor_model.keras" 
        metadata_path = "spelling_tutor_metadata.json"
        
        missing_files = []
        for file_path in [dataset_path, model_path, metadata_path]:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            logger.error("Please ensure you have:")
            logger.error("1. final.csv - Your dataset file")
            logger.error("2. spelling_tutor_model.keras - Run your training script to generate this")
            logger.error("3. spelling_tutor_metadata.json - Generated alongside the model")
            return False
            
        logger.info("All required files found, initializing model...")
        tutor = SpellingTutor(dataset_path, model_path, metadata_path)
        logger.info("Spelling tutor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize tutor: {str(e)}")
        logger.error("If you're missing model files, run your training script first:")
        logger.error("tutor = SpellingTutor(dataset_path='final.csv')")
        return False
    

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': tutor is not None
    })

@app.route('/check_word', methods=['POST'])
def check_word():
    """Check a single word for spelling errors"""
    try:
        if tutor is None:
            return jsonify({'error': 'Model not initialized'}), 500
            
        data = request.get_json()
        if not data or 'word' not in data:
            return jsonify({'error': 'Word parameter is required'}), 400
            
        word = data['word'].strip()
        if not word:
            return jsonify({'error': 'Word cannot be empty'}), 400
            
        result = tutor.process_word(word)
        
        for key, value in result.items():
            if isinstance(value, np.integer):
                result[key] = int(value)
            elif isinstance(value, np.floating):
                result[key] = float(value)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing word: {str(e)}")
        return jsonify({'error': f'Failed to process word: {str(e)}'}), 500

@app.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Get word suggestions based on error type and difficulty"""
    try:
        if tutor is None:
            return jsonify({'error': 'Model not initialized'}), 500
            
        error_type = request.args.get('error_type')
        difficulty = request.args.get('difficulty', type=int)
        count = request.args.get('count', default=5, type=int)
        
        if not error_type:
            return jsonify({'error': 'error_type parameter is required'}), 400
            
        if error_type not in tutor.processor.error_types:
            return jsonify({
                'error': f'Invalid error type. Available types: {tutor.processor.error_types}'
            }), 400
            
        if count > 50: 
            count = 50
            
        suggestions = tutor.get_suggestions(error_type, difficulty, count)
        
        return jsonify({
            'success': True,
            'error_type': error_type,
            'difficulty': difficulty or tutor.difficulty_manager.difficulty,
            'suggestions': suggestions,
            'count': len(suggestions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {str(e)}")
        return jsonify({'error': f'Failed to get suggestions: {str(e)}'}), 500

@app.route('/game_status', methods=['GET'])
def get_game_status():
    """Get current game status"""
    try:
        if tutor is None:
            return jsonify({'error': 'Model not initialized'}), 500
            
        return jsonify({
            'success': True,
            'game_status': {
                'score': tutor.game.score,
                'level': tutor.game.level,
                'streak': tutor.game.streak,
                'difficulty': tutor.difficulty_manager.difficulty,
                'performance_history': list(tutor.difficulty_manager.performance)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting game status: {str(e)}")
        return jsonify({'error': f'Failed to get game status: {str(e)}'}), 500

@app.route('/reset_game', methods=['POST'])
def reset_game():
    """Reset game progress"""
    try:
        if tutor is None:
            return jsonify({'error': 'Model not initialized'}), 500
            
        tutor.difficulty_manager = DifficultyManager()
        tutor.game = GameEngine()
        
        return jsonify({
            'success': True,
            'message': 'Game reset successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error resetting game: {str(e)}")
        return jsonify({'error': f'Failed to reset game: {str(e)}'}), 500

@app.route('/error_types', methods=['GET'])
def get_error_types():
    """Get available error types"""
    try:
        if tutor is None:
            return jsonify({'error': 'Model not initialized'}), 500
            
        return jsonify({
            'success': True,
            'error_types': tutor.processor.error_types,
            'error_counts': tutor.processor.error_type_counts,
            'max_difficulty': tutor.processor.max_difficulty,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting error types: {str(e)}")
        return jsonify({'error': f'Failed to get error types: {str(e)}'}), 500

@app.route('/check_words', methods=['POST'])
def check_words():
    """Check multiple words and return grouped results analysis"""
    try:
        if tutor is None:
            return jsonify({'error': 'Model not initialized'}), 500
            
        data = request.get_json()
        if not data or 'words' not in data:
            return jsonify({'error': 'words list is required in request body'}), 400
            
        test_words = data['words']
        if not isinstance(test_words, list):
            return jsonify({'error': 'words must be a list'}), 400
            
        if len(test_words) > 100:  # Limit to prevent overload
            return jsonify({'error': 'Maximum 100 words allowed per request'}), 400
            
        results = []
        previous_score = tutor.game.score
        
        for word in test_words:
            if not isinstance(word, str) or not word.strip():
                continue
                
            result = tutor.process_word(word.strip())
            score_change = result['score'] - previous_score
            
            processed_result = {}
            for key, value in result.items():
                if isinstance(value, np.integer):
                    processed_result[key] = int(value)
                elif isinstance(value, np.floating):
                    processed_result[key] = float(value)
                elif isinstance(value, dict):
                    processed_result[key] = {k: float(v) if isinstance(v, np.floating) else v for k, v in value.items()}
                else:
                    processed_result[key] = value
            
            results.append({
                'word': processed_result['word'],
                'correct': processed_result['correct'],
                'error_types': processed_result['error_types'],
                'confidences': processed_result['confidences'],
                'max_confidence_type': processed_result['max_confidence_type'],
                'confidence': processed_result['confidence'],
                'predicted_difficulty': processed_result['predicted_difficulty'],
                'actual_difficulty': int(processed_result['actual_difficulty']) if isinstance(processed_result['actual_difficulty'], np.integer) else processed_result['actual_difficulty'],
                'score_change': int(score_change) if isinstance(score_change, np.integer) else score_change,
                'total_score': processed_result['score'],
                'streak': processed_result['streak'],
                'difficulty': processed_result['game_difficulty']
            })
            previous_score = processed_result['score']
        
        grouped_results = defaultdict(lambda: {
            'count': 0, 
            'total_score_change': 0, 
            'words': [], 
            'avg_confidence': 0.0, 
            'total_difficulty': 0.0
        })
        
        for result in results:
            if not result['correct']:
                error_type = result['max_confidence_type']
                grouped_results[error_type]['count'] += 1
                grouped_results[error_type]['total_score_change'] += result['score_change']
                grouped_results[error_type]['words'].append(result['word'])
                grouped_results[error_type]['avg_confidence'] += result['confidence']
                grouped_results[error_type]['total_difficulty'] += result['predicted_difficulty']
        
        analysis_by_error_type = {}
        for error_type, data in grouped_results.items():
            if data['count'] > 0:
                avg_confidence = data['avg_confidence'] / data['count']
                avg_difficulty = data['total_difficulty'] / data['count']
                rounded_difficulty_level = round(avg_difficulty)
                
                suggested_words = []
                
                if hasattr(tutor, "processor") and hasattr(tutor.processor, "df"):
                    error_type_words = tutor.processor.df[
                        (tutor.processor.df['Error_Type'].apply(lambda x: error_type in x)) &
                        (tutor.processor.df['Difficulty_Level'] == rounded_difficulty_level)
                    ]
                    
                    if not error_type_words.empty:
                        new_words = error_type_words[~error_type_words['Incorrect_Word'].isin(data['words'])]
                        
                        if not new_words.empty:
                            sample_size = min(5, len(new_words))
                            suggested_samples = new_words['Incorrect_Word'].sample(sample_size).tolist()
                            
                            for word in suggested_samples:
                                difficulty = tutor.processor.get_difficulty(word)
                                correct_word = error_type_words[
                                    error_type_words['Incorrect_Word'] == word
                                ]['Correct_Word'].iloc[0]
                                suggested_words.append({
                                    'incorrect_word': word,
                                    'correct_word': correct_word,
                                    'difficulty': int(difficulty) if isinstance(difficulty, np.integer) else difficulty
                                })
                
                analysis_by_error_type[error_type] = {
                    'count': data['count'],
                    'avg_confidence': round(avg_confidence, 4),
                    'avg_difficulty': round(avg_difficulty, 2),
                    'rounded_difficulty_level': rounded_difficulty_level,
                    'words': data['words'],
                    'total_score_change': data['total_score_change'],
                    'suggested_words': suggested_words
                }
        
        total_words = len(test_words)
        correct_words = sum(1 for result in results if result['correct'])
        incorrect_words = total_words - correct_words
        accuracy = correct_words / total_words if total_words > 0 else 0
        
        final_game_state = {
            'score': int(tutor.game.score) if isinstance(tutor.game.score, np.integer) else tutor.game.score,
            'level': int(tutor.game.level) if isinstance(tutor.game.level, np.integer) else tutor.game.level,
            'streak': int(tutor.game.streak) if isinstance(tutor.game.streak, np.integer) else tutor.game.streak,
            'difficulty': int(tutor.difficulty_manager.difficulty) if isinstance(tutor.difficulty_manager.difficulty, np.integer) else tutor.difficulty_manager.difficulty
        }
        
        return jsonify({
            'success': True,
            'summary': {
                'total_words': total_words,
                'correct_words': correct_words,
                'incorrect_words': incorrect_words,
                'accuracy': round(accuracy, 4),
                'final_score': int(tutor.game.score) if isinstance(tutor.game.score, np.integer) else tutor.game.score,
                'final_streak': int(tutor.game.streak) if isinstance(tutor.game.streak, np.integer) else tutor.game.streak,
                'final_difficulty': int(tutor.difficulty_manager.difficulty) if isinstance(tutor.difficulty_manager.difficulty, np.integer) else tutor.difficulty_manager.difficulty
            },
            'individual_results': results,
            'grouped_analysis': analysis_by_error_type,
            'game_state': final_game_state,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error checking words: {str(e)}")
        return jsonify({'error': f'Failed to check words: {str(e)}'}), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get dataset statistics"""
    try:
        if tutor is None:
            return jsonify({'error': 'Model not initialized'}), 500
            
        df = tutor.processor.df
        
        stats = {
            'total_words': len(df),
            'unique_correct_words': len(tutor.processor.correct_words),
            'unique_incorrect_words': len(tutor.processor.incorrect_words),
            'error_type_distribution': tutor.processor.error_type_counts,
            'difficulty_distribution': df['Difficulty_Level'].value_counts().to_dict(),
            'average_word_length': df['Incorrect_Word'].str.len().mean(),
            'max_difficulty': tutor.processor.max_difficulty
        }
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({'error': f'Failed to get statistics: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if initialize_tutor():
        logger.info("Starting Flask API server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize tutor. Please check if model files exist.")