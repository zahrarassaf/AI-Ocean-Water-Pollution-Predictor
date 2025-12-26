from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import glob

app = Flask(__name__)
CORS(app)

class ModelManager:
    def __init__(self):
        self.model_data = None
        self.load_latest_model()
    
    def load_latest_model(self):
        model_dirs = glob.glob('../models/final_model_*')
        if not model_dirs:
            print("Warning: No models found")
            return False
        
        model_dirs.sort(key=os.path.getmtime, reverse=True)
        model_path = os.path.join(model_dirs[0], 'final_model.joblib')
        
        try:
            self.model_data = joblib.load(model_path)
            print(f"Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, features):
        if self.model_data is None:
            raise ValueError("Model not loaded")
        
        features_array = np.array(features).reshape(1, -1)
        scaled_features = self.model_data['scaler'].transform(features_array)
        prediction = self.model_data['model'].predict(scaled_features)
        
        return float(prediction[0])

model_manager = ModelManager()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_manager.model_data is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
        
        features = data['features']
        
        if len(features) != len(model_manager.model_data['feature_names']):
            return jsonify({
                'error': f'Expected {len(model_manager.model_data["feature_names"])} features',
                'received': len(features)
            }), 400
        
        prediction = model_manager.predict(features)
        
        return jsonify({
            'prediction': prediction,
            'model_info': {
                'feature_count': len(model_manager.model_data['feature_names']),
                'model_type': type(model_manager.model_data['model']).__name__,
                'r2_score': model_manager.model_data['metrics'].get('test_r2', 0)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    if model_manager.model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'feature_names': model_manager.model_data['feature_names'],
        'metrics': model_manager.model_data['metrics'],
        'training_date': model_manager.model_data.get('training_date', 'N/A'),
        'model_type': type(model_manager.model_data['model']).__name__
    })

@app.route('/batch/predict', methods=['POST'])
def batch_predict():
    try:
        data = request.json
        
        if not data or 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
        
        samples = np.array(data['samples'])
        
        if samples.shape[1] != len(model_manager.model_data['feature_names']):
            return jsonify({
                'error': f'Expected {len(model_manager.model_data["feature_names"])} features per sample'
            }), 400
        
        predictions = []
        for sample in samples:
            prediction = model_manager.predict(sample.tolist())
            predictions.append(prediction)
        
        return jsonify({
            'predictions': predictions,
            'count': len(predictions)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
