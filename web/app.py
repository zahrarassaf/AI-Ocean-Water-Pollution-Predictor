from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import json

app = Flask(__name__)

# Load the latest model
def load_latest_model():
    import glob
    
    model_dirs = glob.glob('models/final_model_*')
    if not model_dirs:
        return None
    
    model_dirs.sort(key=os.path.getmtime, reverse=True)
    model_path = os.path.join(model_dirs[0], 'final_model.joblib')
    
    return joblib.load(model_path)

model_data = load_latest_model()

@app.route('/')
def index():
    if model_data is None:
        return render_template('error.html', 
                             message="No trained model found. Please train a model first.")
    
    return render_template('index.html',
                         feature_names=model_data['feature_names'],
                         metrics=model_data['metrics'])

@app.route('/predict', methods=['POST'])
def predict():
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        if 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        
        if len(features[0]) != len(model_data['feature_names']):
            return jsonify({
                'error': f'Expected {len(model_data["feature_names"])} features, '
                        f'got {len(features[0])}'
            }), 400
        
        scaled_features = model_data['scaler'].transform(features)
        prediction = model_data['model'].predict(scaled_features)
        
        return jsonify({
            'prediction': float(prediction[0]),
            'features_used': model_data['feature_names'],
            'model_metrics': model_data['metrics']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'training_date': model_data.get('training_date', 'N/A'),
        'feature_count': len(model_data['feature_names']),
        'model_type': type(model_data['model']).__name__,
        'metrics': model_data['metrics'],
        'feature_names': model_data['feature_names']
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        if 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
        
        samples = np.array(data['samples'])
        
        if samples.shape[1] != len(model_data['feature_names']):
            return jsonify({
                'error': f'Expected {len(model_data["feature_names"])} features per sample, '
                        f'got {samples.shape[1]}'
            }), 400
        
        scaled_samples = model_data['scaler'].transform(samples)
        predictions = model_data['model'].predict(scaled_samples)
        
        return jsonify({
            'predictions': [float(p) for p in predictions],
            'count': len(predictions)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
