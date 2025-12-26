import joblib
import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime

def test_model(model_path=None):
    print("=" * 60)
    print("MODEL TESTING")
    print("=" * 60)
    
    if model_path is None:
        model_dirs = glob.glob('models/final_model_*')
        if not model_dirs:
            print("No models found. Train a model first.")
            return
        
        model_dirs.sort(key=os.path.getmtime, reverse=True)
        model_path = model_dirs[0] + '/final_model.joblib'
    
    print(f"Loading model from: {model_path}")
    
    try:
        model_data = joblib.load(model_path)
        
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        metrics = model_data['metrics']
        
        print(f"\nMODEL INFORMATION:")
        print(f"  Training date: {model_data.get('training_date', 'N/A')}")
        print(f"  Number of features: {len(feature_names)}")
        print(f"  Model type: {type(model).__name__}")
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Test R²: {metrics.get('test_r2', metrics.get('r2', 0)):.4f}")
        print(f"  Test RMSE: {metrics.get('test_rmse', metrics.get('rmse', 0)):.4f}")
        print(f"  Test MAE: {metrics.get('test_mae', metrics.get('mae', 0)):.4f}")
        
        if 'train_r2' in metrics:
            print(f"  Train R²: {metrics['train_r2']:.4f}")
        
        print(f"\nFEATURE NAMES (first 10):")
        for i, name in enumerate(feature_names[:10]):
            print(f"  {i+1:2d}. {name}")
        
        print(f"\nTESTING WITH SAMPLE DATA:")
        
        np.random.seed(42)
        
        print("\n1. Test with zero values (mean):")
        sample_zero = np.zeros(len(feature_names))
        sample_scaled = scaler.transform([sample_zero])
        prediction = model.predict(sample_scaled)
        print(f"   Prediction: {prediction[0]:.6f}")
        
        print("\n2. Test with random values:")
        for i in range(3):
            sample_random = np.random.randn(len(feature_names))
            sample_scaled = scaler.transform([sample_random])
            prediction = model.predict(sample_scaled)
            print(f"   Sample {i+1}: {prediction[0]:.6f}")
        
        print("\n3. Test with actual test data (if available):")
        if 'X_test' in model_data and 'y_test' in model_data:
            X_test = model_data['X_test']
            y_test = model_data['y_test']
            
            if len(X_test) > 0:
                X_test_scaled = scaler.transform(X_test[:3])
                predictions = model.predict(X_test_scaled)
                
                for i in range(min(3, len(predictions))):
                    actual = y_test[i]
                    predicted = predictions[i]
                    error = abs(actual - predicted)
                    print(f"   Sample {i+1}: Actual={actual:.6f}, "
                          f"Predicted={predicted:.6f}, Error={error:.6f}")
        
        print("\n4. Model validation:")
        print(f"   Number of trees: {model.n_estimators}")
        print(f"   Max depth: {model.max_depth}")
        print(f"   Feature importance sum: {model.feature_importances_.sum():.4f}")
        
        test_results = {
            'test_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'model_path': model_path,
            'metrics': metrics,
            'feature_count': len(feature_names),
            'model_type': type(model).__name__,
            'sample_predictions': {
                'zero_input': float(prediction[0]),
                'random_samples': [float(p) for p in model.predict(
                    scaler.transform(np.random.randn(3, len(feature_names))))
                ]
            }
        }
        
        print(f"\n" + "=" * 60)
        print("MODEL TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return test_results
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained model')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--save', action='store_true', help='Save test results')
    
    args = parser.parse_args()
    
    results = test_model(args.model)
    
    if args.save and results:
        save_path = f"test_results_{results['test_date']}.json"
        import json
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {save_path}")

if __name__ == "__main__":
    main()
