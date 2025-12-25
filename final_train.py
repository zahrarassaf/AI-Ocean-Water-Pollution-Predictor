#!/usr/bin/env python3
"""
train_final.py - Final training script for marine pollution prediction.
"""

import joblib
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import sys

def main():
    print("=" * 60)
    print("FINAL MODEL TRAINING")
    print("=" * 60)
    
    # Correct path - use your latest data
    data_path = Path("data/processed/marine_pollution_prediction_20251225_161035/processed_data.joblib")
    
    if not data_path.exists():
        print(f"ERROR: File not found: {data_path}")
        print("Available files:")
        for f in Path("data/processed").glob("*/*.joblib"):
            print(f"  {f}")
        sys.exit(1)
    
    print(f"Loading: {data_path}")
    data = joblib.load(data_path)
    
    # Extract data splits
    X_train = data['splits']['X_train']
    X_test = data['splits']['X_test']
    y_train = data['splits']['y_train']
    y_test = data['splits']['y_test']
    
    print(f"Data loaded: {X_train.shape[0]:,} training samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Test samples: {X_test.shape[0]:,}")
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nModel Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    # Save model
    Path("models").mkdir(exist_ok=True)
    model_data = {
        'model': model,
        'metadata': {
            'model_name': 'MarinePollutionPredictor',
            'r2_score': r2,
            'rmse': rmse,
            'training_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'features': X_train.shape[1],
            'data_version': '20251225_161035'
        }
    }
    
    model_path = Path("models/final_model.joblib")
    joblib.dump(model_data, model_path, compress=3)
    
    print(f"\nModel saved: {model_path}")
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("Next steps:")
    print("1. Install API: pip install fastapi uvicorn pydantic")
    print("2. Run API: python deploy_model.py serve --model-path models/final_model.joblib")
    print("=" * 60)

if __name__ == "__main__":
    main()
