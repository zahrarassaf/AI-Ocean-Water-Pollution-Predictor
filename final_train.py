import joblib
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import json
from datetime import datetime

# بارگذاری داده‌ها
print("📂 Loading processed data...")
data_path = Path("data/processed/marine_pollution_prediction_20251224_184037/processed_data.joblib")
data = joblib.load(data_path)

X_train = data['splits']['X_train']
X_test = data['splits']['X_test']
y_train = data['splits']['y_train']
y_test = data['splits']['y_test']
feature_names = data['feature_names']

print(f"✅ Data loaded: {X_train.shape[0]:,} training samples")

# آموزش مدل
print("🤖 Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
model.fit(X_train, y_train)

# ارزیابی
print("📊 Evaluating model...")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"🎯 Model Performance:")
print(f"   RMSE: {rmse:.4f}")
print(f"   R²:   {r2:.4f}")

# ذخیره مدل
print("💾 Saving model...")
Path("models").mkdir(exist_ok=True)

model_data = {
    'model': model,
    'metadata': {
        'model_name': 'MarinePollutionPredictor_v1',
        'feature_names': feature_names,
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'performance': {'rmse': rmse, 'r2': r2},
        'creation_time': datetime.now().isoformat(),
        'target_variable': 'KD490',
        'features_count': len(feature_names)
    },
    'feature_importance': dict(zip(feature_names, model.feature_importances_))
}

model_path = Path("models/final_model.joblib")
joblib.dump(model_data, model_path, compress=3)

print(f"✅ Model saved to: {model_path}")
print("🎉 PROJECT COMPLETE! You can now deploy the API.")
