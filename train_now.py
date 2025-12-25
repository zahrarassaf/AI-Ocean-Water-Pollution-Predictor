# فایل: train_now.py
import joblib
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

print("="*60)
print("🎯 TRAINING MODEL ON NEW DATA")
print("="*60)

# مسیر داده‌های جدید
data_path = Path("data/processed/marine_pollution_prediction_20251225_155715/processed_data.joblib")

print(f"📂 Loading data from: {data_path}")
data = joblib.load(data_path)

# استخراج داده‌ها
X_train = data['splits']['X_train']
X_test = data['splits']['X_test']
y_train = data['splits']['y_train']
y_test = data['splits']['y_test']
feature_names = data['feature_names']

print(f"✅ Data loaded:")
print(f"   Training samples: {X_train.shape[0]:,}")
print(f"   Test samples: {X_test.shape[0]:,}")
print(f"   Features: {X_train.shape[1]}")
print(f"   Target: KD490")

# آموزش مدل
print("\n🤖 Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)

# ارزیابی
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 Model Performance:")
print(f"   R² Score: {r2:.4f}")
print(f"   RMSE: {rmse:.4f}")

# ذخیره مدل
Path("models").mkdir(exist_ok=True)
model_data = {
    'model': model,
    'metadata': {
        'model_name': 'MarinePollution_v2',
        'r2_score': r2,
        'rmse': rmse,
        'training_samples': X_train.shape[0],
        'features': X_train.shape[1],
        'feature_names': feature_names,
        'data_version': '20251225_155715'
    }
}

model_path = Path("models/marine_model_v2.joblib")
joblib.dump(model_data, model_path, compress=3)

print(f"\n💾 Model saved to: {model_path}")

# نمایش اهمیت ویژگی‌ها
importances = model.feature_importances_
top_indices = np.argsort(importances)[-10:][::-1]

print("\n🏆 Top 10 Feature Importances:")
for idx in top_indices:
    print(f"   {feature_names[idx]:30s}: {importances[idx]:.4f}")

print("\n" + "="*60)
print("🎉 MODEL TRAINING COMPLETE!")
print("="*60)
print("Next step: Deploy the model:")
print("python deploy_model.py serve --model-path models/marine_model_v2.joblib")
print("="*60)
