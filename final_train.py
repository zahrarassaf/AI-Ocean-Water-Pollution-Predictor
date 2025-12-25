# فایل: train_final.py
import joblib
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

print("="*60)
print("🚀 FINAL MODEL TRAINING")
print("="*60)

# مسیر داده‌ها
data_path = Path("data/processed/marine_pollution_prediction_20251225_155715/processed_data.joblib")

# بارگذاری داده
print(f"📂 Loading: {data_path}")
data = joblib.load(data_path)

# استخراج
X_train = data['splits']['X_train']
X_test = data['splits']['X_test']
y_train = data['splits']['y_train']
y_test = data['splits']['y_test']

print(f"✅ Data: {X_train.shape[0]:,} train, {X_test.shape[0]:,} test")

# آموزش
print("🤖 Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ارزیابی
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print(f"📊 R² Score: {score:.4f}")

# ذخیره
Path("models").mkdir(exist_ok=True)
model_data = {'model': model, 'score': score}
joblib.dump(model_data, "models/final_model.joblib")

print("💾 Model saved: models/final_model.joblib")
print("="*60)
print("🎉 DONE! Next:")
print("python deploy_model.py serve")
print("="*60)
