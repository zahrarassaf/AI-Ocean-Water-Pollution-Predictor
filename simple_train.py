
import joblib
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ۱. بارگذاری داده‌های پردازش شده تو
data_path = Path("data/processed/marine_pollution_prediction_20251224_184037/processed_data.joblib")
data = joblib.load(data_path)

X_train = data['splits']['X_train']
X_test = data['splits']['X_test']
y_train = data['splits']['y_train']
y_test = data['splits']['y_test']
feature_names = data['feature_names']

print("="*50)
print("📊 DATA SUMMARY")
print("="*50)
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Test samples:     {X_test.shape[0]:,}")
print(f"Features:         {X_train.shape[1]}")
print(f"Feature names:    {feature_names[:5]}...")  # فقط ۵ تا نشون بده

# ۲. آموزش مدل ساده
print("\n" + "="*50)
print("🤖 TRAINING MODEL")
print("="*50)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ۳. ارزیابی
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Model trained successfully!")
print(f"📈 RMSE: {rmse:.4f}")
print(f"📈 R² Score: {r2:.4f}")

# ۴. نمایش اهمیت ویژگی‌ها
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]  # 10 ویژگی مهم

print("\n" + "="*50)
print("🏆 TOP 10 FEATURE IMPORTANCES")
print("="*50)
for i in indices[::-1]:  # از مهم‌ترین به کم‌اهمیت
    print(f"{feature_names[i]:30s}: {importances[i]:.4f}")

# ۵. ذخیره مدل
Path("models").mkdir(exist_ok=True)
model_data = {
    'model': model,
    'metadata': {
        'model_name': 'MarinePollutionPredictor',
        'feature_names': feature_names,
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'performance': {
            'rmse': rmse,
            'r2': r2,
            'mse': mse
        },
        'creation_time': '2025-12-25'
    },
    'scaler': None  # اگر اسکیلر داری اضافه کن
}

model_path = Path("models/marine_model.joblib")
joblib.dump(model_data, model_path, compress=3)

print(f"\n💾 Model saved to: {model_path}")
print("="*50)
print("🎉 DONE! Now you can deploy the model with:")
print("python deploy_model.py serve")
print("="*50)
