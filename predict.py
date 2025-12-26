import joblib
import pandas as pd
import numpy as np

class OceanPollutionPredictor:
    def __init__(self):
        """بارگذاری مدل ذخیره شده"""
        try:
            self.model = joblib.load("models/best_ocean_pollution_model.pkl")
            self.scaler = joblib.load("models/scaler.pkl")
            
            # بارگذاری نام ویژگی‌ها
            with open("models/feature_names.txt", "r") as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            print("✅ Model loaded successfully!")
            
        except FileNotFoundError:
            print("❌ Model files not found. Please run train_model.py first")
            self.model = None
            self.scaler = None
            self.feature_names = None
    
    def predict(self, input_data):
        """پیش‌بینی سطح آلودگی"""
        if self.model is None:
            print("Model not loaded. Cannot predict.")
            return None
        
        # تبدیل ورودی به DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data
        else:
            raise ValueError("Input should be a dictionary or DataFrame")
        
        # اطمینان از ترتیب ستون‌ها
        df = df[self.feature_names]
        
        # نرمال‌سازی
        scaled_data = self.scaler.transform(df)
        
        # پیش‌بینی
        prediction = self.model.predict(scaled_data)[0]
        probabilities = self.model.predict_proba(scaled_data)[0]
        
        # تفسیر نتایج
        pollution_levels = {
            0: "Low Pollution - Water quality is good",
            1: "Medium Pollution - Some contamination detected",
            2: "High Pollution - Significant contamination, action needed"
        }
        
        return {
            'prediction': int(prediction),
            'level_description': pollution_levels[prediction],
            'probabilities': {
                'low': float(probabilities[0]),
                'medium': float(probabilities[1]),
                'high': float(probabilities[2])
            }
        }
    
    def predict_sample(self):
        """پیش‌بینی با داده‌های نمونه"""
        print("\n🔮 Sample Prediction with random data:")
        
        # داده‌های نمونه
        sample_data = {
            'sea_surface_temp': 25.5,
            'salinity': 35.2,
            'turbidity': 5.1,
            'ph': 8.0,
            'dissolved_oxygen': 7.8,
            'nitrate': 3.2,
            'phosphate': 0.8,
            'ammonia': 0.2,
            'chlorophyll_a': 4.5,
            'sechi_depth': 12.3,
            'lead': 0.02,
            'mercury': 0.001,
            'cadmium': 0.005,
            'latitude': 35.6895,
            'longitude': 51.3890,
            'month': 7
        }
        
        result = self.predict(sample_data)
        
        if result:
            print(f"\nPrediction Result:")
            print(f"  Pollution Level: {result['prediction']}")
            print(f"  Status: {result['level_description']}")
            print(f"\nProbabilities:")
            print(f"  Low: {result['probabilities']['low']:.2%}")
            print(f"  Medium: {result['probabilities']['medium']:.2%}")
            print(f"  High: {result['probabilities']['high']:.2%}")
        
        return result

def main():
    """تابع اصلی"""
    print("=" * 50)
    print("OCEAN POLLUTION PREDICTOR")
    print("=" * 50)
    
    # ایجاد predictor
    predictor = OceanPollutionPredictor()
    
    if predictor.model is None:
        return
    
    # پیش‌بینی نمونه
    predictor.predict_sample()
    
    # دستورالعمل برای ورودی دلخواه
    print("\n" + "=" * 50)
    print("To make custom predictions, use:")
    print("\nExample code:")
    print("""
predictor = OceanPollutionPredictor()
custom_data = {
    'sea_surface_temp': 28.0,
    'salinity': 36.5,
    'turbidity': 8.2,
    'ph': 7.9,
    'dissolved_oxygen': 6.5,
    'nitrate': 6.8,
    'phosphate': 1.2,
    'ammonia': 0.4,
    'chlorophyll_a': 8.5,
    'sechi_depth': 5.3,
    'lead': 0.04,
    'mercury': 0.0015,
    'cadmium': 0.008,
    'latitude': 40.7128,
    'longitude': -74.0060,
    'month': 8
}
result = predictor.predict(custom_data)
print(result)
    """)

if __name__ == "__main__":
    main()
