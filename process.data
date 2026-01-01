import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request
import zipfile

print("=" * 60)
print("OCEAN DATA PROCESSOR")
print("=" * 60)

def check_files():
    """بررسی فایل‌های موجود"""
    
    raw_path = "data/raw/"
    processed_path = "data/processed/"
    
    os.makedirs(processed_path, exist_ok=True)
    
    files = os.listdir(raw_path)
    print(f"Files in data/raw/: {files}")
    
    # بررسی سایز فایل‌ها
    for file in files:
        file_path = os.path.join(raw_path, file)
        size_kb = os.path.getsize(file_path) / 1024
        print(f"  {file}: {size_kb:.1f} KB")
    
    return files

def create_sample_ocean_data():
    """ایجاد داده‌های نمونه اقیانوسی"""
    
    print("\nCreating sample ocean water quality data...")
    
    np.random.seed(42)
    n_samples = 2000
    
    # داده‌های نمونه با متغیرهای واقعی
    data = {
        # متغیرهای فیزیکی
        'sea_surface_temp': np.random.uniform(10, 35, n_samples),  # دمای سطح دریا (°C)
        'salinity': np.random.uniform(30, 38, n_samples),  # شوری (PSU)
        'turbidity': np.random.uniform(0.1, 15, n_samples),  # کدورت (NTU)
        
        # متغیرهای شیمیایی
        'ph': np.random.uniform(7.5, 8.4, n_samples),  # اسیدیته
        'dissolved_oxygen': np.random.uniform(4, 12, n_samples),  # اکسیژن محلول (mg/L)
        'nitrate': np.random.uniform(0, 8, n_samples),  # نیترات (mg/L)
        'phosphate': np.random.uniform(0, 1.5, n_samples),  # فسفات (mg/L)
        'ammonia': np.random.uniform(0, 0.5, n_samples),  # آمونیاک (mg/L)
        
        # متغیرهای بیولوژیکی
        'chlorophyll_a': np.random.uniform(0.01, 10, n_samples),  # کلروفیل-a (mg/m³)
        'sechi_depth': np.random.uniform(1, 30, n_samples),  # عمق سچی (متر)
        
        # فلزات سنگین
        'lead': np.random.uniform(0, 0.05, n_samples),  # سرب (mg/L)
        'mercury': np.random.uniform(0, 0.002, n_samples),  # جیوه (mg/L)
        'cadmium': np.random.uniform(0, 0.01, n_samples),  # کادمیوم (mg/L)
        
        # موقعیت جغرافیایی
        'latitude': np.random.uniform(-90, 90, n_samples),
        'longitude': np.random.uniform(-180, 180, n_samples),
        
        # زمان
        'month': np.random.randint(1, 13, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # ایجاد ستون target (سطح آلودگی) بر اساس ترکیبی از پارامترها
    pollution_score = (
        df['chlorophyll_a'] * 0.3 +  # شکوفایی جلبکی
        df['nitrate'] * 0.2 +  # مواد مغذی
        df['phosphate'] * 0.15 +
        df['lead'] * 100 +  # فلزات سنگین (ضریب بالا)
        df['mercury'] * 500 +
        df['ammonia'] * 0.1
    )
    
    # طبقه‌بندی به ۳ سطح
    df['pollution_level'] = pd.qcut(pollution_score, q=3, labels=[0, 1, 2])
    
    # 0: کم (Low), 1: متوسط (Medium), 2: بالا (High)
    
    return df

def process_for_ml(df):
    """پردازش داده‌ها برای یادگیری ماشین"""
    
    processed_path = "data/processed/"
    
    print(f"\nProcessing {len(df)} samples...")
    print(f"Original shape: {df.shape}")
    
    # حذف مقادیر NaN
    df = df.dropna()
    print(f"After removing NaN: {df.shape}")
    
    # جدا کردن features و target
    X = df.drop('pollution_level', axis=1)
    y = df['pollution_level']
    
    # تقسیم داده
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ذخیره
    X_train.to_csv(f"{processed_path}/X_train.csv", index=False)
    X_test.to_csv(f"{processed_path}/X_test.csv", index=False)
    y_train.to_csv(f"{processed_path}/y_train.csv", index=False)
    y_test.to_csv(f"{processed_path}/y_test.csv", index=False)
    
    # همچنین یک فایل کامل ذخیره کنیم
    df.to_csv(f"{processed_path}/full_ocean_data.csv", index=False)
    
    print(f"\n✅ Data processing completed!")
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Target classes: {sorted(y.unique())}")
    print(f"   Class distribution:")
    print(y.value_counts().sort_index())
    
    # نمایش آماری
    print("\n📊 Sample statistics:")
    print(df[['chlorophyll_a', 'nitrate', 'phosphate', 'lead', 'pollution_level']].describe())
    
    return X_train, X_test, y_train, y_test

def main():
    """تابع اصلی"""
    
    print("Starting ocean data processing...\n")
    
    # بررسی فایل‌ها
    files = check_files()
    
    # اگر فایل‌های NetCDF مشکل دارند، از داده‌های نمونه استفاده می‌کنیم
    if files and any(f.endswith('.nc') for f in files):
        print("\n⚠️ NetCDF files detected but may be corrupted.")
        print("Using sample data for now...")
    
    # ایجاد داده‌های نمونه
    ocean_df = create_sample_ocean_data()
    
    # پردازش برای ML
    process_for_ml(ocean_df)
    
    # ایجاد یک فایل README برای توضیح داده‌ها
    create_readme()
    
    print("\n" + "=" * 60)
    print("READY FOR MODEL TRAINING!")
    print("=" * 60)
    print("\nNow run: python train_final.py")

def create_readme():
    """ایجاد فایل توضیحات"""
    
    readme_content = """# Ocean Water Quality Dataset

## Variables Description:

### Physical Parameters:
- sea_surface_temp: Sea surface temperature (°C)
- salinity: Salinity (PSU)
- turbidity: Water turbidity (NTU)

### Chemical Parameters:
- ph: Acidity level
- dissolved_oxygen: Dissolved oxygen (mg/L)
- nitrate: Nitrate concentration (mg/L)
- phosphate: Phosphate concentration (mg/L)
- ammonia: Ammonia concentration (mg/L)

### Biological Parameters:
- chlorophyll_a: Chlorophyll-a concentration (mg/m³)
- sechi_depth: Secchi disk depth (m)

### Heavy Metals:
- lead: Lead concentration (mg/L)
- mercury: Mercury concentration (mg/L)
- cadmium: Cadmium concentration (mg/L)

### Geographical & Temporal:
- latitude: Latitude coordinate
- longitude: Longitude coordinate
- month: Month of observation (1-12)

### Target:
- pollution_level: Pollution level (0=Low, 1=Medium, 2=High)

## Data Source:
This is synthetic data created for AI model training.
For real data, replace with actual ocean monitoring data.

## Usage:
1. Train model: python train_final.py
2. Make predictions: python predict.py
"""
    
    with open("data/processed/DATA_DESCRIPTION.md", "w") as f:
        f.write(readme_content)

if __name__ == "__main__":
    main()
