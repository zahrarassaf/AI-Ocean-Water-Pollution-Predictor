"""
Test NetCDF files and create model
"""

import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def check_netcdf_files():
    print("Checking NetCDF files in data/raw/")
    
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("❌ data/raw directory not found")
        return False
    
    nc_files = list(data_dir.glob("*.nc"))
    if not nc_files:
        print("❌ No .nc files found in data/raw/")
        return False
    
    print(f"✅ Found {len(nc_files)} NetCDF files:")
    for f in nc_files:
        print(f"  - {f.name}")
    
    return True

def create_model_from_scratch():
    """Create a model using scientific thresholds"""
    print("\nCreating model using scientific thresholds...")
    
    np.random.seed(42)
    n_samples = 10000
    
    chl_data = []
    pp_data = []
    trans_data = []
    labels = []
    
    for i in range(n_samples):
        if i < 6000:
            chl = np.random.uniform(0.1, 1.0)
            label = 0
        elif i < 9000:
            chl = np.random.uniform(1.0, 5.0)
            label = 1
        else:
            chl = np.random.uniform(5.0, 20.0)
            label = 2
        
        pp = chl * 80 + np.random.normal(0, 30)
        trans = max(1.0, 30 - (chl * 3) + np.random.normal(0, 2))
        
        chl_data.append(chl)
        pp_data.append(pp)
        trans_data.append(trans)
        labels.append(label)
    
    X = np.column_stack([chl_data, pp_data, trans_data])
    y = np.array(labels)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_scaled, y)
    
    accuracy = model.score(X_scaled, y)
    print(f"✅ Model created with accuracy: {accuracy:.2%}")
    
    features = ['CHL', 'PP', 'TRANS']
    
    save_model(model, scaler, features)
    
    return model, scaler, features

def save_model(model, scaler, features):
    Path("models").mkdir(exist_ok=True)
    
    joblib.dump(model, 'models/smart_ocean_model.pkl')
    joblib.dump(scaler, 'models/smart_scaler.pkl')
    
    with open('models/smart_features.txt', 'w') as f:
        for feat in features:
            f.write(f"{feat}\n")
    
    print("✅ Model saved to models/ directory")

def main():
    print("="*60)
    print("OCEAN POLLUTION MODEL SETUP")
    print("="*60)
    
    has_netcdf = check_netcdf_files()
    
    if has_netcdf:
        try:
            import xarray as xr
            
            print("\nTrying to read NetCDF files...")
            data_dir = Path("data/raw")
            nc_files = list(data_dir.glob("*.nc"))
            
            for nc_file in nc_files[:1]:
                try:
                    print(f"\nReading {nc_file.name}...")
                    ds = xr.open_dataset(nc_file)
                    print(f"Variables in file: {list(ds.variables.keys())}")
                    ds.close()
                except Exception as e:
                    print(f"Error reading {nc_file.name}: {e}")
        
        except ImportError:
            print("xarray not installed. Run: pip install xarray netcdf4")
    
    create_model_from_scratch()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("Now run: python predict.py")
    print("="*60)

if __name__ == "__main__":
    main()
