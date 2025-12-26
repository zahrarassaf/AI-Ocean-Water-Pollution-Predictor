#!/usr/bin/env python3
"""
Process raw NetCDF files into training data.
"""

import xarray as xr
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

def process_data():
    print("=" * 60)
    print("PROCESSING RAW DATA")
    print("=" * 60)
    
    # List NetCDF files
    raw_dir = Path("data/raw")
    nc_files = list(raw_dir.glob("*.nc"))
    
    print(f"Found {len(nc_files)} NetCDF files:")
    for file in nc_files:
        print(f"  {file.name}")
    
    if not nc_files:
        print("No NetCDF files found!")
        return
    
    # Load and combine datasets
    datasets = []
    for file in nc_files:
        print(f"Loading {file.name}...")
        try:
            ds = xr.open_dataset(file)
            datasets.append(ds)
            print(f"  Variables: {list(ds.data_vars.keys())}")
        except Exception as e:
            print(f"  Error loading {file}: {e}")
    
    if not datasets:
        print("No datasets loaded!")
        return
    
    # Merge datasets
    print("\nMerging datasets...")
    merged = xr.merge(datasets, compat='override')
    print(f"Merged dataset variables: {list(merged.data_vars.keys())}")
    
    # Convert to DataFrame
    print("\nConverting to DataFrame...")
    df = merged.to_dataframe().reset_index()
    print(f"DataFrame shape: {df.shape}")
    
    # Drop columns with coordinates
    coord_cols = ['time', 'lat', 'lon']
    df = df.drop(columns=[col for col in coord_cols if col in df.columns])
    
    # Drop rows with NaN
    df_clean = df.dropna()
    print(f"After dropping NaN: {df_clean.shape}")
    
    if df_clean.empty:
        print("No valid data after cleaning!")
        return
    
    # Prepare features and target
    print("\nPreparing features...")
    all_columns = list(df_clean.columns)
    
    # Try to guess target (look for KD490 or similar)
    possible_targets = ['KD490', 'chl', 'CHL', 'chlorophyll', 'primary_production']
    target_col = None
    
    for col in all_columns:
        if any(target in col.upper() for target in ['KD490', 'CHL', 'CHLOR']):
            target_col = col
            break
    
    if target_col is None and all_columns:
        target_col = all_columns[0]  # Use first column as target
    
    print(f"Using target: {target_col}")
    
    feature_cols = [col for col in all_columns if col != target_col]
    print(f"Using {len(feature_cols)} features")
    
    # Extract X and y
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Save processed data
    processed_dir = Path("data/processed/simple_processing")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols,
        'target_name': target_col,
        'source_files': [str(f) for f in nc_files]
    }
    
    save_path = processed_dir / "processed_data.joblib"
    joblib.dump(processed_data, save_path, compress=3)
    
    print(f"\n✅ Data saved to: {save_path}")
    print("=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    
    return save_path

if __name__ == "__main__":
    process_data()
