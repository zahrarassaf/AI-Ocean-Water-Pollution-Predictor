# train_real_data.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import logging
import xarray as xr
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_process_netcdf():
    """Load and process NetCDF files directly"""
    logger.info("Loading NetCDF files from data/raw/")
    
    raw_files = glob.glob("data/raw/*.nc")
    if not raw_files:
        logger.error("No NetCDF files found in data/raw/")
        return None, None
    
    logger.info(f"Found {len(raw_files)} NetCDF files")
    
    all_data = []
    for file_path in raw_files:
        try:
            logger.info(f"Processing: {os.path.basename(file_path)}")
            ds = xr.open_dataset(file_path)
            
            # Convert to DataFrame
            df = ds.to_dataframe().reset_index()
            
            # Keep only numeric columns
            df = df.select_dtypes(include=[np.number])
            
            # Remove columns with all NaN
            df = df.dropna(axis=1, how='all')
            
            if not df.empty:
                all_data.append(df)
                logger.info(f"  Added {len(df)} samples, {df.shape[1]} features")
            
            ds.close()
            
        except Exception as e:
            logger.warning(f"  Error processing {file_path}: {str(e)[:100]}")
            continue
    
    if not all_data:
        logger.error("Could not process any NetCDF files")
        return None, None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True, sort=False)
    
    # Fill NaN values with column mean
    combined_df = combined_df.fillna(combined_df.mean())
    
    logger.info(f"Combined dataset: {combined_df.shape[0]} samples, {combined_df.shape[1]} features")
    
    if combined_df.shape[0] < 100:
        logger.error("Not enough data samples")
        return None, None
    
    # Create target variable (use first column as target)
    target_col = combined_df.columns[0]
    X = combined_df.drop(columns=[target_col])
    y = combined_df[target_col]
    
    logger.info(f"Target variable: {target_col}")
    logger.info(f"Features: {list(X.columns)[:10]}...")  # Show first 10 features
    
    return X, y

def train_model(X, y):
    """Train machine learning model"""
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Determine if regression or classification
    is_classification = len(np.unique(y)) < 10  # If less than 10 unique values, do classification
    
    if is_classification:
        logger.info("Training Classification model...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
    else:
        logger.info("Training Regression model...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"R² Score: {r2:.4f}")
    
    return model, scaler, X.columns.tolist()

def save_model_components(model, scaler, feature_names):
    """Save model and components"""
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/ocean_model.pkl')
    logger.info("Saved model: models/ocean_model.pkl")
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    logger.info("Saved scaler: models/scaler.pkl")
    
    # Save feature names
    with open('models/feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    logger.info(f"Saved {len(feature_names)} feature names")
    
    # Save sample data for testing
    sample_data = {
        'feature_names': feature_names,
        'model_type': type(model).__name__
    }
    joblib.dump(sample_data, 'models/model_info.pkl')

def main():
    logger.info("=" * 60)
    logger.info("TRAINING WITH REAL NETCDF DATA")
    logger.info("=" * 60)
    
    # Load and process NetCDF data
    X, y = load_and_process_netcdf()
    if X is None or y is None:
        logger.error("Failed to load data")
        return False
    
    # Train model
    model, scaler, feature_names = train_model(X, y)
    
    # Save everything
    save_model_components(model, scaler, feature_names)
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
