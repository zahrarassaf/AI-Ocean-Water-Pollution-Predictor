# فایل جدید: src/data/simple_preprocessor.py
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import xarray as xr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class SimpleDataPreprocessor:
    """A simplified preprocessor for marine data."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def process(self, dataset):
        """Basic processing of xarray dataset."""
        print(f"DEBUG: Processing dataset with variables: {list(dataset.data_vars)}")
        
        # 1. Handle missing values
        dataset = self.handle_missing_values(dataset)
        
        # 2. Normalize data
        dataset = self.normalize_data(dataset)
        
        # 3. Reshape for ML
        processed_data = self.reshape_for_ml(dataset)
        
        return processed_data
    
    def handle_missing_values(self, dataset):
        """Fill missing values using forward fill then backward fill."""
        print("DEBUG: Handling missing values...")
        
        # Forward fill
        dataset = dataset.ffill(dim='time')
        
        # Backward fill for any remaining NaNs
        dataset = dataset.bfill(dim='time')
        
        # Fill any remaining with mean
        for var in dataset.data_vars:
            if dataset[var].isnull().any():
                mean_val = dataset[var].mean().values
                dataset[var] = dataset[var].fillna(mean_val)
        
        return dataset
    
    def normalize_data(self, dataset):
        """Normalize each variable separately."""
        print("DEBUG: Normalizing data...")
        
        for var in dataset.data_vars:
            data = dataset[var].values
            if data.size > 0:
                mean = np.nanmean(data)
                std = np.nanstd(data)
                if std > 0:
                    dataset[var].values = (data - mean) / std
        
        return dataset
    
    def reshape_for_ml(self, dataset):
        """Reshape xarray dataset to 2D array for ML."""
        print("DEBUG: Reshaping for ML...")
        
        # Stack dimensions
        stacked = dataset.stack(sample=('time', 'lat', 'lon'))
        
        # Create feature matrix
        features = []
        feature_names = []
        
        for var in dataset.data_vars:
            if var in stacked:
                var_data = stacked[var].values
                if var_data.ndim == 1:
                    features.append(var_data)
                    feature_names.append(var)
        
        if not features:
            raise ValueError("No valid features found")
        
        X = np.column_stack(features)
        
        # Remove rows with any NaN
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        
        print(f"DEBUG: Final shape: {X.shape}, Features: {feature_names}")
        
        return {
            'X': X,
            'feature_names': feature_names,
            'valid_mask': valid_mask,
            'stacked': stacked
        }
    
    def prepare_features(self, processed_data, target_var=None, feature_vars=None):
        """Prepare features and target for ML."""
        X = processed_data['X']
        feature_names = processed_data['feature_names']
        
        # If no target specified, use first variable
        if target_var is None:
            target_var = feature_names[0]
        
        # If no features specified, use all except target
        if feature_vars is None:
            feature_vars = [f for f in feature_names if f != target_var]
        
        # Find indices
        try:
            target_idx = feature_names.index(target_var)
        except ValueError:
            raise ValueError(f"Target variable '{target_var}' not found in features")
        
        feature_indices = []
        selected_features = []
        for f in feature_vars:
            try:
                idx = feature_names.index(f)
                feature_indices.append(idx)
                selected_features.append(f)
            except ValueError:
                print(f"Warning: Feature '{f}' not found in dataset")
        
        if not feature_indices:
            raise ValueError("No valid feature variables found")
        
        # Split features and target
        y = X[:, target_idx]
        X_features = X[:, feature_indices]
        
        print(f"DEBUG: X shape: {X_features.shape}, y shape: {y.shape}")
        print(f"DEBUG: Features: {selected_features}, Target: {target_var}")
        
        return X_features, y, selected_features
