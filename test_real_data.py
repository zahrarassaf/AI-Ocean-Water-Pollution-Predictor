# src/data_processor.py
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging
import os

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class OceanDataProcessor:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.datasets = {}
        self.processed_data = None
        
    def load_all_netcdf(self) -> Dict:
        """Load all NetCDF files from directory."""
        if not self.data_path.exists():
            logger.error(f"Data path does not exist: {self.data_path}")
            return {}
            
        netcdf_files = list(self.data_path.glob("*.nc"))
        
        if not netcdf_files:
            logger.warning(f"No NetCDF files found in {self.data_path}")
            # Create dummy data for testing
            return self._create_dummy_data()
        
        logger.info(f"Found {len(netcdf_files)} NetCDF files")
        
        for file in netcdf_files:
            try:
                ds = xr.open_dataset(file)
                self.datasets[file.stem] = ds
                logger.info(f"Loaded {file.name}: {list(ds.variables.keys())}")
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")
                
        return self.datasets
    
    def _create_dummy_data(self) -> Dict:
        """Create dummy data for testing when no NetCDF files are found."""
        logger.info("Creating dummy data for testing...")
        
        # Create a dummy dataset
        dummy_ds = xr.Dataset({
            'CHL': (['lat', 'lon', 'time'], 
                   np.random.exponential(1, (10, 10, 5))),
            'KD490': (['lat', 'lon', 'time'], 
                     np.random.uniform(0.02, 1.5, (10, 10, 5))),
            'PROCHLO': (['lat', 'lon', 'time'], 
                       np.random.uniform(0.001, 1.0, (10, 10, 5))),
            'lat': np.linspace(-90, 90, 10),
            'lon': np.linspace(-180, 180, 10),
            'time': pd.date_range('2024-01-01', periods=5)
        })
        
        self.datasets['dummy_data'] = dummy_ds
        return self.datasets
    
    def extract_features(self) -> pd.DataFrame:
        """Extract and merge features from all datasets."""
        if not self.datasets:
            logger.error("No datasets loaded. Call load_all_netcdf() first.")
            return pd.DataFrame()
        
        all_data = []
        
        for name, ds in self.datasets.items():
            logger.info(f"Processing dataset: {name}")
            
            # Extract numerical variables (excluding coordinates)
            for var_name in ds.variables:
                if var_name not in ['time', 'lat', 'lon', 'latitude', 'longitude']:
                    try:
                        # Flatten the data
                        var_data = ds[var_name].values.ravel()
                        
                        # Remove NaN and inf
                        var_data = var_data[~np.isnan(var_data)]
                        var_data = var_data[~np.isinf(var_data)]
                        
                        if len(var_data) > 0:
                            # Take a sample if too large
                            if len(var_data) > 10000:
                                var_data = np.random.choice(var_data, 10000, replace=False)
                            
                            df = pd.DataFrame({
                                'variable': var_name,
                                'value': var_data,
                                'dataset': name
                            })
                            all_data.append(df)
                            logger.info(f"  Added {var_name}: {len(var_data)} points")
                            
                    except Exception as e:
                        logger.warning(f"  Could not process {var_name}: {e}")
                        continue
        
        if not all_data:
            logger.warning("No data extracted. Creating synthetic data.")
            return self._create_synthetic_features()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total data points: {len(combined_df)}")
        
        return self._pivot_features(combined_df)
    
    def _create_synthetic_features(self) -> pd.DataFrame:
        """Create synthetic features for testing."""
        logger.info("Creating synthetic features...")
        
        n_samples = 1000
        data = {
            'CHL': np.random.exponential(1, n_samples),
            'KD490': np.random.uniform(0.02, 1.5, n_samples),
            'PROCHLO': np.random.uniform(0.001, 1.0, n_samples),
            'TEMP': np.random.uniform(10, 30, n_samples),
            'SALINITY': np.random.uniform(30, 38, n_samples)
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Created synthetic data: {df.shape}")
        return df
    
    def _pivot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert long format to wide format with features."""
        try:
            # Group by dataset and variable, take mean
            pivoted = df.groupby(['dataset', 'variable'])['value'].mean().unstack()
            
            # Fill missing values
            pivoted = pivoted.fillna(0)
            
            logger.info(f"Pivoted features: {pivoted.shape}")
            return pivoted.reset_index()
            
        except Exception as e:
            logger.error(f"Error pivoting features: {e}")
            return pd.DataFrame()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        if df.empty:
            logger.warning("Empty dataframe received for cleaning")
            return df
        
        df_clean = df.copy()
        
        # Remove dataset column if exists
        if 'dataset' in df_clean.columns:
            df_clean = df_clean.drop('dataset', axis=1)
        
        # Remove infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Remove outliers using IQR method (only if enough data)
        if len(df_clean) > 10:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Avoid division by zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Keep values within bounds
                    mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                    df_clean = df_clean[mask]
        
        # Log transform skewed features
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if len(df_clean[col].unique()) > 10:  # Only if enough unique values
                skewness = df_clean[col].skew()
                if abs(skewness) > 1:
                    # Add small constant to avoid log(0)
                    df_clean[col] = np.log1p(df_clean[col] - df_clean[col].min() + 1e-6)
        
        logger.info(f"Data cleaned: {len(df_clean)} samples")
        return df_clean
    
    def create_target(self, df: pd.DataFrame, 
                     target_col: str = 'CHL') -> pd.Series:
        """Create target variable based on chlorophyll levels."""
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found in dataframe")
            logger.info(f"Available columns: {df.columns.tolist()}")
            
            # Create synthetic target
            np.random.seed(42)
            target_values = np.random.choice(['LOW', 'MEDIUM', 'HIGH'], len(df))
            return pd.Series(target_values, index=df.index)
        
        def classify_pollution(chl_value: float) -> str:
            if chl_value <= 1.0:
                return 'LOW'
            elif chl_value <= 5.0:
                return 'MEDIUM'
            elif chl_value <= 20.0:
                return 'HIGH'
            else:
                return 'CRITICAL'
        
        target = df[target_col].apply(classify_pollution)
        
        # Print distribution
        logger.info("Target distribution:")
        for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
            count = (target == level).sum()
            if count > 0:
                percentage = count / len(target) * 100
                logger.info(f"  {level}: {count} samples ({percentage:.1f}%)")
        
        return target
    
    def split_data(self, features: pd.DataFrame, target: pd.Series,
                  test_size: float = 0.2, val_size: float = 0.1):
        """Split data into train, validation, and test sets."""
        from sklearn.model_selection import train_test_split
        
        if len(features) < 10:
            logger.warning("Not enough data for splitting. Using all for training.")
            return features, pd.DataFrame(), pd.DataFrame(), target, pd.Series(), pd.Series()
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to parquet format."""
        output_path = Path("data/processed") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path)
        logger.info(f"Data saved to {output_path}")

# تست سریع
if __name__ == "__main__":
    print("🧪 Testing OceanDataProcessor...")
    print("=" * 60)
    
    processor = OceanDataProcessor("data/raw/")
    
    print("1. Loading data...")
    datasets = processor.load_all_netcdf()
    print(f"   Loaded {len(datasets)} datasets")
    
    print("2. Extracting features...")
    features = processor.extract_features()
    print(f"   Features shape: {features.shape}")
    print(f"   Columns: {features.columns.tolist()}")
    
    print("3. Cleaning data...")
    cleaned = processor.clean_data(features)
    print(f"   Cleaned shape: {cleaned.shape}")
    
    print("4. Creating target...")
    if not cleaned.empty:
        target = processor.create_target(cleaned)
        print(f"   Target created: {len(target)} samples")
        
        print("5. Splitting data...")
        splits = processor.split_data(cleaned, target)
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        
        print(f"   Train: {X_train.shape}")
        print(f"   Validation: {X_val.shape}")
        print(f"   Test: {X_test.shape}")
    
    print("\n✅ OceanDataProcessor test completed!")
