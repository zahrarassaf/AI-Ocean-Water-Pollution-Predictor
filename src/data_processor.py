import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class OceanDataProcessor:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.datasets = {}
        self.processed_data = None
        
    def load_all_netcdf(self) -> Dict:
        """Load all NetCDF files from directory."""
        netcdf_files = list(self.data_path.glob("*.nc"))
        
        for file in netcdf_files:
            try:
                ds = xr.open_dataset(file)
                self.datasets[file.stem] = ds
                logger.info(f"Loaded {file.name}: {len(ds.variables)} variables")
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")
                
        return self.datasets
    
    def extract_features(self) -> pd.DataFrame:
        """Extract and merge features from all datasets."""
        all_data = []
        
        for name, ds in self.datasets.items():
            for var_name in ds.variables:
                if var_name not in ['time', 'lat', 'lon']:
                    try:
                        var_data = ds[var_name].values.ravel()
                        df = pd.DataFrame({
                            'variable': var_name,
                            'value': var_data,
                            'dataset': name
                        })
                        all_data.append(df)
                    except:
                        continue
        
        if not all_data:
            raise ValueError("No data extracted from NetCDF files")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        return self._pivot_features(combined_df)
    
    def _pivot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert long format to wide format with features."""
        pivoted = df.pivot_table(
            index=['dataset'],
            columns='variable',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        pivoted.columns.name = None
        return pivoted.fillna(0)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        df_clean = df.copy()
        
        # Remove infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values
        df_clean = df_clean.fillna(df_clean.median())
        
        # Remove outliers using IQR method
        Q1 = df_clean.quantile(0.25)
        Q3 = df_clean.quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_mask = ~((df_clean < (Q1 - 1.5 * IQR)) | 
                         (df_clean > (Q3 + 1.5 * IQR))).any(axis=1)
        df_clean = df_clean[outlier_mask]
        
        # Log transform skewed features
        skewed_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in skewed_cols:
            if df_clean[col].skew() > 1:
                df_clean[col] = np.log1p(df_clean[col])
        
        logger.info(f"Data cleaned: {len(df_clean)} samples")
        return df_clean
    
    def create_target(self, df: pd.DataFrame, 
                     target_col: str = 'CHL') -> pd.Series:
        """Create target variable based on chlorophyll levels."""
        def classify_pollution(chl_value: float) -> str:
            if chl_value <= 1.0:
                return 'LOW'
            elif chl_value <= 5.0:
                return 'MEDIUM'
            elif chl_value <= 20.0:
                return 'HIGH'
            else:
                return 'CRITICAL'
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        target = df[target_col].apply(classify_pollution)
        logger.info(f"Target distribution:\n{target.value_counts()}")
        
        return target
    
    def split_data(self, features: pd.DataFrame, target: pd.Series,
                  test_size: float = 0.2, val_size: float = 0.1):
        """Split data into train, validation, and test sets."""
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42, stratify=target
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to parquet format."""
        output_path = Path("data/processed") / filename
        df.to_parquet(output_path)
        logger.info(f"Data saved to {output_path}")
