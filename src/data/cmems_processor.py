"""
CMEMS data processor for marine productivity
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

# CORRECTED: Direct logging setup instead of import
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if no handlers exist
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class CMEMSDataProcessor:
    """Process CMEMS data for marine productivity analysis"""
    
    def __init__(self, data_dir: Union[str, Path], config: Optional[Dict] = None):
        self.data_dir = Path(data_dir)
        self.config = config or {}
        self.datasets = {}
        
        logger.info(f"Initialized CMEMSDataProcessor with data_dir: {data_dir}")
    
    def load_dataset(self, filename: str, variables: Optional[List[str]] = None) -> xr.Dataset:
        """Load CMEMS NetCDF dataset"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading dataset: {filename}")
        
        try:
            # Load dataset
            ds = xr.open_dataset(filepath, engine='netcdf4')
            
            # Select specific variables if requested
            if variables:
                available_vars = [v for v in variables if v in ds]
                missing_vars = set(variables) - set(available_vars)
                
                if missing_vars:
                    logger.warning(f"Missing variables: {missing_vars}")
                
                ds = ds[available_vars]
            
            # Store dataset
            self.datasets[filename] = ds
            
            logger.info(f"Successfully loaded {filename}")
            logger.info(f"Dimensions: {dict(ds.dims)}")
            logger.info(f"Variables: {list(ds.data_vars)}")
            
            return ds
            
        except Exception as e:
            logger.error(f"Error loading dataset {filename}: {e}")
            raise
    
    def extract_primary_productivity(self, ds: xr.Dataset, method: str = 'vgpm') -> xr.DataArray:
        """Extract primary productivity using various algorithms"""
        
        if method == 'vgpm':
            # Vertically Generalized Production Model (VGPM)
            if 'chl' not in ds:
                raise ValueError("Chlorophyll data required for VGPM method")
            
            # Simplified VGPM calculation
            pp = ds['chl'] * 1000  # Simple conversion
            
            # Add attributes
            pp.attrs = {
                'long_name': 'Primary Productivity',
                'units': 'mg C m^-2 day^-1',
                'method': 'VGPM',
                'description': 'Vertically Generalized Production Model'
            }
            
        elif method == 'cafe':
            # Carbon-based Productivity Model (CbPM)
            if 'chl' not in ds or 'sst' not in ds:
                raise ValueError("Chlorophyll and SST data required for CbPM method")
            
            # Simplified CbPM calculation
            pp = ds['chl'] * ds['sst'] * 50  # Placeholder
            
            pp.attrs = {
                'long_name': 'Primary Productivity',
                'units': 'mg C m^-2 day^-1',
                'method': 'CbPM',
                'description': 'Carbon-based Productivity Model'
            }
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Extracted primary productivity using {method} method")
        
        return pp
    
    def calculate_euphotic_depth(self, ds: xr.Dataset) -> xr.DataArray:
        """Calculate euphotic depth from light attenuation"""
        
        if 'kd490' not in ds:
            raise ValueError("kd490 (diffuse attenuation coefficient) required")
        
        # Euphotic depth: depth where light is 1% of surface
        euphotic_depth = 4.605 / ds['kd490']
        
        # Clip unrealistic values
        euphotic_depth = euphotic_depth.where(euphotic_depth > 0, 0)
        euphotic_depth = euphotic_depth.where(euphotic_depth < 200, 200)
        
        euphotic_depth.attrs = {
            'long_name': 'Euphotic Depth',
            'units': 'm',
            'description': 'Depth of 1% light penetration',
            'formula': 'Z_eu = ln(100) / Kd490'
        }
        
        logger.info("Calculated euphotic depth")
        
        return euphotic_depth
    
    def create_feature_dataset(self, ds: xr.Dataset, target_var: str = 'pp') -> xr.Dataset:
        """Create feature dataset for machine learning"""
        
        features = xr.Dataset()
        
        # 1. Basic features
        basic_vars = ['chl', 'kd490', 'zsd', 'sst', 'sss']
        for var in basic_vars:
            if var in ds:
                features[var] = ds[var]
        
        # 2. Derived features
        if 'kd490' in ds:
            features['euphotic_depth'] = self.calculate_euphotic_depth(ds)
        
        # 3. Target variable
        if target_var in ds:
            features[target_var] = ds[target_var]
        else:
            # Calculate primary productivity if not present
            logger.info(f"Target variable {target_var} not found, calculating...")
            features[target_var] = self.extract_primary_productivity(ds)
        
        # 4. Temporal features (if time dimension exists)
        if 'time' in features.dims:
            features['day_of_year'] = features['time'].dt.dayofyear
            features['month'] = features['time'].dt.month
            features['season'] = ((features['time'].dt.month % 12 + 3) // 3).astype(int)
        
        # 5. Spatial features (if lat/lon dimensions exist)
        if 'lat' in features.dims:
            features['abs_latitude'] = np.abs(features['lat'])
        
        logger.info(f"Created feature dataset with {len(features.data_vars)} variables")
        
        return features
    
    def prepare_ml_data(self, 
                       features_ds: xr.Dataset,
                       target_var: str = 'pp',
                       feature_vars: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for machine learning"""
        
        if feature_vars is None:
            # Default feature set
            feature_vars = [
                'chl', 'kd490', 'zsd', 'euphotic_depth',
                'day_of_year', 'month', 'season', 'abs_latitude'
            ]
        
        # Filter available features
        available_features = [var for var in feature_vars if var in features_ds]
        
        # Prepare X matrix
        X_arrays = []
        for var in available_features:
            data = features_ds[var].values.flatten()
            X_arrays.append(data)
        
        X = np.column_stack(X_arrays) if X_arrays else np.array([])
        
        # Prepare y vector
        if target_var not in features_ds:
            raise ValueError(f"Target variable {target_var} not in dataset")
        
        y = features_ds[target_var].values.flatten()
        
        # Remove NaN values
        valid_mask = ~np.isnan(y)
        if X.size > 0:
            valid_mask = valid_mask & ~np.any(np.isnan(X), axis=1)
        
        X_clean = X[valid_mask] if X.size > 0 else np.array([])
        y_clean = y[valid_mask]
        
        logger.info(f"Prepared ML data: {len(y_clean)} samples, {len(available_features)} features")
        
        return X_clean, y_clean, available_features
    
    def save_processed_data(self, 
                           dataset: xr.Dataset, 
                           output_path: Union[str, Path],
                           format: str = 'netcdf'):
        """Save processed dataset"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'netcdf':
            dataset.to_netcdf(output_path)
            logger.info(f"Saved dataset to NetCDF: {output_path}")
        
        elif format == 'zarr':
            dataset.to_zarr(output_path, consolidated=True)
            logger.info(f"Saved dataset to Zarr: {output_path}")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def close(self):
        """Close all open datasets"""
        for ds in self.datasets.values():
            ds.close()
        
        self.datasets.clear()
        logger.info("Closed all datasets")
