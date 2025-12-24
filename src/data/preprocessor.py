"""
Advanced data preprocessing for oceanographic data
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
import logging
from scipy import stats
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
import bottleneck as bn

from ..config.constants import DataConfig
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DataPreprocessor:
    """Advanced preprocessing for marine data"""
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.scalers = {}
        self.imputers = {}
    
    def handle_missing_values(self,
                            dataset: xr.Dataset,
                            method: str = 'interpolate',
                            limit: Optional[int] = None) -> xr.Dataset:
        """
        Handle missing values in dataset
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
        method : str
            Method for handling missing values
            Options: 'interpolate', 'ffill', 'bfill', 'mean', 'median'
        limit : int, optional
            Maximum number of consecutive NaN values to fill
            
        Returns
        -------
        xr.Dataset
            Processed dataset
        """
        processed = dataset.copy()
        
        for var in processed.data_vars:
            data = processed[var]
            nan_count_before = np.isnan(data.values).sum()
            
            if method == 'interpolate':
                # Multi-dimensional interpolation
                processed[var] = data.interpolate_na(
                    dim='time',
                    method=self.config.interpolation_method,
                    limit=limit
                )
            elif method == 'ffill':
                processed[var] = data.ffill(dim='time', limit=limit)
            elif method == 'bfill':
                processed[var] = data.bfill(dim='time', limit=limit)
            elif method == 'mean':
                mean_val = data.mean(skipna=True)
                processed[var] = data.fillna(mean_val)
            elif method == 'median':
                median_val = data.median(skipna=True)
                processed[var] = data.fillna(median_val)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            nan_count_after = np.isnan(processed[var].values).sum()
            filled_count = nan_count_before - nan_count_after
            
            if filled_count > 0:
                logger.info(f"Filled {filled_count} missing values in {var}")
        
        return processed
    
    def detect_outliers(self,
                       dataset: xr.Dataset,
                       method: str = 'iqr',
                       threshold: Optional[float] = None) -> Dict:
        """
        Detect outliers in dataset
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
        method : str
            Outlier detection method
            Options: 'iqr', 'zscore', 'modified_zscore', 'isolation_forest'
        threshold : float, optional
            Outlier threshold
            
        Returns
        -------
        dict
            Outlier information
        """
        if threshold is None:
            threshold = self.config.outlier_threshold
        
        outliers = {
            'counts': {},
            'indices': {},
            'percentages': {}
        }
        
        for var in dataset.data_vars:
            data = dataset[var].values.flatten()
            data = data[~np.isnan(data)]  # Remove NaNs
            
            if method == 'iqr':
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outlier_mask = (data < lower_bound) | (data > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outlier_mask = z_scores > threshold
                
            elif method == 'modified_zscore':
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                outlier_mask = np.abs(modified_z_scores) > threshold
                
            else:
                raise ValueError(f"Unknown method: {method}")
            
            outlier_count = np.sum(outlier_mask)
            outlier_percentage = (outlier_count / len(data)) * 100
            
            outliers['counts'][var] = int(outlier_count)
            outliers['percentages'][var] = float(outlier_percentage)
            
            if outlier_count > 0:
                logger.warning(f"Found {outlier_count} outliers in {var} "
                             f"({outlier_percentage:.2f}%)")
        
        return outliers
    
    def remove_outliers(self,
                       dataset: xr.Dataset,
                       method: str = 'clip',
                       **kwargs) -> xr.Dataset:
        """
        Remove or handle outliers
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
        method : str
            Outlier handling method
            Options: 'clip', 'remove', 'winsorize', 'impute'
            
        Returns
        -------
        xr.Dataset
            Processed dataset
        """
        processed = dataset.copy()
        outlier_info = self.detect_outliers(dataset, **kwargs)
        
        for var in processed.data_vars:
            data = processed[var].values
            
            if method == 'clip':
                # Clip outliers to percentile bounds
                lower_percentile = np.percentile(data[~np.isnan(data)], 1)
                upper_percentile = np.percentile(data[~np.isnan(data)], 99)
                processed[var] = xr.where(
                    processed[var] < lower_percentile,
                    lower_percentile,
                    processed[var]
                )
                processed[var] = xr.where(
                    processed[var] > upper_percentile,
                    upper_percentile,
                    processed[var]
                )
                
            elif method == 'winsorize':
                # Winsorize data
                lower_limit = np.percentile(data[~np.isnan(data)], 5)
                upper_limit = np.percentile(data[~np.isnan(data)], 95)
                processed[var] = xr.where(
                    processed[var] < lower_limit,
                    lower_limit,
                    processed[var]
                )
                processed[var] = xr.where(
                    processed[var] > upper_limit,
                    upper_limit,
                    processed[var]
                )
            
            logger.info(f"Processed outliers in {var} using {method} method")
        
        return processed
    
    def scale_data(self,
                  dataset: xr.Dataset,
                  method: str = None,
                  feature_range: Tuple[float, float] = (0, 1)) -> xr.Dataset:
        """
        Scale dataset features
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
        method : str, optional
            Scaling method
            Options: 'standard', 'minmax', 'robust', 'log', 'quantile'
        feature_range : tuple
            Range for min-max scaling
            
        Returns
        -------
        xr.Dataset
            Scaled dataset
        """
        if method is None:
            method = self.config.scale_method
        
        processed = dataset.copy()
        self.scalers = {}
        
        for var in processed.data_vars:
            data = processed[var].values
            mask = ~np.isnan(data)
            
            if method == 'standard':
                # Standard scaling (zero mean, unit variance)
                mean_val = np.mean(data[mask])
                std_val = np.std(data[mask])
                processed[var].values[mask] = (data[mask] - mean_val) / (std_val + 1e-10)
                self.scalers[var] = {'method': 'standard', 'mean': mean_val, 'std': std_val}
                
            elif method == 'minmax':
                # Min-max scaling
                min_val = np.min(data[mask])
                max_val = np.max(data[mask])
                processed[var].values[mask] = (data[mask] - min_val) / (max_val - min_val + 1e-10)
                processed[var].values[mask] = processed[var].values[mask] * (feature_range[1] - feature_range[0]) + feature_range[0]
                self.scalers[var] = {'method': 'minmax', 'min': min_val, 'max': max_val}
                
            elif method == 'robust':
                # Robust scaling (using median and IQR)
                median_val = np.median(data[mask])
                q1 = np.percentile(data[mask], 25)
                q3 = np.percentile(data[mask], 75)
                iqr = q3 - q1
                processed[var].values[mask] = (data[mask] - median_val) / (iqr + 1e-10)
                self.scalers[var] = {'method': 'robust', 'median': median_val, 'q1': q1, 'q3': q3}
                
            elif method == 'log':
                # Log scaling (for skewed distributions)
                min_val = np.min(data[mask])
                if min_val <= 0:
                    # Add offset to make all values positive
                    offset = -min_val + 1
                    data[mask] = data[mask] + offset
                processed[var].values[mask] = np.log1p(data[mask])
                self.scalers[var] = {'method': 'log'}
                
            elif method == 'quantile':
                # Quantile transformation
                from sklearn.preprocessing import QuantileTransformer
                qt = QuantileTransformer(output_distribution='normal', random_state=42)
                reshaped_data = data[mask].reshape(-1, 1)
                transformed = qt.fit_transform(reshaped_data)
                processed[var].values[mask] = transformed.flatten()
                self.scalers[var] = {'method': 'quantile', 'transformer': qt}
            
            logger.info(f"Scaled {var} using {method} method")
        
        return processed
    
    def create_derived_features(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Create derived features for marine productivity
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
            
        Returns
        -------
        xr.Dataset
            Dataset with derived features
        """
        processed = dataset.copy()
        
        # 1. Euphotic depth (based on light attenuation)
        if 'kd490' in processed:
            # euphotic_depth = ln(1000) / Kd490 ≈ 4.605 / Kd490
            processed['euphotic_depth'] = 4.605 / processed['kd490']
            processed['euphotic_depth'].attrs = {
                'long_name': 'Euphotic depth',
                'units': 'm',
                'description': 'Depth of 1% light penetration'
            }
        
        # 2. Chlorophyll-specific absorption
        if 'chl' in processed and 'kd490' in processed:
            processed['chl_specific_absorption'] = processed['kd490'] / (processed['chl'] + 0.01)
            processed['chl_specific_absorption'].attrs = {
                'long_name': 'Chlorophyll-specific absorption',
                'units': 'm^-1/(mg/m^3)'
            }
        
        # 3. Water clarity index
        if 'zsd' in processed:
            processed['clarity_index'] = 1 / (processed['zsd'] + 0.1)
            processed['clarity_index'].attrs = {
                'long_name': 'Water clarity index',
                'units': 'm^-1'
            }
        
        # 4. Current energy
        if 'current_speed' in processed:
            processed['current_energy'] = 0.5 * processed['current_speed'] ** 2
            processed['current_energy'].attrs = {
                'long_name': 'Current kinetic energy',
                'units': 'm^2/s^2'
            }
        
        # 5. Mixed layer stability (simplified)
        if 'temperature' in processed and 'salinity' in processed:
            # Brunt-Väisälä frequency approximation
            density = self._calculate_density(
                processed['temperature'],
                processed['salinity']
            )
            processed['density'] = density
            processed['density'].attrs = {
                'long_name': 'Seawater density',
                'units': 'kg/m^3'
            }
        
        # 6. Temporal features
        if 'time' in processed.dims:
            # Day of year
            processed['day_of_year'] = processed['time'].dt.dayofyear
            processed['day_of_year'].attrs = {
                'long_name': 'Day of year',
                'units': 'day'
            }
            
            # Seasonal sine/cosine
            processed['season_sin'] = np.sin(2 * np.pi * processed['day_of_year'] / 365.25)
            processed['season_cos'] = np.cos(2 * np.pi * processed['day_of_year'] / 365.25)
        
        logger.info(f"Created {len(processed.data_vars) - len(dataset.data_vars)} derived features")
        
        return processed
    
    def _calculate_density(self, temperature: xr.DataArray, salinity: xr.DataArray) -> xr.DataArray:
        """Calculate seawater density using UNESCO equation of state"""
        # Simplified version - in production, use gsw package
        a0 = 999.842594
        a1 = 6.793952e-2
        a2 = -9.095290e-3
        a3 = 1.001685e-4
        a4 = -1.120083e-6
        a5 = 6.536332e-9
        
        b0 = 8.24493e-1
        b1 = -4.0899e-3
        b2 = 7.6438e-5
        b3 = -8.2467e-7
        b4 = 5.3875e-9
        
        c0 = -5.72466e-3
        c1 = 1.0227e-4
        c2 = -1.6546e-6
        
        d0 = 4.8314e-4
        
        # Density of pure water
        rho_w = (a0 + a1*temperature + a2*temperature**2 + 
                a3*temperature**3 + a4*temperature**4 + a5*temperature**5)
        
        # Density correction for salinity
        rho = (rho_w + 
              (b0 + b1*temperature + b2*temperature**2 + b3*temperature**3 + b4*temperature**4) * salinity +
              (c0 + c1*temperature + c2*temperature**2) * salinity**1.5 +
              d0 * salinity**2)
        
        return rho
    
    def apply_temporal_filters(self,
                              dataset: xr.Dataset,
                              window_size: int = 7,
                              filter_type: str = 'savgol') -> xr.Dataset:
        """
        Apply temporal filters to remove noise
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
        window_size : int
            Filter window size
        filter_type : str
            Filter type: 'savgol', 'moving_avg', 'exponential'
            
        Returns
        -------
        xr.Dataset
            Filtered dataset
        """
        processed = dataset.copy()
        
        for var in processed.data_vars:
            if 'time' in processed[var].dims:
                if filter_type == 'savgol':
                    # Savitzky-Golay filter
                    processed[var] = xr.apply_ufunc(
                        savgol_filter,
                        processed[var],
                        input_core_dims=[['time']],
                        output_core_dims=[['time']],
                        kwargs={
                            'window_length': window_size,
                            'polyorder': 2,
                            'mode': 'interp'
                        },
                        dask='parallelized',
                        output_dtypes=[processed[var].dtype]
                    )
                    
                elif filter_type == 'moving_avg':
                    # Moving average
                    processed[var] = processed[var].rolling(
                        time=window_size,
                        center=True
                    ).mean()
                    
                elif filter_type == 'exponential':
                    # Exponential moving average
                    processed[var] = processed[var].ewm(
                        span=window_size,
                        adjust=True
                    ).mean()
            
            logger.info(f"Applied {filter_type} filter to {var}")
        
        return processed
    
    def get_preprocessing_summary(self) -> Dict:
        """Get summary of preprocessing operations"""
        summary = {
            'scalers_applied': len(self.scalers),
            'scaler_methods': {var: info.get('method', 'unknown') 
                             for var, info in self.scalers.items()},
            'outlier_threshold': self.config.outlier_threshold,
            'interpolation_method': self.config.interpolation_method
        }
        
        return summary
