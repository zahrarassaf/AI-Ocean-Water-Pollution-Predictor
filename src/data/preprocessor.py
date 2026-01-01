"""
Advanced data preprocessing for marine NetCDF datasets.
Professional grade with comprehensive error handling and validation.
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
import logging
from scipy import stats
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass


class DataValidator:
    """Validate data quality and consistency."""
    
    @staticmethod
    def validate_dataset(ds: xr.Dataset) -> Dict[str, Any]:
        """Validate dataset structure and quality."""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Check required dimensions
            required_dims = ['time', 'lat', 'lon']
            missing_dims = [dim for dim in required_dims if dim not in ds.dims]
            if missing_dims:
                validation['issues'].append(f"Missing required dimensions: {missing_dims}")
                validation['is_valid'] = False
            
            # Check variable consistency
            for var_name in ds.data_vars:
                var = ds[var_name]
                
                # Check for NaN values
                nan_count = np.isnan(var.values).sum()
                nan_percentage = (nan_count / var.values.size) * 100
                
                validation['statistics'][var_name] = {
                    'nan_count': int(nan_count),
                    'nan_percentage': float(nan_percentage),
                    'shape': var.shape,
                    'dtype': str(var.dtype),
                    'has_fill_value': '_FillValue' in var.attrs or 'missing_value' in var.attrs
                }
                
                if nan_percentage > 50:
                    validation['warnings'].append(
                        f"High NaN percentage in {var_name}: {nan_percentage:.1f}%"
                    )
                
                # Check for unrealistic values
                if var_name in ['chl', 'kd490', 'zsd', 'pp']:
                    data_flat = var.values.flatten()
                    data_clean = data_flat[~np.isnan(data_flat)]
                    
                    if len(data_clean) > 0:
                        q1 = np.percentile(data_clean, 25)
                        q3 = np.percentile(data_clean, 75)
                        iqr = q3 - q1
                        lower_bound = q1 - 3 * iqr
                        upper_bound = q3 + 3 * iqr
                        
                        outliers = ((data_clean < lower_bound) | (data_clean > upper_bound)).sum()
                        outlier_percentage = (outliers / len(data_clean)) * 100
                        
                        if outlier_percentage > 10:
                            validation['warnings'].append(
                                f"High outlier percentage in {var_name}: {outlier_percentage:.1f}%"
                            )
            
            # Check temporal consistency
            if 'time' in ds.dims:
                time_diff = np.diff(ds.time.values)
                if hasattr(time_diff[0], 'days'):
                    days_diff = [td.days for td in time_diff]
                    if len(set(days_diff)) > 1:
                        validation['warnings'].append("Irregular time intervals detected")
            
            # Check spatial consistency
            if 'lat' in ds.dims and 'lon' in ds.dims:
                lat_diff = np.diff(ds.lat.values)
                lon_diff = np.diff(ds.lon.values)
                
                if not np.allclose(lat_diff, lat_diff[0], rtol=1e-3):
                    validation['warnings'].append("Non-uniform latitude grid")
                if not np.allclose(lon_diff, lon_diff[0], rtol=1e-3):
                    validation['warnings'].append("Non-uniform longitude grid")
            
            logger.info(f"Dataset validation completed: {len(validation['issues'])} issues, "
                       f"{len(validation['warnings'])} warnings")
            
        except Exception as e:
            validation['is_valid'] = False
            validation['issues'].append(f"Validation error: {str(e)}")
            logger.error(f"Dataset validation failed: {e}")
        
        return validation


class DataPreprocessor:
    """
    Professional data preprocessor for marine NetCDF datasets.
    
    Features:
    - Comprehensive missing value handling
    - Advanced outlier detection and removal
    - Multiple scaling methods
    - Temporal and spatial smoothing
    - Derived feature creation
    - Data validation and quality control
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or {
            'missing_value_method': 'interpolate',
            'outlier_method': 'iqr',
            'outlier_threshold': 3.0,
            'scaling_method': 'standard',
            'temporal_filter': 'savgol',
            'temporal_window': 7,
            'spatial_smoothing': True,
            'create_derived_features': True,
            'validate_output': True
        }
        
        self.validator = DataValidator()
        self.scalers = {}
        self.imputers = {}
        self.feature_names = []
        
        logger.info(f"Initialized DataPreprocessor with config: {self.config}")
    
    def process(self, dataset: xr.Dataset, target_var: str = 'pp') -> xr.Dataset:
        """
        Complete preprocessing pipeline.
        
        Args:
            dataset: Input xarray Dataset
            target_var: Target variable name
            
        Returns:
            Preprocessed xarray Dataset
        """
        logger.info("Starting complete preprocessing pipeline")
        
        try:
            # Step 1: Validation
            validation = self.validator.validate_dataset(dataset)
            if not validation['is_valid'] and validation['issues']:
                logger.warning(f"Dataset has issues: {validation['issues']}")
            
            # Step 2: Handle missing values
            logger.info("Step 1: Handling missing values")
            dataset = self.handle_missing_values(dataset)
            
            # Step 3: Detect and handle outliers
            logger.info("Step 2: Handling outliers")
            outlier_info = self.detect_outliers(dataset)
            dataset = self.remove_outliers(dataset)
            
            # Step 4: Scale features (but not target)
            logger.info("Step 3: Scaling features")
            if target_var in dataset:
                target_data = dataset[target_var].copy()
                dataset = dataset.drop_vars([target_var])
            
            dataset = self.scale_data(dataset)
            
            # Add target back
            if target_var in locals() and 'target_data' in locals():
                dataset[target_var] = target_data
            
            # Step 5: Create derived features
            if self.config.get('create_derived_features', True):
                logger.info("Step 4: Creating derived features")
                dataset = self.create_derived_features(dataset)
            
            # Step 6: Apply temporal filters
            if self.config.get('temporal_filter'):
                logger.info("Step 5: Applying temporal filters")
                dataset = self.apply_temporal_filters(dataset)
            
            # Step 7: Final validation
            if self.config.get('validate_output', True):
                final_validation = self.validator.validate_dataset(dataset)
                logger.info(f"Final validation: {len(final_validation['issues'])} issues")
            
            logger.info(f"Preprocessing completed. Final dataset: {len(dataset.data_vars)} variables")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {e}")
            raise PreprocessingError(f"Preprocessing failed: {e}")
    
    def handle_missing_values(
        self,
        dataset: xr.Dataset,
        method: Optional[str] = None,
        limit: Optional[int] = None
    ) -> xr.Dataset:
        """
        Handle missing values in dataset.
        
        Args:
            dataset: Input dataset
            method: Interpolation method ('linear', 'nearest', 'cubic', 'ffill', 'bfill', 'mean', 'median')
            limit: Maximum number of consecutive NaN values to fill
            
        Returns:
            Dataset with missing values handled
        """
        method = method or self.config.get('missing_value_method', 'interpolate')
        limit = limit or self.config.get('missing_value_limit', 3)
        
        processed = dataset.copy()
        total_filled = 0
        
        for var_name in processed.data_vars:
            var = processed[var_name]
            nan_before = np.isnan(var.values).sum()
            
            if nan_before == 0:
                continue
            
            try:
                if method == 'interpolate':
                    # Multi-dimensional interpolation
                    if 'time' in var.dims:
                        var = var.interpolate_na(
                            dim='time',
                            method='linear',
                            limit=limit,
                            use_coordinate=True
                        )
                    
                    # Spatial interpolation if still NaN
                    if np.isnan(var.values).any() and all(dim in var.dims for dim in ['lat', 'lon']):
                        var = self._spatial_interpolation(var)
                
                elif method == 'ffill':
                    var = var.ffill(dim='time', limit=limit)
                
                elif method == 'bfill':
                    var = var.bfill(dim='time', limit=limit)
                
                elif method == 'mean':
                    mean_val = var.mean(skipna=True)
                    var = var.fillna(mean_val)
                
                elif method == 'median':
                    median_val = var.median(skipna=True)
                    var = var.fillna(median_val)
                
                elif method == 'knn':
                    var = self._knn_imputation(var)
                
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                nan_after = np.isnan(var.values).sum()
                filled = nan_before - nan_after
                total_filled += filled
                
                processed[var_name] = var
                
                if filled > 0:
                    logger.debug(f"Filled {filled} missing values in {var_name}")
            
            except Exception as e:
                logger.warning(f"Could not handle missing values in {var_name}: {e}")
                continue
        
        if total_filled > 0:
            logger.info(f"Total missing values filled: {total_filled}")
        
        return processed
    
    def _spatial_interpolation(self, data_array: xr.DataArray) -> xr.DataArray:
        """Perform spatial interpolation for 2D fields."""
        try:
            # Create grid
            lon_grid, lat_grid = np.meshgrid(data_array.lon.values, data_array.lat.values)
            
            # Flatten arrays
            points = np.column_stack([lon_grid.flatten(), lat_grid.flatten()])
            values = data_array.values.flatten()
            
            # Find non-NaN points
            valid_mask = ~np.isnan(values)
            
            if valid_mask.sum() < 10:  # Not enough points for interpolation
                return data_array
            
            # Interpolate using nearest neighbor
            tree = KDTree(points[valid_mask])
            distances, indices = tree.query(points[~valid_mask], k=1)
            
            # Fill NaN values
            values_interp = values.copy()
            values_interp[~valid_mask] = values[valid_mask][indices]
            
            # Reshape back to original shape
            result = data_array.copy()
            result.values = values_interp.reshape(data_array.shape)
            
            return result
            
        except Exception as e:
            logger.warning(f"Spatial interpolation failed: {e}")
            return data_array
    
    def _knn_imputation(self, data_array: xr.DataArray, k: int = 5) -> xr.DataArray:
        """K-Nearest Neighbors imputation for spatial data."""
        # Simplified KNN imputation - in production, use fancyimpute
        result = data_array.copy()
        
        if 'time' in data_array.dims:
            # Impute along time dimension
            for i in range(data_array.shape[0]):
                slice_2d = data_array[i].values
                if np.isnan(slice_2d).any():
                    # Simple mean imputation for now
                    mean_val = np.nanmean(slice_2d)
                    slice_2d[np.isnan(slice_2d)] = mean_val
                    result[i].values = slice_2d
        
        return result
    
    def detect_outliers(
        self,
        dataset: xr.Dataset,
        method: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect outliers in dataset.
        
        Args:
            dataset: Input dataset
            method: Detection method ('iqr', 'zscore', 'modified_zscore', 'isolation_forest')
            threshold: Outlier threshold
            
        Returns:
            Dictionary with outlier information
        """
        method = method or self.config.get('outlier_method', 'iqr')
        threshold = threshold or self.config.get('outlier_threshold', 3.0)
        
        outlier_info = {
            'method': method,
            'threshold': threshold,
            'variables': {},
            'total_outliers': 0
        }
        
        for var_name in dataset.data_vars:
            var = dataset[var_name]
            data = var.values.flatten()
            data_clean = data[~np.isnan(data)]
            
            if len(data_clean) == 0:
                continue
            
            outliers_mask = None
            
            if method == 'iqr':
                q1 = np.percentile(data_clean, 25)
                q3 = np.percentile(data_clean, 75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                outliers_mask = (data_clean < lower_bound) | (data_clean > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data_clean))
                outliers_mask = z_scores > threshold
            
            elif method == 'modified_zscore':
                median = np.median(data_clean)
                mad = np.median(np.abs(data_clean - median))
                modified_z_scores = 0.6745 * (data_clean - median) / (mad + 1e-10)
                outliers_mask = np.abs(modified_z_scores) > threshold
            
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            if outliers_mask is not None:
                outlier_count = outliers_mask.sum()
                outlier_percentage = (outlier_count / len(data_clean)) * 100
                
                outlier_info['variables'][var_name] = {
                    'outlier_count': int(outlier_count),
                    'outlier_percentage': float(outlier_percentage),
                    'lower_bound': float(np.percentile(data_clean, 1)) if method == 'iqr' else None,
                    'upper_bound': float(np.percentile(data_clean, 99)) if method == 'iqr' else None
                }
                
                outlier_info['total_outliers'] += outlier_count
                
                if outlier_percentage > 5:
                    logger.warning(
                        f"High outlier percentage in {var_name}: {outlier_percentage:.1f}%"
                    )
        
        logger.info(f"Outlier detection completed: {outlier_info['total_outliers']} total outliers")
        return outlier_info
    
    def remove_outliers(
        self,
        dataset: xr.Dataset,
        method: Optional[str] = None,
        **kwargs
    ) -> xr.Dataset:
        """
        Remove or handle outliers.
        
        Args:
            dataset: Input dataset
            method: Handling method ('clip', 'remove', 'winsorize', 'impute')
            
        Returns:
            Dataset with outliers handled
        """
        method = method or self.config.get('outlier_handling', 'clip')
        processed = dataset.copy()
        
        for var_name in processed.data_vars:
            var = processed[var_name]
            data = var.values
            
            if method == 'clip':
                # Clip to percentile bounds
                data_clean = data[~np.isnan(data)]
                if len(data_clean) > 0:
                    lower_bound = np.percentile(data_clean, 1)
                    upper_bound = np.percentile(data_clean, 99)
                    
                    # Clip outliers
                    clipped = np.clip(data, lower_bound, upper_bound)
                    processed[var_name].values = clipped
                    
                    logger.debug(f"Clipped outliers in {var_name} to [{lower_bound:.4f}, {upper_bound:.4f}]")
            
            elif method == 'winsorize':
                # Winsorize data (replace outliers with percentiles)
                data_clean = data[~np.isnan(data)]
                if len(data_clean) > 0:
                    lower_bound = np.percentile(data_clean, 5)
                    upper_bound = np.percentile(data_clean, 95)
                    
                    # Replace outliers
                    winsorized = data.copy()
                    winsorized[winsorized < lower_bound] = lower_bound
                    winsorized[winsorized > upper_bound] = upper_bound
                    processed[var_name].values = winsorized
                    
                    logger.debug(f"Winsorized {var_name} to 5th-95th percentiles")
            
            elif method == 'impute':
                # Replace outliers with median
                data_clean = data[~np.isnan(data)]
                if len(data_clean) > 0:
                    median_val = np.median(data_clean)
                    q1 = np.percentile(data_clean, 25)
                    q3 = np.percentile(data_clean, 75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                    
                    # Create mask and replace
                    outlier_mask = (data < lower_bound) | (data > upper_bound)
                    imputed = data.copy()
                    imputed[outlier_mask] = median_val
                    processed[var_name].values = imputed
                    
                    logger.debug(f"Imputed outliers in {var_name} with median")
        
        logger.info(f"Outlier handling completed using {method} method")
        return processed
    
    def scale_data(
        self,
        dataset: xr.Dataset,
        method: Optional[str] = None,
        feature_range: Tuple[float, float] = (0, 1)
    ) -> xr.Dataset:
        """
        Scale dataset features.
        
        Args:
            dataset: Input dataset
            method: Scaling method ('standard', 'minmax', 'robust', 'log', 'quantile', 'power')
            feature_range: Range for min-max scaling
            
        Returns:
            Scaled dataset
        """
        method = method or self.config.get('scaling_method', 'standard')
        processed = dataset.copy()
        
        self.scalers = {}
        
        for var_name in processed.data_vars:
            var = processed[var_name]
            data = var.values
            mask = ~np.isnan(data)
            
            if not np.any(mask):
                continue
            
            data_valid = data[mask]
            
            try:
                if method == 'standard':
                    # Standard scaling (zero mean, unit variance)
                    mean_val = np.mean(data_valid)
                    std_val = np.std(data_valid)
                    
                    scaled = data.copy()
                    scaled[mask] = (data_valid - mean_val) / (std_val + 1e-10)
                    
                    self.scalers[var_name] = {
                        'method': 'standard',
                        'mean': float(mean_val),
                        'std': float(std_val)
                    }
                
                elif method == 'minmax':
                    # Min-max scaling
                    min_val = np.min(data_valid)
                    max_val = np.max(data_valid)
                    range_val = max_val - min_val
                    
                    scaled = data.copy()
                    if range_val > 0:
                        scaled[mask] = (data_valid - min_val) / range_val
                        # Scale to specified range
                        scaled[mask] = scaled[mask] * (feature_range[1] - feature_range[0]) + feature_range[0]
                    
                    self.scalers[var_name] = {
                        'method': 'minmax',
                        'min': float(min_val),
                        'max': float(max_val),
                        'feature_range': feature_range
                    }
                
                elif method == 'robust':
                    # Robust scaling (using median and IQR)
                    median_val = np.median(data_valid)
                    q1 = np.percentile(data_valid, 25)
                    q3 = np.percentile(data_valid, 75)
                    iqr = q3 - q1
                    
                    scaled = data.copy()
                    scaled[mask] = (data_valid - median_val) / (iqr + 1e-10)
                    
                    self.scalers[var_name] = {
                        'method': 'robust',
                        'median': float(median_val),
                        'q1': float(q1),
                        'q3': float(q3)
                    }
                
                elif method == 'log':
                    # Log scaling (for skewed distributions)
                    min_val = np.min(data_valid)
                    if min_val <= 0:
                        # Add offset to make all values positive
                        offset = -min_val + 1
                        data_valid = data_valid + offset
                    
                    scaled = data.copy()
                    scaled[mask] = np.log1p(data_valid)
                    
                    self.scalers[var_name] = {
                        'method': 'log',
                        'offset': float(offset) if min_val <= 0 else 0
                    }
                
                elif method == 'power':
                    # Power transformation (Yeo-Johnson like)
                    scaled = data.copy()
                    # Simple square root for positive values
                    if np.min(data_valid) >= 0:
                        scaled[mask] = np.sqrt(data_valid)
                    else:
                        scaled[mask] = np.sign(data_valid) * np.sqrt(np.abs(data_valid))
                    
                    self.scalers[var_name] = {'method': 'power'}
                
                else:
                    raise ValueError(f"Unknown scaling method: {method}")
                
                processed[var_name].values = scaled
                logger.debug(f"Scaled {var_name} using {method} method")
            
            except Exception as e:
                logger.warning(f"Could not scale {var_name}: {e}")
                continue
        
        logger.info(f"Data scaling completed using {method} method")
        return processed
    
    def create_derived_features(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Create derived features for marine productivity.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with derived features
        """
        processed = dataset.copy()
        
        # 1. Euphotic depth (Z_eu = ln(1000) / Kd ≈ 4.605 / Kd490)
        if 'kd490' in processed:
            processed['euphotic_depth'] = 4.605 / processed['kd490']
            processed['euphotic_depth'].attrs.update({
                'long_name': 'Euphotic depth',
                'units': 'm',
                'description': 'Depth of 1% light penetration',
                'formula': 'Z_eu = ln(1000) / Kd490'
            })
            logger.debug("Created euphotic_depth feature")
        
        # 2. Chlorophyll-specific absorption
        if 'chl' in processed and 'kd490' in processed:
            processed['chl_specific_absorption'] = processed['kd490'] / (processed['chl'] + 0.01)
            processed['chl_specific_absorption'].attrs.update({
                'long_name': 'Chlorophyll-specific absorption',
                'units': 'm^-1/(mg/m^3)',
                'description': 'Light attenuation per unit chlorophyll'
            })
            logger.debug("Created chl_specific_absorption feature")
        
        # 3. Water clarity index
        if 'zsd' in processed:
            processed['clarity_index'] = 1 / (processed['zsd'] + 0.1)
            processed['clarity_index'].attrs.update({
                'long_name': 'Water clarity index',
                'units': 'm^-1',
                'description': 'Inverse of Secchi disk depth'
            })
            logger.debug("Created clarity_index feature")
        
        # 4. Current energy
        if 'current_speed' in processed:
            processed['current_energy'] = 0.5 * processed['current_speed'] ** 2
            processed['current_energy'].attrs.update({
                'long_name': 'Current kinetic energy',
                'units': 'm^2/s^2',
                'description': 'Kinetic energy of water movement'
            })
            logger.debug("Created current_energy feature")
        
        # 5. Mixed layer index (simplified)
        if 'sst' in processed and 'sss' in processed:
            # Temperature-salinity gradient
            if 'time' in processed.sst.dims and processed.sst.shape[0] > 1:
                sst_gradient = processed.sst.diff(dim='time')
                sss_gradient = processed.sss.diff(dim='time')
                
                processed['ts_gradient_magnitude'] = np.sqrt(sst_gradient**2 + sss_gradient**2)
                processed['ts_gradient_magnitude'].attrs.update({
                    'long_name': 'T-S gradient magnitude',
                    'units': 'combined units',
                    'description': 'Magnitude of temperature-salinity gradient over time'
                })
                logger.debug("Created ts_gradient_magnitude feature")
        
        # 6. Spatial gradients
        if all(dim in processed.dims for dim in ['lat', 'lon']):
            for var in ['chl', 'sst', 'kd490']:
                if var in processed:
                    try:
                        # Latitude gradient
                        lat_grad = processed[var].differentiate(coord='lat')
                        processed[f'{var}_lat_gradient'] = lat_grad
                        processed[f'{var}_lat_gradient'].attrs.update({
                            'long_name': f'{var} latitude gradient',
                            'units': f'{processed[var].attrs.get("units", "")}/degree',
                            'description': f'North-south gradient of {var}'
                        })
                        
                        # Longitude gradient
                        lon_grad = processed[var].differentiate(coord='lon')
                        processed[f'{var}_lon_gradient'] = lon_grad
                        processed[f'{var}_lon_gradient'].attrs.update({
                            'long_name': f'{var} longitude gradient',
                            'units': f'{processed[var].attrs.get("units", "")}/degree',
                            'description': f'East-west gradient of {var}'
                        })
                        
                        logger.debug(f"Created spatial gradients for {var}")
                    
                    except Exception as e:
                        logger.warning(f"Could not create spatial gradients for {var}: {e}")
        
        # 7. Temporal features
        if 'time' in processed.dims:
            processed['day_of_year'] = processed['time'].dt.dayofyear
            processed['day_of_year'].attrs.update({
                'long_name': 'Day of year',
                'units': 'day',
                'description': 'Day number within the year'
            })
            
            processed['month'] = processed['time'].dt.month
            processed['month'].attrs.update({
                'long_name': 'Month',
                'units': 'month',
                'description': 'Month of the year (1-12)'
            })
            
            # Seasonal features
            processed['season_sin'] = np.sin(2 * np.pi * processed['day_of_year'] / 365.25)
            processed['season_cos'] = np.cos(2 * np.pi * processed['day_of_year'] / 365.25)
            
            processed['season_sin'].attrs.update({
                'long_name': 'Seasonal sine component',
                'units': 'dimensionless',
                'description': 'Sine of annual cycle'
            })
            
            processed['season_cos'].attrs.update({
                'long_name': 'Seasonal cosine component',
                'units': 'dimensionless',
                'description': 'Cosine of annual cycle'
            })
            
            logger.debug("Created temporal features")
        
        # 8. Interaction terms
        if 'chl' in processed and 'euphotic_depth' in processed:
            processed['chl_per_euphotic_depth'] = processed['chl'] / (processed['euphotic_depth'] + 0.1)
            processed['chl_per_euphotic_depth'].attrs.update({
                'long_name': 'Chlorophyll per euphotic depth',
                'units': 'mg/m^3/m',
                'description': 'Chlorophyll concentration normalized by euphotic depth'
            })
            logger.debug("Created chl_per_euphotic_depth feature")
        
        logger.info(f"Created {len(processed.data_vars) - len(dataset.data_vars)} derived features")
        return processed
    
    def apply_temporal_filters(
        self,
        dataset: xr.Dataset,
        window_size: Optional[int] = None,
        filter_type: Optional[str] = None
    ) -> xr.Dataset:
        """
        Apply temporal filters to remove noise.
        
        Args:
            dataset: Input dataset
            window_size: Filter window size
            filter_type: Filter type ('savgol', 'moving_avg', 'exponential', 'median')
            
        Returns:
            Filtered dataset
        """
        window_size = window_size or self.config.get('temporal_window', 7)
        filter_type = filter_type or self.config.get('temporal_filter', 'savgol')
        
        processed = dataset.copy()
        
        for var_name in processed.data_vars:
            var = processed[var_name]
            
            if 'time' not in var.dims or len(var.time) < window_size:
                continue
            
            try:
                if filter_type == 'savgol':
                    # Savitzky-Golay filter for smoothing
                    filtered = xr.apply_ufunc(
                        savgol_filter,
                        var,
                        input_core_dims=[['time']],
                        output_core_dims=[['time']],
                        kwargs={
                            'window_length': min(window_size, len(var.time)),
                            'polyorder': 2,
                            'mode': 'interp'
                        },
                        dask='parallelized',
                        output_dtypes=[var.dtype]
                    )
                
                elif filter_type == 'moving_avg':
                    # Moving average
                    filtered = var.rolling(time=window_size, center=True).mean()
                
                elif filter_type == 'exponential':
                    # Exponential moving average
                    filtered = var.ewm(span=window_size, adjust=True).mean()
                
                elif filter_type == 'median':
                    # Median filter
                    filtered = xr.apply_ufunc(
                        medfilt,
                        var,
                        input_core_dims=[['time']],
                        output_core_dims=[['time']],
                        kwargs={'kernel_size': window_size},
                        dask='parallelized',
                        output_dtypes=[var.dtype]
                    )
                
                else:
                    raise ValueError(f"Unknown filter type: {filter_type}")
                
                # Only apply filter if it worked
                if not filtered.isnull().all():
                    processed[var_name] = filtered
                    logger.debug(f"Applied {filter_type} filter to {var_name}")
            
            except Exception as e:
                logger.warning(f"Could not apply temporal filter to {var_name}: {e}")
                continue
        
        logger.info(f"Applied {filter_type} temporal filter with window size {window_size}")
        return processed
    
    def prepare_features(
        self,
        dataset: xr.Dataset,
        target_var: str = 'pp',
        feature_vars: Optional[List[str]] = None,
        include_interactions: bool = True,
        include_polynomial: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix and target vector for machine learning.
        
        Args:
            dataset: Input dataset
            target_var: Target variable name
            feature_vars: List of feature variables to use
            include_interactions: Whether to include interaction terms
            include_polynomial: Whether to include polynomial features
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature names
        """
        if feature_vars is None:
            # Default marine productivity features
            feature_vars = [
                'chl', 'kd490', 'zsd', 'euphotic_depth',
                'sst', 'sss', 'current_speed',
                'day_of_year', 'month', 'season_sin', 'season_cos'
            ]
        
        # Filter available features
        available_features = [var for var in feature_vars if var in dataset.data_vars]
        
        if target_var not in dataset.data_vars:
            raise ValueError(f"Target variable '{target_var}' not found in dataset")
        
        # Collect feature data
        X_arrays = []
        feature_names = []
        
        for var_name in available_features:
            data = dataset[var_name].values.flatten()
            X_arrays.append(data.reshape(-1, 1))
            feature_names.append(var_name)
        
        # Create interaction terms
        if include_interactions:
            important_interactions = [
                ('chl', 'kd490'),      # Chlorophyll-light interaction
                ('chl', 'euphotic_depth'),  # Chlorophyll-depth interaction
                ('sst', 'chl'),        # Temperature-chlorophyll interaction
                ('current_speed', 'chl')  # Current-chlorophyll interaction
            ]
            
            for var1, var2 in important_interactions:
                if var1 in available_features and var2 in available_features:
                    idx1 = available_features.index(var1)
                    idx2 = available_features.index(var2)
                    
                    interaction = X_arrays[idx1].flatten() * X_arrays[idx2].flatten()
                    X_arrays.append(interaction.reshape(-1, 1))
                    feature_names.append(f"{var1}_{var2}_interaction")
        
        # Create polynomial features
        if include_polynomial:
            for var_name in ['chl', 'kd490', 'sst']:
                if var_name in available_features:
                    idx = available_features.index(var_name)
                    
                    # Quadratic term
                    quadratic = X_arrays[idx].flatten() ** 2
                    X_arrays.append(quadratic.reshape(-1, 1))
                    feature_names.append(f"{var_name}_squared")
                    
                    # Cubic term
                    cubic = X_arrays[idx].flatten() ** 3
                    X_arrays.append(cubic.reshape(-1, 1))
                    feature_names.append(f"{var_name}_cubic")
        
        # Combine all features
        X = np.hstack(X_arrays) if X_arrays else np.array([])
        
        # Target vector
        y = dataset[target_var].values.flatten()
        
        # Remove rows with NaN in either X or y
        valid_mask = ~np.isnan(y)
        if X.size > 0:
            valid_mask = valid_mask & ~np.any(np.isnan(X), axis=1)
        
        X_clean = X[valid_mask] if X.size > 0 else np.array([])
        y_clean = y[valid_mask]
        
        # Store feature names
        self.feature_names = feature_names
        
        logger.info(f"Prepared features: {X_clean.shape[0]} samples, {len(feature_names)} features")
        logger.info(f"Features: {feature_names}")
        
        return X_clean, y_clean, feature_names
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing operations."""
        summary = {
            'config': self.config,
            'scalers_applied': len(self.scalers),
            'scaler_methods': {var: info.get('method', 'unknown') 
                             for var, info in self.scalers.items()},
            'feature_count': len(self.feature_names),
            'features_created': self.feature_names
        }
        
        return summary
    
    def save_preprocessing_info(self, output_path: Union[str, Path]):
        """Save preprocessing information to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        
        info = {
            'preprocessing_summary': self.get_preprocessing_summary(),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        
        logger.info(f"Preprocessing info saved to: {output_path}")


# Utility function for quick preprocessing
def preprocess_marine_data(
    dataset: xr.Dataset,
    config: Optional[Dict] = None,
    target_var: str = 'pp'
) -> Tuple[xr.Dataset, Dict[str, Any]]:
    """
    Utility function for quick preprocessing.
    
    Args:
        dataset: Input dataset
        config: Preprocessing configuration
        target_var: Target variable name
        
    Returns:
        processed_dataset: Preprocessed dataset
        preprocessing_info: Summary of preprocessing
    """
    preprocessor = DataPreprocessor(config)
    processed = preprocessor.process(dataset, target_var)
    info = preprocessor.get_preprocessing_summary()
    
    return processed, info


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data for testing
    time = pd.date_range('2020-01-01', periods=100, freq='D')
    lat = np.linspace(-90, 90, 50)
    lon = np.linspace(-180, 180, 100)
    
    ds = xr.Dataset(
        {
            'chl': xr.DataArray(
                np.random.randn(len(time), len(lat), len(lon)) * 0.5 + 1.0,
                dims=['time', 'lat', 'lon'],
                coords={'time': time, 'lat': lat, 'lon': lon}
            ),
            'kd490': xr.DataArray(
                np.random.randn(len(time), len(lat), len(lon)) * 0.2 + 0.5,
                dims=['time', 'lat', 'lon']
            ),
            'pp': xr.DataArray(
                np.random.randn(len(time), len(lat), len(lon)) * 100 + 500,
                dims=['time', 'lat', 'lon']
            )
        }
    )
    
    print("Testing DataPreprocessor...")
    preprocessor = DataPreprocessor()
    
    # Process data
    processed = preprocessor.process(ds, target_var='pp')
    
    # Prepare features
    X, y, features = preprocessor.prepare_features(processed, target_var='pp')
    
    print(f"Processed data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {features}")
    print(f"Preprocessing summary: {preprocessor.get_preprocessing_summary()}")
