"""
Data loading utilities with memory optimization
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import logging
import dask
from dask.distributed import Client
import zarr

from ..config.constants import DATA_DIR, DataConfig
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DataLoader:
    """Optimized data loader for large oceanographic datasets"""
    
    def __init__(self, config: Optional[DataConfig] = None, use_dask: bool = True):
        self.config = config or DataConfig()
        self.use_dask = use_dask
        
        if use_dask:
            self._init_dask_cluster()
    
    def _init_dask_cluster(self):
        """Initialize Dask cluster for parallel processing"""
        try:
            self.client = Client(
                n_workers=self.config.n_workers,
                threads_per_worker=2,
                memory_limit='4GB'
            )
            logger.info(f"Dask cluster initialized: {self.client}")
        except Exception as e:
            logger.warning(f"Could not initialize Dask cluster: {e}")
            self.client = None
    
    def load_netcdf(self, 
                    filepath: Union[str, Path],
                    variables: Optional[List[str]] = None,
                    time_slice: Optional[slice] = None,
                    spatial_slice: Optional[Dict] = None) -> xr.Dataset:
        """
        Load NetCDF file with lazy loading
        
        Parameters
        ----------
        filepath : str or Path
            Path to NetCDF file
        variables : list, optional
            Specific variables to load
        time_slice : slice, optional
            Time dimension slice
        spatial_slice : dict, optional
            Spatial dimension slices (lat, lon)
            
        Returns
        -------
        xr.Dataset
            Loaded dataset
        """
        try:
            logger.info(f"Loading NetCDF: {filepath}")
            
            # Open dataset with lazy loading
            ds = xr.open_dataset(
                filepath,
                chunks=self.config.chunk_size,
                engine='netcdf4'
            )
            
            # Select variables
            if variables:
                available_vars = [v for v in variables if v in ds]
                missing_vars = set(variables) - set(available_vars)
                if missing_vars:
                    logger.warning(f"Missing variables: {missing_vars}")
                ds = ds[available_vars]
            
            # Apply slices
            if time_slice:
                ds = ds.sel(time=time_slice)
            
            if spatial_slice:
                for dim, slice_val in spatial_slice.items():
                    if dim in ds.dims:
                        ds = ds.sel({dim: slice_val})
            
            logger.info(f"Dataset loaded: {ds.dims}")
            return ds
            
        except Exception as e:
            logger.error(f"Error loading NetCDF: {e}")
            raise
    
    def load_multiple_files(self,
                           filepaths: List[Union[str, Path]],
                           merge_dim: str = 'time',
                           **kwargs) -> xr.Dataset:
        """
        Load and merge multiple NetCDF files
        
        Parameters
        ----------
        filepaths : list
            List of file paths
        merge_dim : str
            Dimension to merge along
            
        Returns
        -------
        xr.Dataset
            Merged dataset
        """
        datasets = []
        
        for filepath in filepaths:
            ds = self.load_netcdf(filepath, **kwargs)
            datasets.append(ds)
        
        # Merge datasets
        merged = xr.concat(datasets, dim=merge_dim)
        
        # Sort by time if time dimension exists
        if 'time' in merged.dims:
            merged = merged.sortby('time')
        
        logger.info(f"Merged {len(datasets)} datasets")
        return merged
    
    def to_zarr(self,
               dataset: xr.Dataset,
               output_path: Union[str, Path],
               compression: bool = True):
        """
        Save dataset to Zarr format for efficient storage
        
        Parameters
        ----------
        dataset : xr.Dataset
            Dataset to save
        output_path : str or Path
            Output path
        compression : bool
            Whether to use compression
        """
        try:
            encoding = {}
            if compression:
                for var in dataset.data_vars:
                    encoding[var] = {
                        'compressor': zarr.Blosc(cname='zstd', clevel=3),
                        'chunks': dataset[var].shape
                    }
            
            dataset.to_zarr(
                output_path,
                encoding=encoding,
                consolidated=True
            )
            logger.info(f"Dataset saved to Zarr: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving to Zarr: {e}")
            raise
    
    def validate_dataset(self, dataset: xr.Dataset) -> Dict:
        """
        Validate dataset structure and quality
        
        Parameters
        ----------
        dataset : xr.Dataset
            Dataset to validate
            
        Returns
        -------
        dict
            Validation results
        """
        validation = {
            'dimensions': {},
            'variables': {},
            'quality_metrics': {},
            'issues': []
        }
        
        # Check dimensions
        for dim in dataset.dims:
            validation['dimensions'][dim] = {
                'size': dataset.dims[dim],
                'has_nan': False
            }
        
        # Check variables
        for var in dataset.data_vars:
            data = dataset[var].values
            nan_count = np.isnan(data).sum()
            nan_percentage = (nan_count / data.size) * 100
            
            validation['variables'][var] = {
                'shape': dataset[var].shape,
                'dtype': str(dataset[var].dtype),
                'nan_count': int(nan_count),
                'nan_percentage': float(nan_percentage),
                'mean': float(np.nanmean(data)),
                'std': float(np.nanstd(data)),
                'min': float(np.nanmin(data)),
                'max': float(np.nanmax(data))
            }
            
            if nan_percentage > 50:
                validation['issues'].append(f"High NaN percentage in {var}: {nan_percentage:.2f}%")
        
        # Check required dimensions
        required_dims = ['time', 'lat', 'lon']
        for dim in required_dims:
            if dim not in dataset.dims:
                validation['issues'].append(f"Missing required dimension: {dim}")
        
        logger.info(f"Dataset validation complete. Issues: {len(validation['issues'])}")
        return validation
    
    def create_data_splits(self,
                          dataset: xr.Dataset,
                          split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                          time_dim: str = 'time') -> Dict[str, xr.Dataset]:
        """
        Split dataset into train/validation/test sets
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
        split_ratio : tuple
            Train/val/test ratios
        time_dim : str
            Time dimension name
            
        Returns
        -------
        dict
            Split datasets
        """
        train_ratio, val_ratio, test_ratio = split_ratio
        
        # Sort by time
        if time_dim in dataset.dims:
            dataset = dataset.sortby(time_dim)
        
        # Calculate split indices
        total_len = len(dataset[time_dim])
        train_end = int(total_len * train_ratio)
        val_end = train_end + int(total_len * val_ratio)
        
        # Split dataset
        train_ds = dataset.isel({time_dim: slice(0, train_end)})
        val_ds = dataset.isel({time_dim: slice(train_end, val_end)})
        test_ds = dataset.isel({time_dim: slice(val_end, None)})
        
        splits = {
            'train': train_ds,
            'validation': val_ds,
            'test': test_ds
        }
        
        logger.info(f"Data splits created - Train: {len(train_ds[time_dim])}, "
                   f"Val: {len(val_ds[time_dim])}, Test: {len(test_ds[time_dim])}")
        
        return splits
    
    def close(self):
        """Clean up resources"""
        if self.client:
            self.client.close()
            logger.info("Dask cluster closed")
