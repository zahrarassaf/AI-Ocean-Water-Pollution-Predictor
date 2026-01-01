"""
Data loading utilities optimized for NetCDF marine data
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any
import logging
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
import zarr
import warnings

from ..config.constants import DATA_DIR, DataConfig
from ..utils.logger import setup_logger
from ..utils.parallel import ParallelProcessor

logger = setup_logger(__name__)

class MarineDataLoader:
    """Optimized data loader for marine NetCDF datasets"""
    
    def __init__(self, 
                 config: Optional[DataConfig] = None, 
                 use_dask: bool = True,
                 max_memory_gb: float = 4.0):
        """
        Initialize marine data loader
        
        Parameters
        ----------
        config : DataConfig, optional
            Configuration
        use_dask : bool
            Whether to use Dask for parallel processing
        max_memory_gb : float
            Maximum memory to use (in GB)
        """
        self.config = config or DataConfig()
        self.use_dask = use_dask
        self.max_memory_gb = max_memory_gb
        
        # Track loaded datasets
        self.datasets = {}
        self.variable_info = {}
        
        if use_dask:
            self._init_dask_cluster()
    
    def _init_dask_cluster(self):
        """Initialize Dask cluster optimized for NetCDF"""
        try:
            # Calculate optimal chunk size based on memory
            memory_per_worker = self.max_memory_gb * 1024**3  # Convert to bytes
            
            self.client = Client(
                n_workers=max(1, int(self.max_memory_gb / 2)),  # 2GB per worker
                threads_per_worker=1,
                memory_limit=f'{int(self.max_memory_gb)}GB',
                processes=True,
                silence_logs=logging.WARNING
            )
            
            logger.info(f"Dask cluster initialized: {self.client}")
            
        except Exception as e:
            logger.warning(f"Could not initialize Dask cluster: {e}")
            self.client = None
            self.use_dask = False
    
    def load_marine_dataset(self,
                           filepath: Union[str, Path],
                           variables: Optional[List[str]] = None,
                           time_range: Optional[Tuple[str, str]] = None,
                           spatial_range: Optional[Dict[str, Tuple[float, float]]] = None,
                           preprocess: bool = True) -> xr.Dataset:
        """
        Load marine NetCDF dataset with optimization
        
        Parameters
        ----------
        filepath : str or Path
            Path to NetCDF file
        variables : list, optional
            Specific variables to load
        time_range : tuple, optional
            (start_time, end_time) as strings
        spatial_range : dict, optional
            {'lat': (min, max), 'lon': (min, max)}
        preprocess : bool
            Whether to apply basic preprocessing
            
        Returns
        -------
        xr.Dataset
            Loaded dataset
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading marine dataset: {filepath.name}")
        
        try:
            # Determine optimal chunks based on file size
            file_size_gb = filepath.stat().st_size / (1024**3)
            
            if file_size_gb > 1.0:  # Large file
                # Use chunking for large files
                chunks = self._optimize_chunks(filepath, file_size_gb)
                logger.info(f"Using chunks: {chunks}")
                
                ds = xr.open_dataset(
                    filepath,
                    chunks=chunks,
                    engine='netcdf4',
                    decode_cf=True,
                    decode_times=True,
                    cache=True
                )
            else:
                # Small file, load directly
                ds = xr.open_dataset(
                    filepath,
                    engine='netcdf4',
                    decode_cf=True,
                    decode_times=True
                )
            
            # Select specific variables
            if variables:
                available_vars = [v for v in variables if v in ds]
                missing_vars = set(variables) - set(available_vars)
                
                if missing_vars:
                    logger.warning(f"Missing variables: {missing_vars}")
                
                ds = ds[available_vars]
                logger.info(f"Selected variables: {available_vars}")
            
            # Apply time range filter
            if time_range and 'time' in ds.dims:
                start_time, end_time = time_range
                ds = ds.sel(time=slice(start_time, end_time))
                logger.info(f"Filtered time: {start_time} to {end_time}")
            
            # Apply spatial range filter
            if spatial_range:
                for dim, (min_val, max_val) in spatial_range.items():
                    if dim in ds.dims:
                        ds = ds.sel({dim: slice(min_val, max_val)})
                        logger.info(f"Filtered {dim}: {min_val} to {max_val}")
            
            # Basic preprocessing
            if preprocess:
                ds = self._preprocess_dataset(ds)
            
            # Store dataset
            dataset_id = filepath.stem
            self.datasets[dataset_id] = ds
            
            # Store variable information
            self._analyze_variables(ds, dataset_id)
            
            logger.info(f"✓ Dataset loaded: {dict(ds.dims)} dimensions, "
                       f"{len(ds.data_vars)} variables")
            
            return ds
            
        except Exception as e:
            logger.error(f"Error loading dataset {filepath}: {e}")
            raise
    
    def _optimize_chunks(self, filepath: Path, file_size_gb: float) -> Dict:
        """Optimize chunk sizes for NetCDF file"""
        # Try to get dimension sizes
        try:
            with xr.open_dataset(filepath, engine='netcdf4') as temp_ds:
                dims = temp_ds.dims
                
                # Default chunks
                chunks = {}
                
                # Optimize based on dimensions
                for dim_name, dim_size in dims.items():
                    if dim_name == 'time':
                        # Time: chunk by months or reasonable size
                        chunks[dim_name] = min(365, dim_size)  # 1 year of daily data
                    elif dim_name in ['lat', 'latitude']:
                        chunks[dim_name] = min(100, dim_size)
                    elif dim_name in ['lon', 'longitude']:
                        chunks[dim_name] = min(100, dim_size)
                    else:
                        chunks[dim_name] = dim_size  # No chunking for small dimensions
                
                # Adjust based on file size
                if file_size_gb > 10:
                    # Very large file, use smaller chunks
                    for dim in chunks:
                        if isinstance(chunks[dim], int) and chunks[dim] > 50:
                            chunks[dim] = max(10, chunks[dim] // 4)
                
                return chunks
                
        except:
            # Fallback to auto-chunking
            return 'auto'
    
    def _preprocess_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply basic preprocessing to dataset"""
        processed = ds.copy()
        
        # Convert time to datetime if needed
        if 'time' in processed.coords:
            try:
                if not np.issubdtype(processed.time.dtype, np.datetime64):
                    processed['time'] = xr.decode_cf(processed).time
            except:
                pass
        
        # Ensure consistent coordinate names
        coord_mapping = {
            'latitude': 'lat',
            'longitude': 'lon',
            'Latitude': 'lat',
            'Longitude': 'lon'
        }
        
        for old_name, new_name in coord_mapping.items():
            if old_name in processed.dims:
                processed = processed.rename({old_name: new_name})
        
        # Add CF-compliant attributes if missing
        if 'lat' in processed.coords and 'long_name' not in processed.lat.attrs:
            processed.lat.attrs['long_name'] = 'latitude'
            processed.lat.attrs['units'] = 'degrees_north'
            processed.lat.attrs['standard_name'] = 'latitude'
        
        if 'lon' in processed.coords and 'long_name' not in processed.lon.attrs:
            processed.lon.attrs['long_name'] = 'longitude'
            processed.lon.attrs['units'] = 'degrees_east'
            processed.lon.attrs['standard_name'] = 'longitude'
        
        return processed
    
    def _analyze_variables(self, ds: xr.Dataset, dataset_id: str):
        """Analyze variables in dataset"""
        info = {}
        
        for var_name in ds.data_vars:
            var = ds[var_name]
            
            info[var_name] = {
                'dtype': str(var.dtype),
                'dimensions': list(var.dims),
                'shape': var.shape,
                'units': var.attrs.get('units', 'unknown'),
                'long_name': var.attrs.get('long_name', var_name),
                'has_nan': bool(np.isnan(var.values).any()),
                'valid_range': self._get_valid_range(var)
            }
            
            # Calculate statistics if not too large
            if var.values.size < 1e6:  # 1 million elements
                flat_data = var.values.flatten()
                flat_data = flat_data[~np.isnan(flat_data)]
                
                if len(flat_data) > 0:
                    info[var_name].update({
                        'mean': float(np.mean(flat_data)),
                        'std': float(np.std(flat_data)),
                        'min': float(np.min(flat_data)),
                        'max': float(np.max(flat_data)),
                        'median': float(np.median(flat_data))
                    })
        
        self.variable_info[dataset_id] = info
    
    def _get_valid_range(self, var: xr.DataArray) -> Optional[Tuple[float, float]]:
        """Get valid range from variable attributes"""
        valid_range = var.attrs.get('valid_range')
        if valid_range is not None:
            if isinstance(valid_range, (list, tuple)) and len(valid_range) == 2:
                return (float(valid_range[0]), float(valid_range[1]))
        
        # Check for actual_range
        actual_range = var.attrs.get('actual_range')
        if actual_range is not None:
            if isinstance(actual_range, (list, tuple)) and len(actual_range) == 2:
                return (float(actual_range[0]), float(actual_range[1]))
        
        return None
    
    def load_multiple_datasets(self,
                              filepaths: List[Union[str, Path]],
                              merge_dim: str = 'time',
                              preprocess: bool = True,
                              **kwargs) -> xr.Dataset:
        """
        Load and merge multiple NetCDF files
        
        Parameters
        ----------
        filepaths : list
            List of file paths
        merge_dim : str
            Dimension to merge along
        preprocess : bool
            Whether to preprocess each dataset
        **kwargs
            Additional arguments for load_marine_dataset
            
        Returns
        -------
        xr.Dataset
            Merged dataset
        """
        datasets = []
        
        for i, filepath in enumerate(filepaths):
            logger.info(f"Loading file {i+1}/{len(filepaths)}: {Path(filepath).name}")
            
            ds = self.load_marine_dataset(
                filepath,
                preprocess=preprocess,
                **kwargs
            )
            
            datasets.append(ds)
        
        # Check if datasets can be merged
        if len(datasets) > 1:
            logger.info(f"Merging {len(datasets)} datasets along '{merge_dim}' dimension...")
            
            # Check consistency
            self._check_merge_compatibility(datasets, merge_dim)
            
            # Merge datasets
            merged = xr.concat(datasets, dim=merge_dim)
            
            # Sort by time if time dimension exists
            if 'time' in merged.dims:
                merged = merged.sortby('time')
            
            logger.info(f"✓ Merged dataset: {dict(merged.dims)} dimensions, "
                       f"{len(merged.data_vars)} variables")
            
            return merged
        
        elif len(datasets) == 1:
            return datasets[0]
        else:
            raise ValueError("No datasets loaded")
    
    def _check_merge_compatibility(self, datasets: List[xr.Dataset], merge_dim: str):
        """Check if datasets can be merged"""
        if not datasets:
            return
        
        # Get reference dataset
        ref_ds = datasets[0]
        
        # Check dimensions (excluding merge dimension)
        ref_dims = set(ref_ds.dims) - {merge_dim}
        
        for i, ds in enumerate(datasets[1:], 1):
            ds_dims = set(ds.dims) - {merge_dim}
            
            if ref_dims != ds_dims:
                logger.warning(f"Dataset {i} has different dimensions: "
                             f"{ds_dims} vs {ref_dims}")
            
            # Check coordinate values for non-merge dimensions
            for dim in ref_dims:
                if dim in ds.dims:
                    try:
                        # Check if coordinates match (within tolerance)
                        if not np.allclose(ref_ds[dim].values, ds[dim].values):
                            logger.warning(f"Dataset {i} has different {dim} coordinates")
                    except:
                        logger.warning(f"Cannot compare {dim} coordinates")
    
    def extract_time_series(self,
                           ds: xr.Dataset,
                           variable: str,
                           location: Optional[Dict[str, float]] = None,
                           aggregation: str = 'mean') -> pd.DataFrame:
        """
        Extract time series for a variable
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset
        variable : str
            Variable name
        location : dict, optional
            {'lat': value, 'lon': value} for specific location
        aggregation : str
            Aggregation method: 'mean', 'median', 'sum', 'min', 'max'
            
        Returns
        -------
        pd.DataFrame
            Time series DataFrame
        """
        if variable not in ds:
            raise ValueError(f"Variable {variable} not in dataset")
        
        var_data = ds[variable]
        
        if location and 'lat' in var_data.dims and 'lon' in var_data.dims:
            # Extract at specific location
            lat = location.get('lat')
            lon = location.get('lon')
            
            # Find nearest grid point
            lat_idx = np.abs(var_data.lat.values - lat).argmin()
            lon_idx = np.abs(var_data.lon.values - lon).argmin()
            
            ts = var_data.isel(lat=lat_idx, lon=lon_idx)
            
        elif 'time' in var_data.dims:
            # Aggregate over spatial dimensions
            spatial_dims = [d for d in var_data.dims if d != 'time']
            
            if spatial_dims:
                if aggregation == 'mean':
                    ts = var_data.mean(dim=spatial_dims)
                elif aggregation == 'median':
                    ts = var_data.median(dim=spatial_dims)
                elif aggregation == 'sum':
                    ts = var_data.sum(dim=spatial_dims)
                elif aggregation == 'min':
                    ts = var_data.min(dim=spatial_dims)
                elif aggregation == 'max':
                    ts = var_data.max(dim=spatial_dims)
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
            else:
                ts = var_data
        else:
            raise ValueError("No time dimension found")
        
        # Convert to DataFrame
        df = ts.to_dataframe(name=variable)
        
        # Reset index if needed
        if 'time' in df.index.names and len(df.index.names) > 1:
            df = df.reset_index('time')
        
        logger.info(f"Extracted time series for {variable}: {len(df)} points")
        
        return df
    
    def close(self):
        """Close all datasets and clean up"""
        # Close datasets
        for ds in self.datasets.values():
            if hasattr(ds, 'close'):
                ds.close()
        
        self.datasets.clear()
        
        # Close Dask client
        if hasattr(self, 'client') and self.client:
            self.client.close()
            logger.info("Dask cluster closed")
