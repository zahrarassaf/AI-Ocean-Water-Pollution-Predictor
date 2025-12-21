"""
CMEMS data processor for OceanPredict framework
Advanced processing pipeline for oceanographic data
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import warnings
from scipy import interpolate, stats
import dask.array as da
from dask.diagnostics import ProgressBar

from ..utils.config_loader import ConfigLoader
from ..utils.logger import logger

warnings.filterwarnings('ignore')

class CMEMSProcessor:
    """
    Advanced CMEMS data processor with quality control,
    gap filling, and derived variable calculation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize CMEMS processor
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration file
        """
        self.config = ConfigLoader()
        self.logger = logger
        
        # Setup directories
        self.setup_directories()
        
        # Variable mappings
        self.variable_mappings = self._create_variable_mappings()
        
        # Quality control parameters
        self.qc_params = self._get_qc_parameters()
        
        self.logger.info("CMEMSProcessor initialized successfully")
    
    def setup_directories(self) -> None:
        """Create necessary directory structure"""
        directories = [
            Path("data/raw/bgc_optics"),
            Path("data/raw/bgc_plankton"),
            Path("data/raw/bgc_pp"),
            Path("data/raw/physics_currents"),
            Path("data/processed"),
            Path("data/interim"),
            Path("results/figures"),
            Path("results/models"),
            Path("results/reports")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Directory structure created")
    
    def _create_variable_mappings(self) -> Dict[str, List[str]]:
        """Create variable name mappings for different CMEMS products"""
        return {
            'primary_productivity': ['nppv', 'pp', 'primary_production'],
            'chlorophyll': ['chl', 'chlorophyll_a', 'CHLA'],
            'diffuse_attenuation': ['kd490', 'Kd_490'],
            'secchi_depth': ['zsd', 'secchi_depth'],
            'backscattering': ['bbp', 'bbp443'],
            'cdom': ['cdm', 'adg443'],
            'eastward_current': ['uo', 'eastward_sea_water_velocity'],
            'northward_current': ['vo', 'northward_sea_water_velocity']
        }
    
    def _get_qc_parameters(self) -> Dict[str, Tuple[float, float]]:
        """Get quality control parameters for each variable"""
        return {
            'pp': (0, 1000),      # mg C m^-2 d^-1
            'chl': (0.01, 30),    # mg m^-3
            'kd490': (0.01, 5),   # m^-1
            'zsd': (0.1, 50),     # m
            'bbp': (0.0001, 0.1), # m^-1
            'cdm': (0.001, 1),    # m^-1
            'uo': (-5, 5),        # m/s
            'vo': (-5, 5)         # m/s
        }
    
    def process_single_file(self, file_path: Union[str, Path], 
                           data_type: str) -> Optional[xr.Dataset]:
        """
        Process a single CMEMS file with comprehensive quality control
        
        Parameters
        ----------
        file_path : str or Path
            Path to NetCDF file
        data_type : str
            Type of data (bgc_optics, physics_currents, etc.)
            
        Returns
        -------
        xr.Dataset or None
            Processed dataset
        """
        file_path = Path(file_path)
        
        try:
            self.logger.info(f"Processing file: {file_path.name}")
            
            # 1. Open dataset with Dask for large files
            ds = xr.open_dataset(file_path, chunks={'time': 10})
            
            # 2. Standardize coordinates and dimensions
            ds = self._standardize_dataset(ds)
            
            # 3. Extract relevant variables
            ds = self._extract_variables(ds, data_type)
            
            if ds is None or len(ds.data_vars) == 0:
                self.logger.warning(f"No relevant variables found in {file_path.name}")
                return None
            
            # 4. Subset to region of interest
            ds = self._subset_to_region(ds)
            
            # 5. Subset to time period
            ds = self._subset_to_time(ds)
            
            # 6. Apply comprehensive quality control
            ds = self._apply_quality_control(ds)
            
            # 7. Gap filling and interpolation
            ds = self._fill_gaps(ds)
            
            # 8. Add metadata
            ds = self._add_metadata(ds, file_path, data_type)
            
            self.logger.info(f"✓ Processed {file_path.name}: {list(ds.data_vars)}")
            
            return ds
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {e}")
            return None
    
    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize dataset coordinates and dimensions"""
        
        # Standard coordinate names
        rename_dict = {}
        
        # Latitude
        if 'latitude' in ds.dims:
            rename_dict['latitude'] = 'lat'
        elif 'nav_lat' in ds.dims:
            rename_dict['nav_lat'] = 'lat'
        elif 'LATITUDE' in ds.dims:
            rename_dict['LATITUDE'] = 'lat'
        
        # Longitude
        if 'longitude' in ds.dims:
            rename_dict['longitude'] = 'lon'
        elif 'nav_lon' in ds.dims:
            rename_dict['nav_lon'] = 'lon'
        elif 'LONGITUDE' in ds.dims:
            rename_dict['LONGITUDE'] = 'lon'
        
        # Time
        if 'TIME' in ds.dims:
            rename_dict['TIME'] = 'time'
        elif 't' in ds.dims:
            rename_dict['t'] = 'time'
        
        if rename_dict:
            ds = ds.rename(rename_dict)
        
        # Ensure coordinates are 1D
        for coord in ['lat', 'lon']:
            if coord in ds.dims and ds[coord].ndim > 1:
                # Convert 2D coordinate to 1D
                if coord == 'lat':
                    ds = ds.assign_coords(lat=ds.lat.mean(dim='lon'))
                elif coord == 'lon':
                    ds = ds.assign_coords(lon=ds.lon.mean(dim='lat'))
        
        return ds
    
    def _extract_variables(self, ds: xr.Dataset, data_type: str) -> Optional[xr.Dataset]:
        """Extract relevant variables based on data type"""
        
        # Map data type to target variables
        type_to_vars = {
            'bgc_optics': ['kd490', 'zsd', 'bbp', 'cdm'],
            'bgc_plankton': ['chl'],
            'bgc_pp': ['pp'],
            'physics_currents': ['uo', 'vo']
        }
        
        if data_type not in type_to_vars:
            self.logger.warning(f"Unknown data type: {data_type}")
            return None
        
        target_vars = type_to_vars[data_type]
        extracted_vars = {}
        
        for target_var in target_vars:
            # Find variable in dataset using mappings
            found_var = self._find_variable(ds, target_var)
            
            if found_var:
                extracted_vars[target_var] = ds[found_var]
                
                # Preserve attributes
                if hasattr(ds[found_var], 'attrs'):
                    extracted_vars[target_var].attrs = ds[found_var].attrs.copy()
        
        if not extracted_vars:
            return None
        
        # Create new dataset with extracted variables
        extracted_ds = xr.Dataset(extracted_vars)
        
        # Copy global attributes
        if hasattr(ds, 'attrs'):
            extracted_ds.attrs = ds.attrs.copy()
        
        return extracted_ds
    
    def _find_variable(self, ds: xr.Dataset, target_var: str) -> Optional[str]:
        """Find variable in dataset using name mappings"""
        if target_var in self.variable_mappings:
            possible_names = self.variable_mappings[target_var]
            
            # Direct match
            for name in possible_names:
                if name in ds.data_vars:
                    return name
            
            # Case-insensitive match
            ds_vars_lower = [v.lower() for v in ds.data_vars]
            for name in possible_names:
                if name.lower() in ds_vars_lower:
                    idx = ds_vars_lower.index(name.lower())
                    return list(ds.data_vars)[idx]
        
        return None
    
    def _subset_to_region(self, ds: xr.Dataset) -> xr.Dataset:
        """Subset dataset to Gulf of Mexico region"""
        if 'lat' not in ds.dims or 'lon' not in ds.dims:
            return ds
        
        region = self.config.get('region.bounds')
        
        if not region:
            self.logger.warning("Region configuration not found")
            return ds
        
        lat_min = region['latitude']['min']
        lat_max = region['latitude']['max']
        lon_min = region['longitude']['min']
        lon_max = region['longitude']['max']
        
        # Handle longitude conventions
        lon_data = ds.lon.values
        if lon_data.min() >= 0 and lon_min < 0:
            # Convert negative longitudes to 0-360 range
            lon_min = (lon_min + 360) % 360
            lon_max = (lon_max + 360) % 360
        
        try:
            # Subset to region
            ds_subset = ds.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max)
            )
            
            if len(ds_subset.lat) == 0 or len(ds_subset.lon) == 0:
                self.logger.warning(f"No data in region: lat={lat_min}:{lat_max}, lon={lon_min}:{lon_max}")
                return ds
            
            return ds_subset
            
        except Exception as e:
            self.logger.warning(f"Error subsetting to region: {e}")
            return ds
    
    def _subset_to_time(self, ds: xr.Dataset) -> xr.Dataset:
        """Subset dataset to study time period"""
        if 'time' not in ds.dims:
            return ds
        
        time_config = self.config.get('time_period')
        
        if not time_config:
            return ds
        
        start_date = pd.Timestamp(time_config['start'])
        end_date = pd.Timestamp(time_config['end'])
        
        try:
            ds_time = ds.sel(time=slice(start_date, end_date))
            
            if len(ds_time.time) == 0:
                self.logger.warning(f"No data in time range: {start_date} to {end_date}")
                return ds
            
            return ds_time
            
        except Exception as e:
            self.logger.warning(f"Error subsetting time: {e}")
            return ds
    
    def _apply_quality_control(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply comprehensive quality control"""
        ds_qc = ds.copy()
        
        for var_name in ds_qc.data_vars:
            var_base = var_name.split('_')[0]
            
            if var_base in self.qc_params:
                vmin, vmax = self.qc_params[var_base]
                
                # Apply valid range
                ds_qc[var_name] = ds_qc[var_name].where(
                    (ds_qc[var_name] >= vmin) & (ds_qc[var_name] <= vmax)
                )
                
                # Statistical outlier detection (3-sigma rule)
                if 'time' in ds_qc[var_name].dims:
                    mean_val = ds_qc[var_name].mean(dim=['lat', 'lon'], skipna=True)
                    std_val = ds_qc[var_name].std(dim=['lat', 'lon'], skipna=True)
                    
                    # Mask outliers
                    outlier_mask = (
                        (ds_qc[var_name] < mean_val - 3 * std_val) |
                        (ds_qc[var_name] > mean_val + 3 * std_val)
                    )
                    
                    ds_qc[var_name] = ds_qc[var_name].where(~outlier_mask)
        
        return ds_qc
    
    def _fill_gaps(self, ds: xr.Dataset) -> xr.Dataset:
        """Fill gaps using interpolation and reconstruction"""
        ds_filled = ds.copy()
        
        for var_name in ds_filled.data_vars:
            var_data = ds_filled[var_name]
            
            # Count missing values before filling
            missing_before = np.isnan(var_data.values).sum()
            
            if missing_before == 0:
                continue
            
            # 1. Temporal interpolation (for time series)
            if 'time' in var_data.dims:
                # Linear interpolation in time (max 3 time steps)
                ds_filled[var_name] = var_data.interpolate_na(
                    dim='time', method='linear', limit=3
                )
            
            # 2. Spatial interpolation
            if all(dim in var_data.dims for dim in ['lat', 'lon']):
                # Use scipy's griddata for spatial interpolation
                var_data = ds_filled[var_name]
                
                # Only interpolate if we have enough valid points
                valid_mask = ~np.isnan(var_data.values)
                valid_count = np.sum(valid_mask)
                
                if valid_count > 100:  # Minimum points for interpolation
                    # Create grid coordinates
                    lon_grid, lat_grid = np.meshgrid(var_data.lon, var_data.lat)
                    
                    # Get valid points
                    valid_points = np.column_stack([
                        lat_grid[valid_mask].flatten(),
                        lon_grid[valid_mask].flatten()
                    ])
                    valid_values = var_data.values[valid_mask].flatten()
                    
                    # All grid points
                    all_points = np.column_stack([
                        lat_grid.flatten(),
                        lon_grid.flatten()
                    ])
                    
                    # Interpolate using nearest neighbor (fast and robust)
                    from scipy.interpolate import NearestNDInterpolator
                    interpolator = NearestNDInterpolator(valid_points, valid_values)
                    interpolated = interpolator(all_points)
                    
                    # Reshape to original dimensions
                    interpolated = interpolated.reshape(var_data.shape)
                    
                    ds_filled[var_name].values = interpolated
            
            # Count missing values after filling
            missing_after = np.isnan(ds_filled[var_name].values).sum()
            
            if missing_before > 0:
                filled_percent = (missing_before - missing_after) / missing_before * 100
                self.logger.debug(f"Filled {filled_percent:.1f}% gaps for {var_name}")
        
        return ds_filled
    
    def _add_metadata(self, ds: xr.Dataset, file_path: Path, 
                     data_type: str) -> xr.Dataset:
        """Add comprehensive metadata to dataset"""
        
        # Standard variable attributes
        var_attributes = {
            'pp': {
                'long_name': 'Primary Productivity',
                'units': 'mg C m^-2 d^-1',
                'standard_name': 'net_primary_production_of_biomass_expressed_as_carbon'
            },
            'chl': {
                'long_name': 'Chlorophyll-a concentration',
                'units': 'mg m^-3',
                'standard_name': 'mass_concentration_of_chlorophyll_a_in_sea_water'
            },
            'kd490': {
                'long_name': 'Diffuse attenuation coefficient at 490nm',
                'units': 'm^-1',
                'standard_name': 'volume_beam_attenuation_coefficient_of_radiative_flux_in_sea_water'
            },
            'zsd': {
                'long_name': 'Secchi disk depth',
                'units': 'm',
                'standard_name': 'sea_water_secchi_depth'
            },
            'uo': {
                'long_name': 'Eastward sea water velocity',
                'units': 'm s^-1',
                'standard_name': 'eastward_sea_water_velocity'
            },
            'vo': {
                'long_name': 'Northward sea water velocity',
                'units': 'm s^-1',
                'standard_name': 'northward_sea_water_velocity'
            }
        }
        
        # Add variable attributes
        for var_name in ds.data_vars:
            var_base = var_name.split('_')[0]
            if var_base in var_attributes:
                ds[var_name].attrs.update(var_attributes[var_base])
        
        # Add global attributes
        ds.attrs.update({
            'title': 'Gulf of Mexico Environmental Dataset',
            'institution': 'OceanPredict Framework',
            'source': str(file_path.name),
            'history': f'Processed by OceanPredict CMEMSProcessor on {pd.Timestamp.now().isoformat()}',
            'data_type': data_type,
            'processing_steps': [
                'Coordinate standardization',
                'Region subsetting',
                'Quality control',
                'Gap filling'
            ],
            'region': 'Gulf of Mexico',
            'contact': 'zahara@example.com',
            'references': 'CMEMS products: http://marine.copernicus.eu',
            'Conventions': 'CF-1.8',
            'license': 'Creative Commons Attribution 4.0 International'
        })
        
        return ds
    
    def calculate_derived_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Calculate derived environmental variables
        
        Parameters
        ----------
        ds : xr.Dataset
            Input dataset
            
        Returns
        -------
        xr.Dataset
            Dataset with derived variables
        """
        ds_derived = ds.copy()
        
        self.logger.info("Calculating derived variables...")
        
        # 1. Current magnitude and direction
        if all(v in ds_derived.data_vars for v in ['uo', 'vo']):
            # Current speed
            ds_derived['current_speed'] = np.sqrt(ds_derived['uo']**2 + ds_derived['vo']**2)
            ds_derived['current_speed'].attrs = {
                'long_name': 'Current speed',
                'units': 'm s^-1',
                'standard_name': 'sea_water_speed',
                'formula': 'sqrt(u^2 + v^2)'
            }
            
            # Current direction (degrees from North, 0 = North, 90 = East)
            ds_derived['current_direction'] = (
                np.arctan2(ds_derived['vo'], ds_derived['uo']) * 180 / np.pi
            )
            
            # Adjust to 0-360 range
            ds_derived['current_direction'] = (
                (90 - ds_derived['current_direction']) % 360
            )
            
            ds_derived['current_direction'].attrs = {
                'long_name': 'Current direction',
                'units': 'degrees',
                'standard_name': 'sea_water_velocity_to_direction',
                'description': 'Direction toward which current flows (0 = North, 90 = East)'
            }
            
            self.logger.debug("Calculated current speed and direction")
        
        # 2. Euphotic depth (depth of 1% light penetration)
        if 'kd490' in ds_derived.data_vars:
            ds_derived['euphotic_depth'] = 4.605 / ds_derived['kd490']
            ds_derived['euphotic_depth'].attrs = {
                'long_name': 'Euphotic depth',
                'units': 'm',
                'standard_name': 'euphotic_zone_depth',
                'formula': 'Zeu = ln(100) / Kd(490) = 4.605 / Kd(490)',
                'description': 'Depth of 1% surface irradiance'
            }
            self.logger.debug("Calculated euphotic depth")
        
        # 3. Primary productivity per chlorophyll (assimilation number)
        if all(v in ds_derived.data_vars for v in ['pp', 'chl']):
            ds_derived['assimilation_number'] = ds_derived['pp'] / (ds_derived['chl'] + 0.001)
            ds_derived['assimilation_number'].attrs = {
                'long_name': 'Assimilation number',
                'units': 'mg C mg Chl^-1 d^-1',
                'standard_name': 'primary_productivity_per_chlorophyll',
                'description': 'Primary productivity per unit chlorophyll (assimilation number)'
            }
            self.logger.debug("Calculated assimilation number")
        
        # 4. Water clarity index (normalized Secchi depth)
        if 'zsd' in ds_derived.data_vars:
            ds_derived['clarity_index'] = ds_derived['zsd'] / 30.0
            ds_derived['clarity_index'].attrs = {
                'long_name': 'Water clarity index',
                'units': 'dimensionless',
                'description': 'Normalized Secchi depth (1 = 30m, typical maximum)'
            }
            self.logger.debug("Calculated water clarity index")
        
        # 5. Mixed layer depth proxy (using temperature gradient if available)
        # Note: This is a simplified proxy - real MLD calculation requires temperature/salinity profiles
        
        # 6. Carbon flux estimates (simplified)
        if 'pp' in ds_derived.data_vars:
            # Convert to carbon flux (assuming 1 mg C m^-2 d^-1 = 0.001 g C m^-2 d^-1)
            ds_derived['carbon_flux'] = ds_derived['pp'] * 0.001
            ds_derived['carbon_flux'].attrs = {
                'long_name': 'Carbon flux',
                'units': 'g C m^-2 d^-1',
                'description': 'Carbon flux from primary productivity'
            }
            self.logger.debug("Calculated carbon flux")
        
        # 7. Productivity efficiency (PP per unit light)
        if all(v in ds_derived.data_vars for v in ['pp', 'kd490']):
            # Light utilization efficiency proxy
            ds_derived['light_utilization'] = ds_derived['pp'] * ds_derived['kd490']
            ds_derived['light_utilization'].attrs = {
                'long_name': 'Light utilization efficiency proxy',
                'units': 'mg C m^-1 d^-1',
                'description': 'Productivity per unit light attenuation'
            }
        
        # Add metadata about derived variables
        derived_vars = [v for v in ds_derived.data_vars if v not in ds.data_vars]
        if derived_vars:
            ds_derived.attrs['derived_variables'] = ', '.join(derived_vars)
            ds_derived.attrs['derived_variables_calculation_date'] = pd.Timestamp.now().isoformat()
        
        self.logger.info(f"✓ Calculated {len(derived_vars)} derived variables")
        
        return ds_derived
    
    def merge_datasets(self, datasets: List[xr.Dataset]) -> xr.Dataset:
        """
        Merge multiple datasets into a unified dataset
        
        Parameters
        ----------
        datasets : List[xr.Dataset]
            List of datasets to merge
            
        Returns
        -------
        xr.Dataset
            Merged unified dataset
        """
        if not datasets:
            raise ValueError("No datasets to merge")
        
        self.logger.info(f"Merging {len(datasets)} datasets...")
        
        # Use the dataset with currents as reference (usually has best temporal resolution)
        reference_idx = next(
            (i for i, ds in enumerate(datasets) 
             if all(v in ds.data_vars for v in ['uo', 'vo'])),
            0
        )
        
        reference = datasets[reference_idx]
        
        self.logger.info(f"Using dataset {reference_idx} as reference grid")
        
        # Align all datasets to reference grid
        aligned_datasets = []
        
        for i, ds in enumerate(datasets):
            try:
                if i == reference_idx:
                    aligned_datasets.append(ds)
                    continue
                
                self.logger.debug(f"Aligning dataset {i} to reference...")
                
                # Spatial interpolation to reference grid
                ds_aligned = ds.interp(
                    lat=reference.lat,
                    lon=reference.lon,
                    method='linear',
                    assume_sorted=True
                )
                
                # Temporal alignment if both have time dimension
                if 'time' in ds_aligned.dims and 'time' in reference.dims:
                    # Find common time range
                    common_time = np.intersect1d(
                        ds_aligned.time.values,
                        reference.time.values
                    )
                    
                    if len(common_time) > 0:
                        # Reindex to common times
                        ds_aligned = ds_aligned.sel(time=common_time)
                        ds_aligned = ds_aligned.reindex(
                            time=reference.time,
                            method='nearest',
                            tolerance=pd.Timedelta('1D')
                        )
                
                aligned_datasets.append(ds_aligned)
                self.logger.debug(f"✓ Dataset {i} aligned successfully")
                
            except Exception as e:
                self.logger.warning(f"Could not align dataset {i}: {e}")
        
        # Merge all aligned datasets
        self.logger.info("Merging aligned datasets...")
        
        # Use combine='by_coords' for intelligent merging
        with ProgressBar():
            try:
                merged_ds = xr.merge(aligned_datasets, combine='by_coords')
            except:
                # Fallback to simple merge
                merged_ds = xr.merge(aligned_datasets, combine_attrs='drop_conflicts')
        
        # Add comprehensive metadata
        merged_ds.attrs.update({
            'title': 'Gulf of Mexico Unified Environmental Dataset',
            'description': 'Merged dataset from multiple CMEMS products',
            'variables': ', '.join(list(merged_ds.data_vars)),
            'data_sources': ', '.join([ds.attrs.get('data_type', 'unknown') for ds in datasets]),
            'merge_date': pd.Timestamp.now().isoformat(),
            'merge_method': 'Spatial interpolation and temporal alignment',
            'reference_grid': f"{len(reference.lat)}×{len(reference.lon)}",
            'time_coverage': f"{merged_ds.time.values[0]} to {merged_ds.time.values[-1]}"
        })
        
        self.logger.info(f"✓ Merged dataset created with {len(merged_ds.data_vars)} variables")
        
        return merged_ds
    
    def save_dataset(self, ds: xr.Dataset, filename: str = "gulf_mexico_unified.nc") -> Path:
        """
        Save dataset with compression and proper encoding
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset to save
        filename : str
            Output filename
            
        Returns
        -------
        Path
            Path to saved file
        """
        output_path = Path("data/processed") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        # Compression settings for NetCDF4
        encoding = {}
        
        for var_name in ds.data_vars:
            encoding[var_name] = {
                'zlib': True,
                'complevel': 5,  # Compression level (1-9)
                'dtype': 'float32',
                '_FillValue': -9999.0,
                'chunksizes': self._get_optimal_chunksizes(ds[var_name])
            }
        
        # Coordinate encoding
        for coord in ds.coords:
            if coord in ['lat', 'lon']:
                encoding[coord] = {'dtype': 'float32'}
            elif coord == 'time':
                encoding[coord] = {'dtype': 'float64', 'units': 'days since 1950-01-01'}
        
        # Save with progress bar for large datasets
        self.logger.info(f"Saving dataset to {output_path}...")
        
        with ProgressBar():
            ds.to_netcdf(
                output_path,
                encoding=encoding,
                unlimited_dims=['time'] if 'time' in ds.dims else None
            )
        
        # Report file size
        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        self.logger.info(f"✓ Dataset saved: {file_size_mb:.1f} MB")
        self.logger.info(f"  Variables: {len(ds.data_vars)}")
        if 'time' in ds.dims:
            self.logger.info(f"  Time steps: {len(ds.time)}")
        self.logger.info(f"  Grid: {len(ds.lat)}×{len(ds.lon)}")
        
        return output_path
    
    def _get_optimal_chunksizes(self, data_array: xr.DataArray) -> tuple:
        """Calculate optimal chunk sizes for NetCDF storage"""
        chunk_sizes = []
        
        for dim in data_array.dims:
            dim_size = data_array.sizes[dim]
            
            if dim == 'time':
                # Chunk time dimension moderately
                chunk_sizes.append(min(30, dim_size))
            elif dim in ['lat', 'lon']:
                # Chunk spatial dimensions fully for spatial operations
                chunk_sizes.append(dim_size)
            else:
                # Default chunk size
                chunk_sizes.append(min(10, dim_size))
        
        return tuple(chunk_sizes)
    
    def run_processing_pipeline(self, download_dir: Optional[str] = None) -> xr.Dataset:
        """
        Run complete processing pipeline
        
        Parameters
        ----------
        download_dir : str, optional
            Directory with downloaded files
            
        Returns
        -------
        xr.Dataset
            Processed unified dataset
        """
        self.logger.info("=" * 70)
        self.logger.info("OCEANPREDICT PROCESSING PIPELINE")
        self.logger.info("=" * 70)
        
        # Step 1: Organize downloaded files
        if download_dir:
            self.logger.info("Step 1: Organizing downloaded files...")
            # Assuming files are already in Google Drive structure
            # You would add organization logic here if needed
            pass
        
        # Step 2: Process individual datasets
        self.logger.info("Step 2: Processing individual datasets...")
        
        datasets = []
        data_types = ['physics_currents', 'bgc_pp', 'bgc_plankton', 'bgc_optics']
        
        for data_type in data_types:
            data_dir = Path(f"data/raw/{data_type}")
            
            if data_dir.exists():
                nc_files = list(data_dir.glob("*.nc"))
                
                if nc_files:
                    # Process the first file of each type
                    file_path = nc_files[0]
                    ds = self.process_single_file(file_path, data_type)
                    
                    if ds is not None:
                        datasets.append(ds)
                        self.logger.info(f"  Processed {data_type}: {list(ds.data_vars)}")
                else:
                    self.logger.warning(f"No .nc files found in {data_dir}")
        
        if not datasets:
            raise ValueError("No valid datasets processed")
        
        # Step 3: Calculate derived variables
        self.logger.info("Step 3: Calculating derived variables...")
        datasets_with_derived = []
        
        for ds in datasets:
            ds_derived = self.calculate_derived_variables(ds)
            datasets_with_derived.append(ds_derived)
        
        # Step 4: Merge datasets
        self.logger.info("Step 4: Merging datasets...")
        merged_ds = self.merge_datasets(datasets_with_derived)
        
        # Step 5: Final quality check
        self.logger.info("Step 5: Final quality check...")
        self._final_quality_check(merged_ds)
        
        # Step 6: Save dataset
        self.logger.info("Step 6: Saving processed dataset...")
        self.save_dataset(merged_ds)
        
        self.logger.info("=" * 70)
        self.logger.info("PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 70)
        
        return merged_ds
    
    def _final_quality_check(self, ds: xr.Dataset) -> None:
        """Perform final quality check on merged dataset"""
        self.logger.info("Final quality check...")
        
        # Check for NaN values
        nan_report = {}
        for var_name in ds.data_vars:
            nan_count = np.isnan(ds[var_name].values).sum()
            total_count = ds[var_name].size
            nan_percent = nan_count / total_count * 100 if total_count > 0 else 0
            
            if nan_percent > 10:  # Warning threshold
                self.logger.warning(f"  {var_name}: {nan_percent:.1f}% NaN values")
            else:
                self.logger.debug(f"  {var_name}: {nan_percent:.1f}% NaN values")
            
            nan_report[var_name] = nan_percent
        
        # Check temporal consistency
        if 'time' in ds.dims:
            time_diff = np.diff(ds.time.values)
            unique_diffs = np.unique(time_diff)
            
            if len(unique_diffs) == 1:
                self.logger.info(f"  Time series: Regular, Δt = {unique_diffs[0]}")
            else:
                self.logger.warning(f"  Time series: Irregular, unique Δt = {len(unique_diffs)}")
        
        # Check spatial coverage
        if all(dim in ds.dims for dim in ['lat', 'lon']):
            self.logger.info(f"  Spatial grid: {len(ds.lat)}×{len(ds.lon)} points")
            self.logger.info(f"  Spatial extent: {ds.lat.values.min():.2f} to {ds.lat.values.max():.2f}°N, "
                           f"{ds.lon.values.min():.2f} to {ds.lon.values.max():.2f}°E")
        
        # Save quality report
        report_path = Path("results/reports/quality_check.json")
        report_path.parent.mkdir(exist_ok=True)
        
        import json
        with open(report_path, 'w') as f:
            json.dump(nan_report, f, indent=2)
        
        self.logger.info(f"✓ Quality report saved to {report_path}")
