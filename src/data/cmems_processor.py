"""
CMEMS Data Processor for environmental data science portfolio
Handles downloading, processing, and organizing CMEMS data
"""

import xarray as xr
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

from ..utils.config_loader import ConfigLoader

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class CMEMSDataProcessor:
    """Processor for CMEMS oceanographic data"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CMEMS data processor
        
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary
        """
        if config is None:
            config_loader = ConfigLoader()
            config = config_loader.get('cmems_variables', {})
        
        self.config = config
        self.region = config.get('region', {})
        self.time_period = config.get('time_period', {})
        
        # Initialize directories
        self.setup_directories()
        
        # Variable mappings
        self.variable_mapping = {
            'primary_productivity': ['nppv', 'pp', 'primary_production'],
            'chlorophyll': ['chl', 'chlorophyll_a', 'CHLA'],
            'diffuse_attenuation': ['kd490', 'Kd_490'],
            'secchi_depth': ['zsd', 'secchi_depth'],
            'backscattering': ['bbp', 'bbp443'],
            'cdom': ['cdm', 'adg443'],
            'eastward_current': ['uo', 'eastward_sea_water_velocity'],
            'northward_current': ['vo', 'northward_sea_water_velocity']
        }
    
    def setup_directories(self) -> None:
        """Create necessary directory structure"""
        dirs = [
            Path("data/raw/bgc_optics"),
            Path("data/raw/bgc_plankton"),
            Path("data/raw/bgc_pp"),
            Path("data/raw/bgc_transport"),
            Path("data/raw/physics_currents"),
            Path("data/processed"),
            Path("results/figures"),
            Path("results/models"),
            Path("results/reports")
        ]
        
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("Directory structure created")
    
    def organize_downloaded_files(self, download_dir: Union[str, Path]) -> Dict[str, List[Path]]:
        """
        Organize downloaded CMEMS files into structured directories
        
        Parameters
        ----------
        download_dir : str or Path
            Directory containing downloaded files
            
        Returns
        -------
        Dict[str, List[Path]]
            Organized file paths by data type
        """
        download_dir = Path(download_dir)
        organized_files = {}
        
        # Define patterns for each data type
        patterns = {
            'bgc_optics': ['*kd490*', '*zsd*', '*bbp*', '*cdm*'],
            'bgc_plankton': ['*chl*', '*chlorophyll*'],
            'bgc_pp': ['*nppv*', '*pp*', '*primary_production*'],
            'physics_currents': ['*uo*', '*vo*', '*current*']
        }
        
        for data_type, pattern_list in patterns.items():
            target_dir = Path(f"data/raw/{data_type}")
            organized_files[data_type] = []
            
            for pattern in pattern_list:
                for file_path in download_dir.glob(pattern):
                    if file_path.suffix == '.nc':
                        dest_path = target_dir / file_path.name
                        file_path.rename(dest_path)
                        organized_files[data_type].append(dest_path)
                        logger.debug(f"Moved {file_path.name} to {data_type}")
        
        # Log summary
        total_files = sum(len(files) for files in organized_files.values())
        logger.info(f"Organized {total_files} files into structured directories")
        
        return organized_files
    
    def load_dataset(self, file_path: Union[str, Path], 
                    data_type: str) -> Optional[xr.Dataset]:
        """
        Load and preprocess a single dataset
        
        Parameters
        ----------
        file_path : str or Path
            Path to NetCDF file
        data_type : str
            Type of data (bgc_optics, physics_currents, etc.)
            
        Returns
        -------
        xr.Dataset or None
            Processed dataset or None if loading fails
        """
        try:
            logger.info(f"Loading dataset: {Path(file_path).name}")
            
            # Open dataset
            ds = xr.open_dataset(file_path, chunks={'time': 10})
            
            # Rename coordinates to standard names
            ds = self._standardize_coordinates(ds)
            
            # Subset to region
            ds = self._subset_to_region(ds)
            
            # Subset to time period
            ds = self._subset_to_time_period(ds)
            
            # Extract relevant variables
            ds = self._extract_variables(ds, data_type)
            
            # Apply quality control
            ds = self._apply_quality_control(ds)
            
            return ds
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _standardize_coordinates(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize coordinate names"""
        rename_dict = {}
        
        # Latitude
        if 'latitude' in ds.dims:
            rename_dict['latitude'] = 'lat'
        elif 'lat' not in ds.dims and 'nav_lat' in ds.dims:
            rename_dict['nav_lat'] = 'lat'
        
        # Longitude
        if 'longitude' in ds.dims:
            rename_dict['longitude'] = 'lon'
        elif 'lon' not in ds.dims and 'nav_lon' in ds.dims:
            rename_dict['nav_lon'] = 'lon'
        
        # Time
        if 'time' not in ds.dims and 'TIME' in ds.dims:
            rename_dict['TIME'] = 'time'
        
        if rename_dict:
            ds = ds.rename(rename_dict)
        
        return ds
    
    def _subset_to_region(self, ds: xr.Dataset) -> xr.Dataset:
        """Subset dataset to study region"""
        if 'lat' in ds.dims and 'lon' in ds.dims:
            lat_min = self.region['latitude']['min']
            lat_max = self.region['latitude']['max']
            lon_min = self.region['longitude']['min']
            lon_max = self.region['longitude']['max']
            
            # Handle longitude wrapping if necessary
            if ds.lon.min() < 0 and lon_min < 0:
                # Both in negative longitude (Western hemisphere)
                ds = ds.sel(
                    lat=slice(lat_min, lat_max),
                    lon=slice(lon_min, lon_max)
                )
            else:
                # Convert to 0-360 if necessary
                if ds.lon.min() >= 0:
                    # Convert region bounds to 0-360
                    lon_min = lon_min % 360
                    lon_max = lon_max % 360
                
                ds = ds.sel(
                    lat=slice(lat_min, lat_max),
                    lon=slice(lon_min, lon_max)
                )
        
        return ds
    
    def _subset_to_time_period(self, ds: xr.Dataset) -> xr.Dataset:
        """Subset dataset to study time period"""
        if 'time' in ds.dims:
            start_date = pd.Timestamp(self.time_period['start'])
            end_date = pd.Timestamp(self.time_period['end'])
            
            ds = ds.sel(time=slice(start_date, end_date))
        
        return ds
    
    def _extract_variables(self, ds: xr.Dataset, data_type: str) -> xr.Dataset:
        """Extract and rename relevant variables"""
        extracted_vars = {}
        
        # Get target variables for this data type
        if data_type == 'bgc_optics':
            target_vars = ['kd490', 'zsd', 'bbp', 'cdm']
        elif data_type == 'bgc_plankton':
            target_vars = ['chl']
        elif data_type == 'bgc_pp':
            target_vars = ['pp']
        elif data_type == 'physics_currents':
            target_vars = ['uo', 'vo']
        else:
            target_vars = []
        
        for target_var in target_vars:
            # Find the variable in the dataset
            found_var = self._find_variable(ds, target_var)
            if found_var:
                extracted_vars[target_var] = ds[found_var]
        
        # Create new dataset with extracted variables
        if extracted_vars:
            ds_extracted = xr.Dataset(extracted_vars)
            ds_extracted.attrs.update(ds.attrs)
            return ds_extracted
        
        return ds
    
    def _find_variable(self, ds: xr.Dataset, target_var: str) -> Optional[str]:
        """Find variable in dataset using mapping"""
        if target_var in self.variable_mapping:
            possible_names = self.variable_mapping[target_var]
            
            for name in possible_names:
                if name in ds.data_vars:
                    return name
            
            # Try case-insensitive search
            ds_vars_lower = [v.lower() for v in ds.data_vars]
            for name in possible_names:
                if name.lower() in ds_vars_lower:
                    idx = ds_vars_lower.index(name.lower())
                    return list(ds.data_vars)[idx]
        
        return None
    
    def _apply_quality_control(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply quality control filters"""
        ds_qc = ds.copy()
        
        # Define valid ranges for each variable
        valid_ranges = {
            'pp': (0, 1000),      # mg C m^-2 d^-1
            'chl': (0.01, 30),    # mg m^-3
            'kd490': (0.01, 5),   # m^-1
            'zsd': (0.1, 50),     # m
            'uo': (-5, 5),        # m/s
            'vo': (-5, 5)         # m/s
        }
        
        for var_name in ds_qc.data_vars:
            if var_name in valid_ranges:
                vmin, vmax = valid_ranges[var_name]
                ds_qc[var_name] = ds_qc[var_name].where(
                    (ds_qc[var_name] >= vmin) & (ds_qc[var_name] <= vmax)
                )
        
        # Fill small gaps with interpolation
        for var_name in ds_qc.data_vars:
            if 'time' in ds_qc[var_name].dims:
                # Temporal interpolation
                ds_qc[var_name] = ds_qc[var_name].interpolate_na(
                    dim='time', method='linear', limit=2
                )
            
            # Spatial interpolation
            if 'lat' in ds_qc[var_name].dims and 'lon' in ds_qc[var_name].dims:
                ds_qc[var_name] = ds_qc[var_name].interpolate_na(
                    dim='lat', method='linear'
                )
                ds_qc[var_name] = ds_qc[var_name].interpolate_na(
                    dim='lon', method='linear'
                )
        
        return ds_qc
    
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
        
        # Current magnitude and direction
        if all(v in ds_derived for v in ['uo', 'vo']):
            ds_derived['current_speed'] = np.sqrt(ds_derived['uo']**2 + ds_derived['vo']**2)
            ds_derived['current_speed'].attrs = {
                'long_name': 'Current speed',
                'units': 'm s^-1',
                'standard_name': 'sea_water_speed'
            }
            
            ds_derived['current_direction'] = np.arctan2(ds_derived['vo'], ds_derived['uo']) * 180 / np.pi
            ds_derived['current_direction'].attrs = {
                'long_name': 'Current direction',
                'units': 'degrees',
                'standard_name': 'sea_water_velocity_to_direction'
            }
        
        # Euphotic depth from diffuse attenuation
        if 'kd490' in ds_derived:
            ds_derived['euphotic_depth'] = 4.605 / ds_derived['kd490']
            ds_derived['euphotic_depth'].attrs = {
                'long_name': 'Euphotic depth',
                'units': 'm',
                'description': 'Depth of 1% surface irradiance (Zeu = 4.605/Kd)'
            }
        
        # Primary productivity per chlorophyll (assimilation number)
        if all(v in ds_derived for v in ['pp', 'chl']):
            ds_derived['assimilation_number'] = ds_derived['pp'] / (ds_derived['chl'] + 0.001)
            ds_derived['assimilation_number'].attrs = {
                'long_name': 'Assimilation number',
                'units': 'mg C mg Chl^-1 d^-1',
                'description': 'Primary productivity per unit chlorophyll'
            }
        
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
            Merged dataset
        """
        if not datasets:
            raise ValueError("No datasets to merge")
        
        # Use first dataset as reference
        reference = datasets[0]
        
        # Align all datasets to reference grid
        aligned_datasets = []
        for ds in datasets:
            try:
                # Interpolate to reference grid
                ds_aligned = ds.interp(
                    lat=reference.lat,
                    lon=reference.lon,
                    method='linear'
                )
                
                # Align time dimension if present
                if 'time' in ds_aligned.dims and 'time' in reference.dims:
                    ds_aligned = ds_aligned.reindex(
                        time=reference.time,
                        method='nearest'
                    )
                
                aligned_datasets.append(ds_aligned)
                
            except Exception as e:
                logger.warning(f"Could not align dataset: {e}")
                continue
        
        # Merge all datasets
        merged_ds = xr.merge(aligned_datasets, combine_attrs='no_conflicts')
        
        # Add metadata
        merged_ds.attrs.update({
            'title': 'Gulf of Mexico Environmental Dataset',
            'description': 'Unified dataset from CMEMS',
            'region': 'Gulf of Mexico',
            'processing_date': pd.Timestamp.now().isoformat(),
            'processing_software': 'CMEMSDataProcessor v1.0'
        })
        
        return merged_ds
    
    def process_pipeline(self, download_dir: Union[str, Path]) -> xr.Dataset:
        """
        Run complete processing pipeline
        
        Parameters
        ----------
        download_dir : str or Path
            Directory containing downloaded files
            
        Returns
        -------
        xr.Dataset
            Processed and merged dataset
        """
        logger.info("Starting CMEMS data processing pipeline")
        
        # Step 1: Organize files
        organized_files = self.organize_downloaded_files(download_dir)
        
        # Step 2: Load and process datasets
        datasets = []
        for data_type, file_paths in organized_files.items():
            if file_paths:
                # Process first file of each type
                ds = self.load_dataset(file_paths[0], data_type)
                if ds is not None:
                    datasets.append(ds)
                    logger.info(f"Processed {data_type}: {len(ds.data_vars)} variables")
        
        # Step 3: Merge datasets
        if datasets:
            merged_ds = self.merge_datasets(datasets)
            
            # Step 4: Calculate derived variables
            merged_ds = self.calculate_derived_variables(merged_ds)
            
            # Step 5: Save processed dataset
            output_path = Path("data/processed/gulf_mexico_processed.nc")
            self.save_dataset(merged_ds, output_path)
            
            logger.info(f"Pipeline completed. Dataset saved to {output_path}")
            return merged_ds
        
        else:
            logger.error("No datasets were successfully processed")
            raise ValueError("Processing pipeline failed")
    
    def save_dataset(self, ds: xr.Dataset, output_path: Union[str, Path]) -> None:
        """
        Save dataset to NetCDF file
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset to save
        output_path : str or Path
            Output file path
        """
        output_path = Path(output_path)
        
        # Set encoding for compression
        encoding = {}
        for var_name in ds.data_vars:
            encoding[var_name] = {
                'zlib': True,
                'complevel': 5,
                'dtype': 'float32',
                '_FillValue': -9999.0
            }
        
        # Save dataset
        ds.to_netcdf(output_path, encoding=encoding)
        logger.info(f"Dataset saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
