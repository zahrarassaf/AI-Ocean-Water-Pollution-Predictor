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
                if name in ds.data_v
