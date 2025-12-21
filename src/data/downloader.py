"""
Data downloader for OceanPredict with support for multiple sources
"""

import requests
import gdown
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import zipfile
import tarfile
from tqdm import tqdm
import hashlib
import json

from ..utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class DataDownloader:
    """Download and manage oceanographic data from multiple sources"""
    
    def __init__(self):
        self.config = ConfigLoader()
        self.download_dir = Path("data/raw")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset configuration
        self.dataset_config = self.config.get("datasets", {}, config="datasets")
        
    def download_sample_data(self) -> Dict[str, Path]:
        """
        Download sample datasets for testing
        
        Returns
        -------
        Dict[str, Path]
            Dictionary mapping data type to file path
        """
        logger.info("Downloading sample datasets...")
        
        sample_config = self.dataset_config.get("sample", {})
        base_url = sample_config.get("base_url", "")
        
        downloaded_files = {}
        
        for data_type, filename in sample_config.get("files", {}).items():
            url = f"{base_url}{filename}"
            output_path = self.download_dir / filename
            
            try:
                self._download_file(url, output_path)
                
                # Verify download
                if output_path.exists():
                    downloaded_files[data_type] = output_path
                    size_mb = output_path.stat().st_size / (1024 ** 2)
                    logger.info(f"✓ Downloaded {data_type}: {size_mb:.1f} MB")
                else:
                    logger.error(f"Failed to download {data_type}")
                    
            except Exception as e:
                logger.error(f"Error downloading {data_type}: {e}")
        
        return downloaded_files
    
    def download_full_data(self, data_types: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        Download full datasets from Google Drive
        
        Parameters
        ----------
        data_types : List[str], optional
            Specific data types to download, defaults to all
            
        Returns
        -------
        Dict[str, Path]
            Dictionary mapping data type to file path
        """
        logger.info("Downloading full datasets from Google Drive...")
        
        full_config = self.dataset_config.get("full", {})
        organization = self.dataset_config.get("organization", {})
        
        if data_types is None:
            # Download all datasets
            data_types = list(full_config.get("files", {}).keys())
        
        downloaded_files = {}
        
        for data_type in data_types:
            if data_type not in full_config.get("files", {}):
                logger.warning(f"Dataset {data_type} not found in configuration")
                continue
            
            dataset_info = full_config["files"][data_type]
            url = dataset_info.get("url")
            
            if not url:
                logger.error(f"No URL for {data_type}")
                continue
            
            # Determine target directory
            target_dir = None
            for dir_name, dir_types in organization.items():
                if data_type in dir_types:
                    target_dir = self.download_dir / dir_name
                    break
            
            if not target_dir:
                target_dir = self.download_dir / data_type
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine filename
            if data_type == "currents":
                filename = "currents.nc"
            elif data_type == "primary_productivity":
                filename = "pp.nc"
            elif data_type == "chlorophyll":
                filename = "chl.nc"
            else:
                filename = f"{data_type}.nc"
            
            output_path = target_dir / filename
            
            try:
                logger.info(f"Downloading {data_type} ({dataset_info.get('size_gb', 'unknown')} GB)...")
                
                # Download from Google Drive
                gdown.download(url, str(output_path), quiet=False)
                
                # Verify download
                if output_path.exists():
                    downloaded_files[data_type] = output_path
                    size_mb = output_path.stat().st_size / (1024 ** 2)
                    logger.info(f"✓ Downloaded {data_type}: {size_mb:.1f} MB")
                    
                    # Organize into proper directory structure
                    self._organize_file(output_path, data_type)
                else:
                    logger.error(f"Failed to download {data_type}")
                    
            except Exception as e:
                logger.error(f"Error downloading {data_type}: {e}")
        
        return downloaded_files
    
    def _download_file(self, url: str, output_path: Path, chunk_size: int = 8192) -> None:
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
    
    def _organize_file(self, file_path: Path, data_type: str) -> None:
        """Organize downloaded file into proper directory"""
        organization = self.dataset_config.get("organization", {})
        
        for dir_name, dir_types in organization.items():
            if data_type in dir_types:
                target_dir = self.download_dir / dir_name
                target_dir.mkdir(exist_ok=True)
                
                # Move file to target directory
                if data_type == "currents":
                    new_name = "currents.nc"
                elif data_type == "primary_productivity":
                    new_name = "pp.nc"
                elif data_type == "chlorophyll":
                    new_name = "chl.nc"
                else:
                    new_name = f"{data_type}.nc"
                
                target_path = target_dir / new_name
                file_path.rename(target_path)
                logger.info(f"  Organized → {dir_name}/{new_name}")
                break
    
    def verify_integrity(self, file_path: Path, expected_md5: Optional[str] = None) -> bool:
        """Verify file integrity using MD5 checksum"""
        if not file_path.exists():
            return False
        
        if expected_md5:
            # Calculate MD5
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            calculated_md5 = hash_md5.hexdigest()
            
            if calculated_md5 != expected_md5:
                logger.warning(f"MD5 mismatch for {file_path.name}")
                return False
        
        # Try to open the file to verify it's valid
        try:
            with xr.open_dataset(file_path) as ds:
                logger.debug(f"✓ Valid NetCDF file: {file_path.name}")
                return True
        except Exception as e:
            logger.error(f"Invalid NetCDF file {file_path.name}: {e}")
            return False
    
    def create_synthetic_data(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Create synthetic data for testing and development
        
        Returns
        -------
        Dict[str, Path]
            Dictionary of created synthetic files
        """
        if output_dir is None:
            output_dir = self.download_dir / "synthetic"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating synthetic datasets...")
        
        # Create time coordinates
        times = pd.date_range('2023-01-01', periods=10, freq='D')
        lats = np.linspace(18, 30, 100)
        lons = np.linspace(-98, -88, 100)
        
        synthetic_files = {}
        
        # 1. Ocean currents
        logger.info("Creating synthetic currents...")
        uo = np.random.randn(len(times), len(lats), len(lons)) * 0.1
        vo = np.random.randn(len(times), len(lats), len(lons)) * 0.1
        
        ds_currents = xr.Dataset(
            {
                'uo': (['time', 'lat', 'lon'], uo.astype('float32')),
                'vo': (['time', 'lat', 'lon'], vo.astype('float32'))
            },
            coords={
                'time': times,
                'lat': lats,
                'lon': lons
            },
            attrs={
                'description': 'Synthetic ocean currents',
                'units': 'm/s',
                'creator': 'OceanPredict Synthetic Data Generator'
            }
        )
        
        currents_path = output_dir / "physics_currents" / "currents.nc"
        currents_path.parent.mkdir(exist_ok=True)
        ds_currents.to_netcdf(currents_path)
        synthetic_files['currents'] = currents_path
        
        # 2. Primary productivity
        logger.info("Creating synthetic primary productivity...")
        pp = np.random.gamma(shape=2, scale=50, size=(len(times), len(lats), len(lons)))
        
        ds_pp = xr.Dataset(
            {
                'pp': (['time', 'lat', 'lon'], pp.astype('float32'))
            },
            coords={
                'time': times,
                'lat': lats,
                'lon': lons
            },
            attrs={
                'description': 'Synthetic primary productivity',
                'units': 'mg C m^-2 d^-1',
                'creator': 'OceanPredict Synthetic Data Generator'
            }
        )
        
        pp_path = output_dir / "bgc_pp" / "pp.nc"
        pp_path.parent.mkdir(exist_ok=True)
        ds_pp.to_netcdf(pp_path)
        synthetic_files['primary_productivity'] = pp_path
        
        # 3. Chlorophyll
        logger.info("Creating synthetic chlorophyll...")
        chl = np.random.gamma(shape=1.5, scale=0.5, size=(len(times), len(lats), len(lons)))
        
        ds_chl = xr.Dataset(
            {
                'chl': (['time', 'lat', 'lon'], chl.astype('float32'))
            },
            coords={
                'time': times,
                'lat': lats,
                'lon': lons
            },
            attrs={
                'description': 'Synthetic chlorophyll concentration',
                'units': 'mg m^-3',
                'creator': 'OceanPredict Synthetic Data Generator'
            }
        )
        
        chl_path = output_dir / "bgc_plankton" / "chl.nc"
        chl_path.parent.mkdir(exist_ok=True)
        ds_chl.to_netcdf(chl_path)
        synthetic_files['chlorophyll'] = chl_path
        
        # 4. Optical properties
        logger.info("Creating synthetic optical properties...")
        kd490 = np.random.gamma(shape=1.2, scale=0.1, size=(len(times), len(lats), len(lons)))
        zsd = 1.0 / (kd490 + 0.01)  # Secchi depth approximation
        
        ds_optics = xr.Dataset(
            {
                'kd490': (['time', 'lat', 'lon'], kd490.astype('float32')),
                'zsd': (['time', 'lat', 'lon'], zsd.astype('float32'))
            },
            coords={
                'time': times,
                'lat': lats,
                'lon': lons
            },
            attrs={
                'description': 'Synthetic optical properties',
                'creator': 'OceanPredict Synthetic Data Generator'
            }
        )
        
        optics_path = output_dir / "bgc_optics" / "optics.nc"
        optics_path.parent.mkdir(exist_ok=True)
        ds_optics.to_netcdf(optics_path)
        synthetic_files['kd490'] = optics_path
        synthetic_files['zsd'] = optics_path
        
        logger.info("✓ Synthetic datasets created successfully")
        
        # Save metadata
        metadata = {
            'creation_date': pd.Timestamp.now().isoformat(),
            'datasets': list(synthetic_files.keys()),
            'grid': {
                'time_steps': len(times),
                'lat_points': len(lats),
                'lon_points': len(lons),
                'time_range': [str(times[0]), str(times[-1])],
                'lat_range': [float(lats.min()), float(lats.max())],
                'lon_range': [float(lons.min()), float(lons.max())]
            }
        }
        
        metadata_path = output_dir / "synthetic_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return synthetic_files
    
    def get_download_status(self) -> Dict[str, Any]:
        """Get current download status"""
        status = {
            'downloaded': {},
            'missing': [],
            'total_size_gb': 0
        }
        
        organization = self.dataset_config.get("organization", {})
        
        for dir_name, dir_types in organization.items():
            dir_path = self.download_dir / dir_name
            
            if dir_path.exists():
                for file_path in dir_path.glob("*.nc"):
                    size_mb = file_path.stat().st_size / (1024 ** 2)
                    status['downloaded'][file_path.name] = {
                        'path': str(file_path),
                        'size_mb': size_mb,
                        'directory': dir_name
                    }
                    status['total_size_gb'] += size_mb / 1024
        
        status['total_size_gb'] = round(status['total_size_gb'], 2)
        
        return status
