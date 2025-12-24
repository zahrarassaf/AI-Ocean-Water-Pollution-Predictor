"""
Google Drive integration for NetCDF data loading
"""

import os
import io
import gdown
import zipfile
import tarfile
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
import logging
import pandas as pd
import numpy as np
import xarray as xr
import tempfile
import shutil
from tqdm import tqdm
import requests

from ..config.constants import DATA_DIR
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class NetCDFDriveDownloader:
    """Specialized downloader for NetCDF files from Google Drive"""
    
    def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB chunks
        self.chunk_size = chunk_size
        self.session = requests.Session()
        logger.info("Initialized NetCDFDriveDownloader")
    
    def extract_file_id(self, url: str) -> Optional[str]:
        """Extract file ID from various Google Drive URL formats"""
        import re
        
        patterns = [
            r'/file/d/([a-zA-Z0-9_-]+)',  # /file/d/FILE_ID/view
            r'id=([a-zA-Z0-9_-]+)',        # ?id=FILE_ID
            r'/d/([a-zA-Z0-9_-]+)/',       # /d/FILE_ID/
            r'/d/([a-zA-Z0-9_-]+)$',       # /d/FILE_ID
            r'open\?id=([a-zA-Z0-9_-]+)',  # open?id=FILE_ID
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def get_direct_download_url(self, file_id: str) -> str:
        """Get direct download URL for Google Drive file"""
        return f"https://drive.google.com/uc?id={file_id}&export=download"
    
    def download_netcdf(self, 
                       url: str, 
                       output_path: Union[str, Path],
                       description: str = "Downloading") -> Path:
        """
        Download NetCDF file with progress bar
        
        Parameters
        ----------
        url : str
            Google Drive URL
        output_path : str or Path
            Output file path
        description : str
            Description for progress bar
            
        Returns
        -------
        Path
            Path to downloaded file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract file ID
        file_id = self.extract_file_id(url)
        if not file_id:
            raise ValueError(f"Could not extract file ID from URL: {url}")
        
        # Get direct download URL
        download_url = self.get_direct_download_url(file_id)
        
        logger.info(f"Downloading {description} from {url}")
        logger.info(f"File ID: {file_id}")
        logger.info(f"Output: {output_path}")
        
        try:
            # Use gdown for reliable download
            gdown.download(
                download_url,
                str(output_path),
                quiet=False,
                fuzzy=True,
                resume=True
            )
            
            # Verify the file is a valid NetCDF
            self._validate_netcdf(output_path)
            
            logger.info(f"✓ Successfully downloaded {output_path.name}")
            return output_path
            
        except Exception as e:
            logger.error(f"✗ Failed to download {url}: {e}")
            
            # Try alternative method
            return self._download_with_requests(download_url, output_path, description)
    
    def _download_with_requests(self, 
                               url: str, 
                               output_path: Path,
                               description: str) -> Path:
        """Alternative download method using requests"""
        try:
            # Start session
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(output_path, 'wb') as f, tqdm(
                desc=description,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            self._validate_netcdf(output_path)
            logger.info(f"✓ Downloaded via requests: {output_path.name}")
            return output_path
            
        except Exception as e:
            logger.error(f"✗ Requests download also failed: {e}")
            
            # Clean up partial file
            if output_path.exists():
                output_path.unlink()
            
            raise
    
    def _validate_netcdf(self, filepath: Path):
        """Validate that file is a proper NetCDF"""
        try:
            # Quick validation by trying to open it
            with xr.open_dataset(filepath, engine='netcdf4') as ds:
                # Just check if it can be opened
                logger.info(f"NetCDF validation: {len(ds.data_vars)} variables, "
                          f"dims: {dict(ds.dims)}")
                return True
        except Exception as e:
            logger.error(f"Invalid NetCDF file {filepath}: {e}")
            raise ValueError(f"File is not a valid NetCDF: {e}")
    
    def download_project_netcdf_files(self,
                                     urls: List[str],
                                     output_dir: Union[str, Path] = DATA_DIR / 'raw',
                                     filenames: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        Download all NetCDF files for the project
        
        Parameters
        ----------
        urls : list
            List of Google Drive URLs
        output_dir : str or Path
            Output directory
        filenames : list, optional
            Custom filenames
            
        Returns
        -------
        dict
            Dictionary mapping names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_files = {}
        
        for i, url in enumerate(urls):
            try:
                # Generate filename
                if filenames and i < len(filenames):
                    filename = filenames[i]
                    # Ensure .nc extension
                    if not filename.endswith('.nc'):
                        filename += '.nc'
                else:
                    # Try to extract name from URL or use generic
                    filename = self._extract_filename_from_url(url) or f"dataset_{i+1}.nc"
                
                output_path = output_dir / filename
                
                # Skip if already exists and valid
                if output_path.exists():
                    try:
                        self._validate_netcdf(output_path)
                        logger.info(f"✓ File already exists and is valid: {filename}")
                        downloaded_files[filename] = output_path
                        continue
                    except:
                        logger.warning(f"Existing file is invalid, re-downloading: {filename}")
                
                # Download file
                description = f"File {i+1}/{len(urls)}"
                downloaded_path = self.download_netcdf(url, output_path, description)
                
                downloaded_files[filename] = downloaded_path
                
            except Exception as e:
                logger.error(f"Failed to download file {i+1}: {e}")
                continue
        
        logger.info(f"Download completed: {len(downloaded_files)}/{len(urls)} files")
        return downloaded_files
    
    def _extract_filename_from_url(self, url: str) -> Optional[str]:
        """Extract filename from URL parameters"""
        import urllib.parse
        
        try:
            parsed = urllib.parse.urlparse(url)
            query = urllib.parse.parse_qs(parsed.query)
            
            # Check for filename in query parameters
            if 'name' in query:
                return query['name'][0]
            
            # Try to get from path
            path = parsed.path
            if path.endswith('/view'):
                path = path[:-5]  # Remove /view
            
            # Extract last part
            filename = path.split('/')[-1]
            if filename and '.' in filename:
                return filename
            
        except:
            pass
        
        return None
    
    def merge_netcdf_files(self,
                          filepaths: List[Path],
                          output_path: Union[str, Path],
                          merge_dim: str = 'time') -> Path:
        """
        Merge multiple NetCDF files
        
        Parameters
        ----------
        filepaths : list
            List of NetCDF file paths
        output_path : str or Path
            Output merged file path
        merge_dim : str
            Dimension to merge along
            
        Returns
        -------
        Path
            Path to merged file
        """
        output_path = Path(output_path)
        
        logger.info(f"Merging {len(filepaths)} NetCDF files...")
        
        try:
            # Open all datasets
            datasets = []
            for filepath in filepaths:
                ds = xr.open_dataset(filepath, engine='netcdf4')
                datasets.append(ds)
                logger.info(f"  Loaded {filepath.name}: {dict(ds.dims)}")
            
            # Merge datasets
            merged = xr.concat(datasets, dim=merge_dim)
            
            # Sort by time if time dimension exists
            if 'time' in merged.dims:
                merged = merged.sortby('time')
            
            # Save merged dataset
            merged.to_netcdf(output_path)
            
            # Close all datasets
            for ds in datasets:
                ds.close()
            
            logger.info(f"✓ Merged dataset saved: {output_path}")
            logger.info(f"  Dimensions: {dict(merged.dims)}")
            logger.info(f"  Variables: {list(merged.data_vars)}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to merge NetCDF files: {e}")
            raise
    
    def analyze_netcdf_structure(self, filepath: Path) -> Dict:
        """
        Analyze structure of NetCDF file
        
        Parameters
        ----------
        filepath : Path
            Path to NetCDF file
            
        Returns
        -------
        dict
            Structure information
        """
        try:
            with xr.open_dataset(filepath, engine='netcdf4') as ds:
                info = {
                    'filename': filepath.name,
                    'dimensions': dict(ds.dims),
                    'variables': {},
                    'global_attributes': dict(ds.attrs),
                    'coordinate_variables': list(ds.coords)
                }
                
                # Variable details
                for var_name in ds.data_vars:
                    var = ds[var_name]
                    info['variables'][var_name] = {
                        'dtype': str(var.dtype),
                        'dimensions': list(var.dims),
                        'shape': var.shape,
                        'attributes': dict(var.attrs),
                        'has_nan': bool(np.isnan(var.values).any()),
                        'nan_percentage': float(np.isnan(var.values).sum() / var.values.size * 100)
                    }
                
                return info
                
        except Exception as e:
            logger.error(f"Failed to analyze {filepath}: {e}")
            return {'error': str(e), 'filename': filepath.name}
    
    def compare_netcdf_files(self, filepaths: List[Path]) -> Dict:
        """
        Compare multiple NetCDF files
        
        Parameters
        ----------
        filepaths : list
            List of NetCDF file paths
            
        Returns
        -------
        dict
            Comparison results
        """
        comparison = {
            'common_variables': set(),
            'unique_variables': {},
            'dimension_analysis': {},
            'temporal_coverage': {},
            'spatial_coverage': {}
        }
        
        all_structures = []
        
        for filepath in filepaths:
            structure = self.analyze_netcdf_structure(filepath)
            all_structures.append(structure)
            
            if 'variables' in structure:
                comparison['unique_variables'][filepath.name] = list(structure['variables'].keys())
                
                if not comparison['common_variables']:
                    comparison['common_variables'] = set(structure['variables'].keys())
                else:
                    comparison['common_variables'] = comparison['common_variables'].intersection(
                        set(structure['variables'].keys())
                    )
        
        # Analyze each file
        for i, structure in enumerate(all_structures):
            filename = filepaths[i].name
            
            if 'dimensions' in structure:
                comparison['dimension_analysis'][filename] = structure['dimensions']
            
            # Try to extract temporal info
            if 'time' in structure.get('dimensions', {}):
                try:
                    with xr.open_dataset(filepaths[i], engine='netcdf4') as ds:
                        if 'time' in ds:
                            time_var = ds['time']
                            if hasattr(time_var, 'values') and len(time_var) > 0:
                                times = time_var.values
                                if hasattr(times, '__len__'):
                                    comparison['temporal_coverage'][filename] = {
                                        'start': str(times[0]) if len(times) > 0 else None,
                                        'end': str(times[-1]) if len(times) > 0 else None,
                                        'n_timesteps': len(times)
                                    }
                except:
                    pass
        
        comparison['common_variables'] = list(comparison['common_variables'])
        
        logger.info(f"Comparison results:")
        logger.info(f"  Common variables: {len(comparison['common_variables'])}")
        logger.info(f"  Unique variables per file: {comparison['unique_variables']}")
        
        return comparison

# Main downloader class
class MarineDataDownloader:
    """Main downloader for marine productivity data"""
    
    def __init__(self):
        self.netcdf_downloader = NetCDFDriveDownloader()
        logger.info("Initialized MarineDataDownloader")
    
    def download_all_data(self,
                         urls: Optional[List[str]] = None,
                         output_dir: Union[str, Path] = DATA_DIR / 'raw') -> Dict:
        """
        Download all marine productivity data
        
        Parameters
        ----------
        urls : list, optional
            List of Google Drive URLs
        output_dir : str or Path
            Output directory
            
        Returns
        -------
        dict
            Download results
        """
        if urls is None:
            # Use your specific NetCDF file URLs
            urls = [
                "https://drive.google.com/file/d/17YEvHDE9DmtLsXKDGYwrGD8OE46swNDc/view?usp=sharing",
                "https://drive.google.com/file/d/16DyROUrgvfRQRvrBS3W3Y4o44vd-ZB67/view?usp=sharing",
                "https://drive.google.com/file/d/1c3f92nsOCY5hJv3zy0SAPXf8WsAVYNVI/view?usp=sharing",
                "https://drive.google.com/file/d/1JapTCN9CLn_hy9CY4u3Gcanv283LBdXy/view?usp=sharing"
            ]
        
        # Suggested filenames based on common marine data
        suggested_filenames = [
            "chlorophyll_data.nc",      # chl data
            "light_attenuation.nc",     # kd490 data
            "water_clarity.nc",         # zsd data
            "primary_productivity.nc"   # pp data
        ]
        
        logger.info("=" * 60)
        logger.info("DOWNLOADING MARINE PRODUCTIVITY DATA")
        logger.info("=" * 60)
        logger.info(f"Found {len(urls)} NetCDF files to download")
        
        # Download files
        downloaded_files = self.netcdf_downloader.download_project_netcdf_files(
            urls=urls,
            output_dir=output_dir,
            filenames=suggested_filenames
        )
        
        # Analyze downloaded files
        analysis_results = {}
        comparison_results = {}
        
        if downloaded_files:
            # Analyze each file
            for filename, filepath in downloaded_files.items():
                logger.info(f"\nAnalyzing {filename}...")
                analysis = self.netcdf_downloader.analyze_netcdf_structure(filepath)
                analysis_results[filename] = analysis
                
                # Log key information
                if 'variables' in analysis:
                    logger.info(f"  Variables: {list(analysis['variables'].keys())}")
                if 'dimensions' in analysis:
                    logger.info(f"  Dimensions: {analysis['dimensions']}")
            
            # Compare files
            logger.info("\n" + "-" * 40)
            logger.info("COMPARING DATASETS")
            logger.info("-" * 40)
            
            comparison_results = self.netcdf_downloader.compare_netcdf_files(
                list(downloaded_files.values())
            )
            
            logger.info(f"Common variables: {comparison_results.get('common_variables', [])}")
        
        results = {
            'downloaded_files': {k: str(v) for k, v in downloaded_files.items()},
            'file_analysis': analysis_results,
            'comparison': comparison_results,
            'output_dir': str(output_dir)
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 60)
        
        return results
    
    def create_merged_dataset(self,
                            input_dir: Union[str, Path],
                            output_file: Union[str, Path] = DATA_DIR / 'processed' / 'merged_data.nc',
                            merge_strategy: str = 'concat') -> Path:
        """
        Create merged dataset from multiple NetCDF files
        
        Parameters
        ----------
        input_dir : str or Path
            Input directory with NetCDF files
        output_file : str or Path
            Output merged file
        merge_strategy : str
            Merge strategy: 'concat', 'merge', 'combine'
            
        Returns
        -------
        Path
            Path to merged file
        """
        input_dir = Path(input_dir)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Find all NetCDF files
        nc_files = list(input_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found in {input_dir}")
        
        logger.info(f"Found {len(nc_files)} NetCDF files to merge")
        
        if merge_strategy == 'concat':
            # Simple concatenation along time dimension
            merged_path = self.netcdf_downloader.merge_netcdf_files(
                nc_files, output_file, merge_dim='time'
            )
        else:
            # More sophisticated merge logic
            merged_path = self._advanced_merge(nc_files, output_file, merge_strategy)
        
        return merged_path
    
    def _advanced_merge(self, 
                       filepaths: List[Path], 
                       output_path: Path,
                       strategy: str) -> Path:
        """Advanced merge strategies"""
        # Implement based on your specific data structure
        # This is a placeholder - customize based on your data
        return self.netcdf_downloader.merge_netcdf_files(filepaths, output_path, 'time')
