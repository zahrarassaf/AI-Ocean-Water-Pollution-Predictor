"""
Professional Google Drive downloader for NetCDF files.
Supports resume, validation, and comprehensive error handling.
"""

import os
import re
import sys
import time
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import gdown
import requests
import numpy as np
import xarray as xr
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class DriveDownloadError(Exception):
    """Custom exception for download errors."""
    pass


class NetCDFValidator:
    """Validate NetCDF file structure and integrity."""
    
    @staticmethod
    def validate_file(filepath: Path) -> Dict:
        """Validate NetCDF file and return metadata."""
        try:
            with xr.open_dataset(filepath, engine='netcdf4') as ds:
                metadata = {
                    'valid': True,
                    'variables': list(ds.data_vars),
                    'dimensions': dict(ds.dims),
                    'attributes': dict(ds.attrs),
                    'coords': list(ds.coords),
                    'file_size': filepath.stat().st_size,
                    'checksum': NetCDFValidator.calculate_checksum(filepath)
                }
                
                # Check for required oceanographic variables
                required_vars = ['lat', 'lon', 'time']
                metadata['has_required_vars'] = all(v in metadata['coords'] 
                                                   for v in required_vars)
                
                # Calculate variable statistics
                variable_stats = {}
                for var in ds.data_vars:
                    data = ds[var].values
                    variable_stats[var] = {
                        'dtype': str(ds[var].dtype),
                        'shape': ds[var].shape,
                        'nan_percentage': float(np.isnan(data).sum() / data.size * 100),
                        'min': float(np.nanmin(data)),
                        'max': float(np.nanmax(data)),
                        'mean': float(np.nanmean(data)),
                        'std': float(np.nanstd(data))
                    }
                metadata['variable_stats'] = variable_stats
                
                return metadata
                
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'file_size': filepath.stat().st_size if filepath.exists() else 0
            }
    
    @staticmethod
    def calculate_checksum(filepath: Path, chunk_size: int = 8192) -> str:
        """Calculate MD5 checksum for file validation."""
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                md5.update(chunk)
        return md5.hexdigest()


class RetryableSession:
    """HTTP session with retry logic for resilient downloads."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 0.3):
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set reasonable timeout
        self.session.request = lambda method, url, **kwargs: requests.Session.request(
            self.session, method, url, timeout=(30, 300), **kwargs
        )
    
    def get(self, url: str, **kwargs):
        return self.session.get(url, **kwargs)


class GoogleDriveDownloader:
    """
    Professional Google Drive downloader with comprehensive features:
    - Resume broken downloads
    - Parallel downloads
    - Checksum validation
    - Progress tracking
    - Error recovery
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "data/raw",
        max_workers: int = 4,
        chunk_size: int = 1024 * 1024,  # 1MB
        timeout: int = 300
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.timeout = timeout
        
        self.session = RetryableSession()
        self.validator = NetCDFValidator()
        
        # Cache for downloaded files
        self.download_cache = self.output_dir / ".download_cache.json"
        
        logger.info(f"Initialized GoogleDriveDownloader with output_dir: {self.output_dir}")
    
    @staticmethod
    def extract_file_id(url: str) -> Optional[str]:
        """Extract file ID from Google Drive URL."""
        patterns = [
            r'/file/d/([a-zA-Z0-9_-]+)',
            r'id=([a-zA-Z0-9_-]+)',
            r'/d/([a-zA-Z0-9_-]+)/',
            r'/d/([a-zA-Z0-9_-]+)$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise DriveDownloadError(f"Could not extract file ID from URL: {url}")
    
    def download_file(
        self,
        url: str,
        filename: Optional[str] = None,
        force_redownload: bool = False,
        validate: bool = True
    ) -> Dict:
        """
        Download a single file from Google Drive.
        
        Returns:
            Dict containing download results and file metadata.
        """
        start_time = datetime.now()
        
        try:
            file_id = self.extract_file_id(url)
            
            if not filename:
                filename = f"dataset_{file_id}.nc"
            
            output_path = self.output_dir / filename
            
            # Check if file already exists and is valid
            if not force_redownload and output_path.exists():
                validation = self.validator.validate_file(output_path)
                if validation['valid']:
                    logger.info(f"File already exists and valid: {filename}")
                    return {
                        'status': 'exists',
                        'filename': filename,
                        'path': str(output_path),
                        'validation': validation,
                        'download_time': 0,
                        'size_mb': output_path.stat().st_size / (1024 * 1024)
                    }
            
            # Download using gdown with resume capability
            download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
            
            logger.info(f"Downloading: {filename} from {url}")
            
            # Create temporary file for download
            temp_path = output_path.with_suffix('.download')
            
            try:
                gdown.download(
                    download_url,
                    str(temp_path),
                    quiet=False,
                    resume=True,
                    fuzzy=True
                )
                
                # Move temp file to final location
                temp_path.rename(output_path)
                
                download_time = (datetime.now() - start_time).total_seconds()
                
                # Validate downloaded file
                validation = {}
                if validate:
                    validation = self.validator.validate_file(output_path)
                    if not validation['valid']:
                        raise DriveDownloadError(f"Downloaded file is invalid: {validation.get('error', 'Unknown error')}")
                
                result = {
                    'status': 'success',
                    'filename': filename,
                    'path': str(output_path),
                    'validation': validation,
                    'download_time': download_time,
                    'size_mb': output_path.stat().st_size / (1024 * 1024),
                    'checksum': NetCDFValidator.calculate_checksum(output_path)
                }
                
                logger.info(f"Successfully downloaded {filename} ({result['size_mb']:.2f} MB) in {download_time:.1f}s")
                return result
                
            except Exception as e:
                # Clean up temp file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise
                
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return {
                'status': 'failed',
                'filename': filename or 'unknown',
                'error': str(e),
                'download_time': (datetime.now() - start_time).total_seconds()
            }
    
    def download_batch(
        self,
        urls: List[str],
        filenames: Optional[List[str]] = None,
        force_redownload: bool = False,
        max_parallel: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Download multiple files in parallel.
        
        Returns:
            Dict mapping filenames to download results.
        """
        if filenames and len(urls) != len(filenames):
            raise ValueError("URLs and filenames must have the same length")
        
        if not filenames:
            filenames = [f"dataset_{self.extract_file_id(url)}.nc" for url in urls]
        
        results = {}
        max_workers = min(self.max_workers, max_parallel or self.max_workers)
        
        logger.info(f"Starting batch download of {len(urls)} files with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create futures
            future_to_url = {}
            for url, filename in zip(urls, filenames):
                future = executor.submit(
                    self.download_file,
                    url=url,
                    filename=filename,
                    force_redownload=force_redownload
                )
                future_to_url[future] = (url, filename)
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_url), total=len(future_to_url), desc="Downloading"):
                url, filename = future_to_url[future]
                try:
                    result = future.result()
                    results[filename] = result
                except Exception as e:
                    logger.error(f"Error downloading {filename}: {e}")
                    results[filename] = {
                        'status': 'failed',
                        'filename': filename,
                        'error': str(e)
                    }
        
        # Generate summary statistics
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        existing = sum(1 for r in results.values() if r['status'] == 'exists')
        failed = sum(1 for r in results.values() if r['status'] == 'failed')
        
        summary = {
            'total': len(urls),
            'successful': successful,
            'existing': existing,
            'failed': failed,
            'total_size_mb': sum(r.get('size_mb', 0) for r in results.values() if 'size_mb' in r),
            'results': results
        }
        
        logger.info(f"Batch download completed: {successful} new, {existing} existing, {failed} failed")
        
        return summary
    
    def validate_downloads(self, filenames: Optional[List[str]] = None) -> Dict:
        """Validate all downloaded files."""
        if not filenames:
            filenames = [f.name for f in self.output_dir.glob("*.nc")]
        
        validation_results = {}
        
        for filename in filenames:
            filepath = self.output_dir / filename
            if filepath.exists():
                validation_results[filename] = self.validator.validate_file(filepath)
            else:
                validation_results[filename] = {'valid': False, 'error': 'File not found'}
        
        valid_count = sum(1 for v in validation_results.values() if v['valid'])
        
        return {
            'total': len(filenames),
            'valid': valid_count,
            'invalid': len(filenames) - valid_count,
            'details': validation_results
        }


class MarineDataManager:
    """
    High-level manager for marine data download and processing.
    """
    
    # Pre-defined marine dataset configurations
    DATASET_CONFIGS = {
        'chlorophyll': {
            'url': 'https://drive.google.com/file/d/17YEvHDE9DmtLsXKDGYwrGD8OE46swNDc/view',
            'filename': 'chlorophyll_concentration.nc',
            'expected_vars': ['chl', 'lat', 'lon', 'time']
        },
        'light_attenuation': {
            'url': 'https://drive.google.com/file/d/16DyROUrgvfRQRvrBS3W3Y4o44vd-ZB67/view',
            'filename': 'diffuse_attenuation.nc',
            'expected_vars': ['kd490', 'lat', 'lon', 'time']
        },
        'water_clarity': {
            'url': 'https://drive.google.com/file/d/1c3f92nsOCY5hJv3zy0SAPXf8WsAVYNVI/view',
            'filename': 'secchi_depth.nc',
            'expected_vars': ['zsd', 'lat', 'lon', 'time']
        },
        'primary_productivity': {
            'url': 'https://drive.google.com/file/d/1JapTCN9CLn_hy9CY4u3Gcanv283LBdXy/view',
            'filename': 'primary_productivity.nc',
            'expected_vars': ['pp', 'lat', 'lon', 'time']
        }
    }
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        self.downloader = GoogleDriveDownloader(output_dir=self.raw_dir)
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized MarineDataManager with data_dir: {self.data_dir}")
    
    def download_all_datasets(
        self,
        datasets: Optional[List[str]] = None,
        force_redownload: bool = False
    ) -> Dict:
        """
        Download all configured marine datasets.
        
        Args:
            datasets: List of dataset names to download (None for all)
            force_redownload: Whether to re-download existing files
            
        Returns:
            Comprehensive download summary
        """
        if datasets is None:
            datasets = list(self.DATASET_CONFIGS.keys())
        
        urls = []
        filenames = []
        
        for dataset_name in datasets:
            if dataset_name not in self.DATASET_CONFIGS:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue
            
            config = self.DATASET_CONFIGS[dataset_name]
            urls.append(config['url'])
            filenames.append(config['filename'])
        
        if not urls:
            raise ValueError("No valid datasets specified")
        
        logger.info(f"Downloading {len(urls)} marine datasets: {', '.join(datasets)}")
        
        # Perform batch download
        download_results = self.downloader.download_batch(
            urls=urls,
            filenames=filenames,
            force_redownload=force_redownload
        )
        
        # Validate downloaded files
        validation_results = self.downloader.validate_downloads(filenames)
        
        # Generate comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'datasets_requested': datasets,
            'download_summary': download_results,
            'validation_summary': validation_results,
            'missing_variables': self._check_missing_variables(validation_results)
        }
        
        # Save report
        report_path = self.data_dir / f"download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Download report saved to: {report_path}")
        
        return report
    
    def _check_missing_variables(self, validation_results: Dict) -> Dict:
        """Check for missing expected variables in downloaded files."""
        missing_vars = {}
        
        for filename, validation in validation_results.get('details', {}).items():
            if not validation.get('valid', False):
                continue
            
            # Find which dataset this file corresponds to
            dataset_name = None
            for name, config in self.DATASET_CONFIGS.items():
                if config['filename'] == filename:
                    dataset_name = name
                    break
            
            if dataset_name:
                expected_vars = self.DATASET_CONFIGS[dataset_name]['expected_vars']
                actual_vars = validation.get('variables', []) + validation.get('coords', [])
                
                missing = set(expected_vars) - set(actual_vars)
                if missing:
                    missing_vars[filename] = {
                        'dataset': dataset_name,
                        'missing_variables': list(missing),
                        'expected': expected_vars,
                        'actual': actual_vars
                    }
        
        return missing_vars
    
    def create_data_inventory(self) -> Dict:
        """Create inventory of all available data files."""
        inventory = {
            'raw_files': [],
            'processed_files': [],
            'total_size_gb': 0,
            'last_updated': datetime.now().isoformat()
        }
        
        # Scan raw directory
        for filepath in self.raw_dir.glob("*.nc"):
            stats = filepath.stat()
            validation = self.downloader.validator.validate_file(filepath)
            
            inventory['raw_files'].append({
                'filename': filepath.name,
                'size_mb': stats.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                'validation': validation
            })
            inventory['total_size_gb'] += stats.st_size / (1024 ** 3)
        
        # Scan processed directory
        for filepath in self.processed_dir.glob("*.*"):
            stats = filepath.stat()
            
            inventory['processed_files'].append({
                'filename': filepath.name,
                'size_mb': stats.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                'type': filepath.suffix[1:]  # Remove dot
            })
            inventory['total_size_gb'] += stats.st_size / (1024 ** 3)
        
        inventory['raw_count'] = len(inventory['raw_files'])
        inventory['processed_count'] = len(inventory['processed_files'])
        
        return inventory


def main():
    """Command-line interface for data download."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Download marine data from Google Drive')
    parser.add_argument('--datasets', nargs='+', 
                       choices=list(MarineDataManager.DATASET_CONFIGS.keys()),
                       default=list(MarineDataManager.DATASET_CONFIGS.keys()),
                       help='Datasets to download')
    parser.add_argument('--output-dir', default='data',
                       help='Output directory for downloaded files')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download of existing files')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Validate downloaded files')
    parser.add_argument('--parallel', type=int, default=4,
                       help='Number of parallel downloads')
    parser.add_argument('--inventory', action='store_true',
                       help='Create data inventory without downloading')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    try:
        manager = MarineDataManager(data_dir=args.output_dir)
        
        if args.inventory:
            inventory = manager.create_data_inventory()
            print(json.dumps(inventory, indent=2))
            return
        
        report = manager.download_all_datasets(
            datasets=args.datasets,
            force_redownload=args.force
        )
        
        # Print summary
        summary = report['download_summary']
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Total files: {summary['total']}")
        print(f"Successfully downloaded: {summary['successful']}")
        print(f"Already existed: {summary['existing']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total size: {summary['total_size_mb']:.2f} MB")
        
        if summary['failed'] > 0:
            print("\nFailed downloads:")
            for filename, result in summary['results'].items():
                if result['status'] == 'failed':
                    print(f"  {filename}: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
