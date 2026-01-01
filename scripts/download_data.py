#!/usr/bin/env python3
"""
Download NetCDF data from Google Drive for marine productivity project
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.google_drive_loader import MarineDataDownloader
from src.utils.logger import setup_logger

def main():
    """Download NetCDF data from Google Drive"""
    
    parser = argparse.ArgumentParser(
        description='Download NetCDF data from Google Drive for marine productivity'
    )
    
    parser.add_argument(
        '--urls-file',
        type=str,
        help='Text file containing Google Drive URLs (one per line)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for downloaded files'
    )
    
    parser.add_argument(
        '--merge',
        action='store_true',
        help='Merge downloaded files into single dataset'
    )
    
    parser.add_argument(
        '--merge-output',
        type=str,
        default='data/processed/merged_data.nc',
        help='Output path for merged dataset'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip download if files already exist'
    )
    
    args = parser.parse_args()
    
    # Setup logger
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logger = setup_logger(
        'download',
        config={
            'level': 'INFO',
            'log_file': str(log_dir / f'download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            'json_format': False
        }
    )
    
    try:
        logger.info("=" * 70)
        logger.info("MARINE PRODUCTIVITY NETCDF DATA DOWNLOAD")
        logger.info("=" * 70)
        
        # Initialize downloader
        downloader = MarineDataDownloader()
        
        # Get URLs
        urls = []
        
        if args.urls_file:
            # Load from file
            urls_file = Path(args.urls_file)
            if urls_file.exists():
                with open(urls_file, 'r') as f:
                    urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                logger.info(f"Loaded {len(urls)} URLs from {urls_file}")
            else:
                raise FileNotFoundError(f"URLs file not found: {urls_file}")
        else:
            # Use hardcoded URLs (your files)
            urls = [
                "https://drive.google.com/file/d/17YEvHDE9DmtLsXKDGYwrGD8OE46swNDc/view?usp=sharing",
                "https://drive.google.com/file/d/16DyROUrgvfRQRvrBS3W3Y4o44vd-ZB67/view?usp=sharing",
                "https://drive.google.com/file/d/1c3f92nsOCY5hJv3zy0SAPXf8WsAVYNVI/view?usp=sharing",
                "https://drive.google.com/file/d/1JapTCN9CLn_hy9CY4u3Gcanv283LBdXy/view?usp=sharing"
            ]
            logger.info("Using predefined marine data URLs")
        
        if not urls:
            raise ValueError("No URLs provided!")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Number of files to download: {len(urls)}")
        
        # Download data
        results = downloader.download_all_data(
            urls=urls,
            output_dir=output_dir
        )
        
        # Save results to JSON
        results_file = output_dir / 'download_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Download results saved to: {results_file}")
        
        # Merge files if requested
        if args.merge and results.get('downloaded_files'):
            logger.info("\n" + "-" * 40)
            logger.info("MERGING DATASETS")
            logger.info("-" * 40)
            
            try:
                merged_path = downloader.create_merged_dataset(
                    input_dir=output_dir,
                    output_file=args.merge_output,
                    merge_strategy='concat'
                )
                
                logger.info(f"✓ Merged dataset created: {merged_path}")
                
                # Analyze merged dataset
                import xarray as xr
                with xr.open_dataset(merged_path) as ds:
                    logger.info(f"Merged dataset info:")
                    logger.info(f"  Dimensions: {dict(ds.dims)}")
                    logger.info(f"  Variables: {list(ds.data_vars)}")
                    if 'time' in ds.dims:
                        logger.info(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
                
            except Exception as e:
                logger.error(f"Failed to merge datasets: {e}")
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 70)
        
        downloaded = results.get('downloaded_files', {})
        logger.info(f"Successfully downloaded: {len(downloaded)} files")
        
        for filename, filepath in downloaded.items():
            filepath = Path(filepath)
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                logger.info(f"  • {filename}: {size_mb:.1f} MB")
        
        # Check common variables
        comparison = results.get('comparison', {})
        common_vars = comparison.get('common_variables', [])
        if common_vars:
            logger.info(f"\nCommon variables across files: {len(common_vars)}")
            for var in common_vars:
                logger.info(f"  • {var}")
        
        logger.info("\n✓ All operations completed successfully!")
        
    except Exception as e:
        logger.error(f"✗ Operation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
