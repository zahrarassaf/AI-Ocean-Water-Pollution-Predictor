#!/usr/bin/env python3
"""
Training script for marine productivity prediction with NetCDF data
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.google_drive_loader import MarineDataDownloader
from src.data.loader import MarineDataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.ensemble import RandomForestEnsemble, GaussianProcessEnsemble, SuperEnsemble
from src.models.neural_networks import BayesianNeuralNetwork
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.utils.logger import setup_logger

def main():
    """Main training pipeline for marine data"""
    
    parser = argparse.ArgumentParser(
        description='Train marine productivity models with NetCDF data'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing NetCDF files'
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download data from Google Drive first'
    )
    
    parser.add_argument(
        '--merge',
        action='store_true',
        help='Merge multiple NetCDF files'
    )
    
    parser.add_argument(
        '--target-var',
        type=str,
        default='pp',
        help='Target variable name (primary productivity)'
    )
    
    parser.add_argument(
        '--feature-vars',
        type=str,
        nargs='+',
        default=['chl', 'kd490', 'zsd', 'current_speed', 'temperature', 'salinity'],
        help='Feature variable names'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/marine',
        help='Output directory'
    )
    
    parser.add_argument(
        '--time-range',
        type=str,
        nargs=2,
        help='Time range filter: start_date end_date'
    )
    
    parser.add_argument(
        '--spatial-range',
        type=float,
        nargs=4,
        metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'),
        help='Spatial range filter'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='ensemble',
        choices=['rf', 'gp', 'nn', 'ensemble', 'all'],
        help='Model type to train'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = Path('logs/marine')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        'marine_training',
        config={
            'level': 'INFO',
            'log_file': str(log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            'json_format': True
        }
    )
    
    try:
        logger.info("=" * 70)
        logger.info("MARINE PRIMARY PRODUCTIVITY PREDICTION TRAINING")
        logger.info("=" * 70)
        
        # 0. Download data if requested
        data_dir = Path(args.data_dir)
        
        if args.download or not data_dir.exists() or not any(data_dir.glob("*.nc")):
            logger.info("\n0. DOWNLOADING DATA FROM GOOGLE DRIVE")
            logger.info("-" * 40)
            
            downloader = MarineDataDownloader()
            download_results = downloader.download_all_data(
                output_dir=data_dir
            )
            
            # Save download report
            report_file = data_dir / 'download_report.json'
            with open(report_file, 'w') as f:
                json.dump(download_results, f, indent=2, default=str)
            
            logger.info(f"Download report saved: {report_file}")
        
        # 1. Load NetCDF data
        logger.info("\n1. LOADING NETCDF DATA")
        logger.info("-" * 40)
        
        # Find NetCDF files
        nc_files = list(data_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found in {data_dir}")
        
        logger.info(f"Found {len(nc_files)} NetCDF files:")
        for nc_file in nc_files:
            size_mb = nc_file.stat().st_size / (1024 * 1024)
            logger.info(f"  • {nc_file.name}: {size_mb:.1f} MB")
        
        # Initialize data loader
        data_loader = MarineDataLoader(use_dask=True, max_memory_gb=8.0)
        
        # Prepare load arguments
        load_kwargs = {
            'variables': [args.target_var] + args.feature_vars,
            'preprocess': True
        }
        
        if args.time_range:
            load_kwargs['time_range'] = tuple(args.time_range)
            logger.info(f"Time filter: {args.time_range[0]} to {args.time_range[1]}")
        
        if args.spatial_range:
            lat_min, lat_max, lon_min, lon_max = args.spatial_range
            load_kwargs['spatial_range'] = {
                'lat': (lat_min, lat_max),
                'lon': (lon_min, lon_max)
            }
            logger.info(f"Spatial filter: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")
        
        # Load data
        if args.merge or len(nc_files) > 1:
            logger.info("\nLoading and merging multiple files...")
            dataset = data_loader.load_multiple_datasets(
                nc_files,
                merge_dim='time',
                **load_kwargs
            )
        else:
            logger.info(f"\nLoading single file: {nc_files[0].name}")
            dataset = data_loader.load_marine_dataset(
                nc_files[0],
                **load_kwargs
            )
        
        # Analyze loaded dataset
        logger.info("\nDataset information:")
        logger.info(f"  Dimensions: {dict(dataset.dims)}")
        logger.info(f"  Variables: {list(dataset.data_vars)}")
        
        if 'time' in dataset.dims:
            logger.info(f"  Time range: {dataset.time.values[0]} to {dataset.time.values[-1]}")
            logger.info(f"  Time steps: {len(dataset.time)}")
        
        # 2. Preprocess data
        logger.info("\n2. PREPROCESSING DATA")
        logger.info("-" * 40)
        
        preprocessor = DataPreprocessor()
        
        # Handle missing values
        dataset = preprocessor.handle_missing_values(dataset, method='interpolate')
        
        # Detect outliers
        outlier_info = preprocessor.detect_outliers(dataset)
        logger.info(f"Outlier detection completed")
        
        # Remove outliers
        dataset = preprocessor.remove_outliers(dataset, method='clip')
        
        # Scale features (but not target)
        feature_vars = [v for v in args.feature_vars if v in dataset]
        target_var = args.target_var if args.target_var in dataset else None
        
        if target_var:
            # Temporarily remove target for scaling
            target_data = dataset[target_var].copy()
            dataset = dataset.drop_vars([target_var])
        
        # Scale features
        dataset = preprocessor.scale_data(dataset, method='standard')
        
        # Add target back
        if target_var:
            dataset[target_var] = target_data
        
        # Create derived features
        dataset = preprocessor.create_derived_features(dataset)
        
        # Apply temporal smoothing
        dataset = preprocessor.apply_temporal_filters(dataset, window_size=7)
        
        # 3. Prepare features and target
        logger.info("\n3. FEATURE ENGINEERING")
        logger.info("-" * 40)
        
        feature_engineer = FeatureEngineer()
        
        # Prepare feature matrix
        all_features = feature_vars + ['euphotic_depth', 'clarity_index', 'current_energy']
        available_features = [f for f in all_features if f in dataset.data_vars]
        
        X, y, feature_names, feature_info = feature_engineer.prepare_marine_features(
            dataset,
            target_var=target_var,
            feature_vars=available_features,
            include_interactions=True,
            include_temporal=True
        )
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Features used: {feature_names}")
        
        # Split data
        splits = feature_engineer.create_splits(
            X, y,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            temporal_split=True if 'time' in dataset.dims else False
        )
        
        X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
        y_train, y_val, y_test = splits['y_train'], splits['y_val'], splits['y_test']
        
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Validation samples: {X_val.shape[0]}")
        logger.info(f"Test samples: {X_test.shape[0]}")
        
        # 4. Train models (remainder similar to previous train.py)
        # ... [Continue with model training code from previous train.py]
        
        # 5. Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset info
        dataset_info = {
            'original_files': [str(f) for f in nc_files],
            'dataset_dimensions': dict(dataset.dims),
            'dataset_variables': list(dataset.data_vars),
            'features_used': feature_names,
            'feature_info': feature_info,
            'data_splits': {
                'train_samples': X_train.shape[0],
                'val_samples': X_val.shape[0],
                'test_samples': X_test.shape[0]
            }
        }
        
        info_file = output_dir / 'dataset_info.json'
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2, default=str)
        
        logger.info(f"\nDataset information saved: {info_file}")
        
        # Close data loader
        data_loader.close()
        
        logger.info("\n" + "=" * 70)
        logger.info("DATA PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Processed data ready for model training in directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
