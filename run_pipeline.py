#!/usr/bin/env python3
"""
Main pipeline runner for marine productivity prediction
"""

import sys
import argparse
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import modules with corrected imports
from src.data.cmems_processor import CMEMSDataProcessor
from src.data.loader import MarineDataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.ensemble import RandomForestEnsemble
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.utils.logger import setup_logger

def setup_pipeline_logger():
    """Setup pipeline logger"""
    logger = setup_logger('pipeline')
    return logger

def download_data_step(args, logger):
    """Step 1: Download data"""
    logger.info("=" * 60)
    logger.info("STEP 1: DOWNLOAD DATA")
    logger.info("=" * 60)
    
    try:
        from src.data.google_drive_loader import MarineDataDownloader
        
        downloader = MarineDataDownloader()
        
        # Your Google Drive links
        urls = [
            "https://drive.google.com/file/d/17YEvHDE9DmtLsXKDGYwrGD8OE46swNDc/view?usp=sharing",
            "https://drive.google.com/file/d/16DyROUrgvfRQRvrBS3W3Y4o44vd-ZB67/view?usp=sharing",
            "https://drive.google.com/file/d/1c3f92nsOCY5hJv3zy0SAPXf8WsAVYNVI/view?usp=sharing",
            "https://drive.google.com/file/d/1JapTCN9CLn_hy9CY4u3Gcanv283LBdXy/view?usp=sharing"
        ]
        
        results = downloader.download_all_data(
            urls=urls,
            output_dir=args.data_dir
        )
        
        logger.info(f"Download completed: {len(results.get('downloaded_files', {}))} files")
        
        return results
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if not args.continue_on_error:
            raise
        return {}

def process_data_step(args, logger):
    """Step 2: Process data"""
    logger.info("=" * 60)
    logger.info("STEP 2: PROCESS DATA")
    logger.info("=" * 60)
    
    try:
        # Initialize processor
        processor = CMEMSDataProcessor(args.data_dir)
        
        # Find NetCDF files
        nc_files = list(Path(args.data_dir).glob("*.nc"))
        
        if not nc_files:
            logger.error(f"No NetCDF files found in {args.data_dir}")
            return None
        
        # Load and process first file
        dataset = processor.load_dataset(nc_files[0].name)
        
        # Create features
        features_ds = processor.create_feature_dataset(dataset, target_var='pp')
        
        # Prepare ML data
        X, y, feature_names = processor.prepare_ml_data(features_ds)
        
        logger.info(f"Processed data: X shape = {X.shape}, y shape = {y.shape}")
        logger.info(f"Features: {feature_names}")
        
        # Save processed data
        output_dir = Path(args.output_dir) / "processed"
        processor.save_processed_data(features_ds, output_dir / "features.nc")
        
        processor.close()
        
        return {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'dataset': features_ds
        }
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        if not args.continue_on_error:
            raise
        return None

def train_model_step(args, data, logger):
    """Step 3: Train model"""
    logger.info("=" * 60)
    logger.info("STEP 3: TRAIN MODEL")
    logger.info("=" * 60)
    
    if data is None:
        logger.error("No data available for training")
        return None
    
    try:
        X = data['X']
        y = data['y']
        feature_names = data['feature_names']
        
        # Initialize model
        model = RandomForestEnsemble(
            n_models=10,
            random_state=args.random_seed
        )
        
        # Initialize trainer
        trainer = ModelTrainer(
            output_dir=Path(args.output_dir) / "models",
            random_state=args.random_seed
        )
        
        # Train model
        results = trainer.train_model(
            model=model,
            X_train=X,  # Using all data for training in this example
            y_train=y,
            X_val=None,  # No validation in simple pipeline
            y_val=None,
            feature_names=feature_names
        )
        
        # Save model
        model_path = Path(args.output_dir) / "models" / "random_forest"
        model.save(model_path)
        
        logger.info(f"Model trained and saved to {model_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        if not args.continue_on_error:
            raise
        return None

def evaluate_model_step(args, data, training_results, logger):
    """Step 4: Evaluate model"""
    logger.info("=" * 60)
    logger.info("STEP 4: EVALUATE MODEL")
    logger.info("=" * 60)
    
    if data is None or training_results is None:
        logger.error("No data or model available for evaluation")
        return None
    
    try:
        # For simplicity, evaluate on training data
        # In production, use separate test set
        X = data['X']
        y = data['y']
        
        # Load model
        from src.models.ensemble import RandomForestEnsemble
        model = RandomForestEnsemble()
        model_path = Path(args.output_dir) / "models" / "random_forest"
        model.load(model_path)
        
        # Make predictions
        predictions, uncertainty = model.predict(X, return_uncertainty=True)
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_all_metrics(y, predictions, uncertainty)
        
        # Save evaluation results
        eval_dir = Path(args.output_dir) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(eval_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Model evaluation completed:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        if not args.continue_on_error:
            raise
        return None

def main():
    """Main pipeline function"""
    
    parser = argparse.ArgumentParser(description='Marine Productivity Prediction Pipeline')
    
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory for input data')
    parser.add_argument('--output-dir', type=str, default='results/pipeline',
                       help='Directory for output results')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue pipeline even if a step fails')
    parser.add_argument('--steps', type=str, nargs='+',
                       choices=['download', 'process', 'train', 'evaluate', 'all'],
                       default=['all'],
                       help='Pipeline steps to run')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_pipeline_logger()
    
    logger.info("=" * 70)
    logger.info("MARINE PRODUCTIVITY PREDICTION PIPELINE")
    logger.info("=" * 70)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine steps to run
    if 'all' in args.steps:
        steps_to_run = ['download', 'process', 'train', 'evaluate']
    else:
        steps_to_run = args.steps
    
    # Run pipeline steps
    download_results = None
    processed_data = None
    training_results = None
    evaluation_results = None
    
    try:
        # Step 1: Download
        if 'download' in steps_to_run and not args.skip_download:
            download_results = download_data_step(args, logger)
        
        # Step 2: Process
        if 'process' in steps_to_run:
            processed_data = process_data_step(args, logger)
        
        # Step 3: Train
        if 'train' in steps_to_run and not args.skip_training:
            training_results = train_model_step(args, processed_data, logger)
        
        # Step 4: Evaluate
        if 'evaluate' in steps_to_run:
            evaluation_results = evaluate_model_step(args, processed_data, training_results, logger)
        
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
        # Summary
        logger.info("Pipeline Summary:")
        if download_results:
            logger.info(f"  Downloaded files: {len(download_results.get('downloaded_files', {}))}")
        if processed_data:
            logger.info(f"  Processed samples: {processed_data['X'].shape[0]}")
        if training_results:
            logger.info(f"  Model trained: {training_results.get('model_name', 'Unknown')}")
        if evaluation_results:
            logger.info(f"  Evaluation metrics calculated: {len(evaluation_results)}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
