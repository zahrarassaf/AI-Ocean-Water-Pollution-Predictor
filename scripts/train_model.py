#!/usr/bin/env python3
"""
Train pollution prediction model from NetCDF data.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processor import OceanDataProcessor
from src.model_trainer import PollutionModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ocean pollution prediction model"
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw/',
        help='Directory containing NetCDF files'
    )
    
    parser.add_argument(
        '--target-col',
        type=str,
        default='CHL',
        help='Target column for prediction'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size ratio'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Validation set size ratio'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/',
        help='Directory to save trained model'
    )
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("Starting model training pipeline...")
    
    try:
        # Step 1: Process data
        logger.info(f"Loading data from {args.data_dir}")
        processor = OceanDataProcessor(args.data_dir)
        
        datasets = processor.load_all_netcdf()
        logger.info(f"Loaded {len(datasets)} datasets")
        
        features_df = processor.extract_features()
        logger.info(f"Extracted features: {features_df.shape}")
        
        cleaned_df = processor.clean_data(features_df)
        logger.info(f"Cleaned data: {cleaned_df.shape}")
        
        # Step 2: Create target
        target = processor.create_target(cleaned_df, args.target_col)
        
        # Step 3: Split data
        X = cleaned_df.drop(columns=[args.target_col], errors='ignore')
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
            X, target, test_size=args.test_size, val_size=args.val_size
        )
        
        # Step 4: Train models
        logger.info("Training ML models...")
        trainer = PollutionModelTrainer()
        
        # Prepare data
        X_train_scaled, X_val_scaled, X_test_scaled = trainer.prepare_features(
            X_train, X_val, X_test
        )
        
        y_train_encoded, y_val_encoded, y_test_encoded = trainer.prepare_target(
            y_train, y_val, y_test
        )
        
        # Train all models
        results = trainer.train_models(
            X_train_scaled, y_train_encoded,
            X_val_scaled, y_val_encoded
        )
        
        # Evaluate on test set
        logger.info("Evaluating best model on test set...")
        metrics = trainer.evaluate_best_model(X_test_scaled, y_test_encoded)
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_results = trainer.cross_validate(
            X_train_scaled, y_train_encoded, cv=5
        )
        
        # Hyperparameter tuning
        logger.info("Performing hyperparameter tuning...")
        best_params = trainer.hyperparameter_tuning(
            X_train_scaled, y_train_encoded
        )
        
        # Save model
        logger.info(f"Saving model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        
        # Print final results
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Best model: {trainer.best_model_name}")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test F1 Score: {metrics['f1']:.4f}")
        
        if best_params:
            logger.info(f"Best hyperparameters: {best_params}")
        
        logger.info("Model saved successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
