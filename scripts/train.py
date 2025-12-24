#!/usr/bin/env python3
"""
Main training script for marine productivity prediction
"""

import argparse
import sys
from pathlib import Path
import json
import yaml
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.ensemble import RandomForestEnsemble, GaussianProcessEnsemble, SuperEnsemble
from src.models.neural_networks import BayesianNeuralNetwork
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.utils.logger import setup_logger, ModelLogger
from src.config.settings import settings

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train marine productivity prediction models'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        default='configs/training_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to data directory or file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for models and results'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='ensemble',
        choices=['rf', 'gp', 'nn', 'ensemble', 'all'],
        help='Type of model to train'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs for neural networks'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )
    
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Test split ratio'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU for training if available'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def setup_environment(args):
    """Setup environment and directories"""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    model_dir = output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    results_dir = output_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Setup logging
    log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logger = setup_logger(
        'training',
        config={
            'level': 'DEBUG' if args.debug else 'INFO',
            'log_file': str(log_file),
            'json_format': True
        }
    )
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    
    return {
        'output_dir': output_dir,
        'model_dir': model_dir,
        'log_dir': log_dir,
        'results_dir': results_dir,
        'logger': logger
    }

def main():
    """Main training pipeline"""
    
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    env = setup_environment(args)
    logger = env['logger']
    
    try:
        logger.info("=" * 60)
        logger.info("MARINE PRODUCTIVITY PREDICTION TRAINING")
        logger.info("=" * 60)
        
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # 1. Load data
        logger.info("\n1. LOADING DATA")
        logger.info("-" * 40)
        
        data_loader = DataLoader(use_dask=True)
        dataset = data_loader.load_netcdf(args.data_path)
        
        # Validate dataset
        validation_result = data_loader.validate_dataset(dataset)
        logger.info(f"Dataset validation: {json.dumps(validation_result, indent=2)}")
        
        # 2. Preprocess data
        logger.info("\n2. PREPROCESSING DATA")
        logger.info("-" * 40)
        
        preprocessor = DataPreprocessor()
        
        # Handle missing values
        dataset = preprocessor.handle_missing_values(
            dataset, 
            method='interpolate'
        )
        
        # Detect and handle outliers
        outlier_info = preprocessor.detect_outliers(dataset)
        logger.info(f"Outlier detection: {json.dumps(outlier_info, indent=2)}")
        
        dataset = preprocessor.remove_outliers(dataset, method='clip')
        
        # Scale data
        dataset = preprocessor.scale_data(dataset, method='standard')
        
        # Create derived features
        dataset = preprocessor.create_derived_features(dataset)
        
        # Apply temporal filters
        dataset = preprocessor.apply_temporal_filters(
            dataset, 
            window_size=7,
            filter_type='savgol'
        )
        
        # 3. Feature engineering
        logger.info("\n3. FEATURE ENGINEERING")
        logger.info("-" * 40)
        
        feature_engineer = FeatureEngineer()
        
        # Prepare features and target
        X, y, feature_names, feature_info = feature_engineer.prepare_features(
            dataset,
            target_var='pp',
            include_interactions=True,
            include_polynomial=True,
            include_temporal=True
        )
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {y.shape}")
        logger.info(f"Features: {feature_names}")
        
        # Split data
        splits = feature_engineer.create_splits(
            X, y,
            test_size=args.test_split,
            val_size=args.validation_split,
            random_state=args.random_state
        )
        
        X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
        y_train, y_val, y_test = splits['y_train'], splits['y_val'], splits['y_test']
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        # 4. Train models
        logger.info("\n4. TRAINING MODELS")
        logger.info("-" * 40)
        
        models = {}
        training_results = {}
        
        # Initialize model trainer
        trainer = ModelTrainer(
            output_dir=env['model_dir'],
            use_gpu=args.use_gpu,
            random_state=args.random_state
        )
        
        # Train Random Forest Ensemble
        if args.model_type in ['rf', 'ensemble', 'all']:
            logger.info("\nTraining Random Forest Ensemble...")
            
            rf_ensemble = RandomForestEnsemble(
                n_models=10,
                random_state=args.random_state
            )
            
            rf_results = trainer.train_model(
                rf_ensemble,
                X_train, y_train,
                X_val, y_val,
                feature_names=feature_names
            )
            
            models['random_forest'] = rf_ensemble
            training_results['random_forest'] = rf_results
            
            # Save model
            model_path = env['model_dir'] / 'random_forest_ensemble'
            rf_ensemble.save(model_path)
        
        # Train Gaussian Process Ensemble
        if args.model_type in ['gp', 'ensemble', 'all']:
            logger.info("\nTraining Gaussian Process Ensemble...")
            
            gp_ensemble = GaussianProcessEnsemble(
                n_models=3,
                random_state=args.random_state
            )
            
            gp_results = trainer.train_model(
                gp_ensemble,
                X_train, y_train,
                X_val, y_val,
                feature_names=feature_names
            )
            
            models['gaussian_process'] = gp_ensemble
            training_results['gaussian_process'] = gp_results
            
            # Save model
            model_path = env['model_dir'] / 'gaussian_process_ensemble'
            gp_ensemble.save(model_path)
        
        # Train Neural Network
        if args.model_type in ['nn', 'ensemble', 'all']:
            logger.info("\nTraining Bayesian Neural Network...")
            
            nn_model = BayesianNeuralNetwork(
                input_dim=X_train.shape[1],
                hidden_dims=[128, 64, 32],
                dropout_rate=0.3,
                random_state=args.random_state
            )
            
            nn_results = trainer.train_model(
                nn_model,
                X_train, y_train,
                X_val, y_val,
                feature_names=feature_names,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            models['neural_network'] = nn_model
            training_results['neural_network'] = nn_results
            
            # Save model
            model_path = env['model_dir'] / 'bayesian_neural_network'
            nn_model.save(model_path)
        
        # Train Super Ensemble
        if args.model_type in ['ensemble', 'all'] and len(models) > 1:
            logger.info("\nTraining Super Ensemble...")
            
            # Create super ensemble from trained models
            base_models = list(models.values())
            super_ensemble = SuperEnsemble(
                base_models=base_models,
                ensemble_method='weighted_average',
                random_state=args.random_state
            )
            
            # Train super ensemble (meta-learning)
            super_results = trainer.train_model(
                super_ensemble,
                X_train, y_train,
                X_val, y_val,
                feature_names=feature_names
            )
            
            models['super_ensemble'] = super_ensemble
            training_results['super_ensemble'] = super_results
            
            # Save model
            model_path = env['model_dir'] / 'super_ensemble'
            super_ensemble.save(model_path)
        
        # 5. Evaluate models
        logger.info("\n5. EVALUATING MODELS")
        logger.info("-" * 40)
        
        evaluator = ModelEvaluator()
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            # Make predictions on test set
            predictions, uncertainty = model.predict(
                X_test, 
                return_uncertainty=True
            )
            
            # Calculate metrics
            metrics = evaluator.calculate_all_metrics(
                y_test, predictions, uncertainty
            )
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'get_feature_importance'):
                feature_importance = model.get_feature_importance()
            
            evaluation_results[model_name] = {
                'metrics': metrics,
                'predictions': predictions.tolist(),
                'uncertainty': uncertainty.tolist() if uncertainty is not None else None,
                'feature_importance': feature_importance
            }
            
            logger.info(f"Metrics for {model_name}:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
        
        # Compare all models
        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)
        
        comparison = evaluator.compare_models(evaluation_results)
        
        for metric in ['r2', 'rmse', 'mae']:
            best_model = comparison[metric]['best_model']
            best_value = comparison[metric]['best_value']
            logger.info(f"Best {metric.upper()}: {best_model} ({best_value:.4f})")
        
        # 6. Save results
        logger.info("\n6. SAVING RESULTS")
        logger.info("-" * 40)
        
        # Save training results
        training_file = env['results_dir'] / 'training_results.json'
        with open(training_file, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        # Save evaluation results
        evaluation_file = env['results_dir'] / 'evaluation_results.json'
        with open(evaluation_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Save model comparison
        comparison_file = env['results_dir'] / 'model_comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Save feature information
        feature_file = env['results_dir'] / 'feature_info.json'
        with open(feature_file, 'w') as f:
            json.dump({
                'feature_names': feature_names,
                'feature_info': feature_info,
                'preprocessing_summary': preprocessor.get_preprocessing_summary()
            }, f, indent=2)
        
        logger.info(f"\nResults saved to: {env['results_dir']}")
        logger.info(f"Models saved to: {env['model_dir']}")
        
        # 7. Generate report
        logger.info("\n7. GENERATING REPORT")
        logger.info("-" * 40)
        
        report = evaluator.generate_report(
            evaluation_results,
            feature_names,
            output_dir=env['results_dir']
        )
        
        logger.info(f"Report generated: {report}")
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        
        # Clean up
        data_loader.close()
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
