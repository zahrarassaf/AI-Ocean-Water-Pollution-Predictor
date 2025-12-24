#!/usr/bin/env python3
"""
Main pipeline for marine pollution prediction system.
Professional grade with comprehensive error handling and monitoring.
"""

import sys
import argparse
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import init_experiment_logger, get_experiment_logger, Timer
from src.utils.config import ConfigManager, ExperimentConfig
from src.data.downloader import MarineDataManager
from src.data.loader import MarineDataLoader
from src.data.preprocessor import DataPreprocessor
# خط 23 - با try-except محافظت می‌کنیم
try:
    from src.models.trainer import ModelTrainer
except ImportError as e:
    print(f"Import error for ModelTrainer: {e}")
    print("Note: You may need to download data first before training.")
    # ModelTrainer را به None تنظیم می‌کنیم تا بعداً کنترل شود
    ModelTrainer = None
from src.evaluation.analyzer import ModelAnalyzer


class PipelineController:
    """Control and monitor pipeline execution."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = get_experiment_logger()
        self.is_running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False
    
    def check_health(self) -> bool:
        """Check system health and dependencies."""
        self.logger.info("Performing system health check...")
        
        checks = []
        
        # Check Python version
        import platform
        python_version = platform.python_version()
        checks.append(('Python version', python_version, '3.8+', python_version >= '3.8'))
        
        # Check available memory
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        checks.append(('Available memory', f"{memory_gb:.1f} GB", '>= 8 GB', memory_gb >= 8))
        
        # Check disk space
        disk = psutil.disk_usage(str(project_root))
        disk_gb = disk.free / (1024 ** 3)
        checks.append(('Free disk space', f"{disk_gb:.1f} GB", '>= 10 GB', disk_gb >= 10))
        
        # Log all checks
        all_passed = True
        for name, actual, required, passed in checks:
            status = "PASS" if passed else "FAIL"
            self.logger.info(f"  {name}: {actual} (required: {required}) [{status}]")
            if not passed:
                all_passed = False
        
        if not all_passed:
            self.logger.warning("Some health checks failed. Pipeline may not run optimally.")
        
        return all_passed


class MarinePollutionPipeline:
    """Main pipeline for marine pollution prediction."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.controller = PipelineController(config)
        self.logger = get_experiment_logger()
        
        # Initialize components
        self.data_manager = None
        self.data_loader = None
        self.preprocessor = None
        self.trainer = None
        self.analyzer = None
        
        # Pipeline state
        self.state = {
            'data_downloaded': False,
            'data_loaded': False,
            'data_processed': False,
            'model_trained': False,
            'model_evaluated': False
        }
    
    def run(self, steps: Optional[list] = None):
        """Run the complete pipeline or specific steps."""
        try:
            self.logger.info("=" * 80)
            self.logger.info("MARINE POLLUTION PREDICTION PIPELINE")
            self.logger.info("=" * 80)
            self.logger.info(f"Experiment ID: {self.config.experiment_id}")
            self.logger.info(f"Start time: {datetime.now().isoformat()}")
            
            # Log configuration
            from src.utils.config import ConfigManager
            config_manager = ConfigManager()
            config_dict = config_manager._config_to_dict(self.config)
            self.logger.log_config(config_dict)
            
            # Perform health check
            self.controller.check_health()
            
            # Determine steps to run
            if steps is None:
                steps = ['download', 'process', 'train', 'evaluate']
            
            self.logger.info(f"Pipeline steps to execute: {', '.join(steps)}")
            
            # Execute pipeline steps
            pipeline_steps = {
                'download': self._download_data,
                'process': self._process_data,
                'train': self._train_model,
                'evaluate': self._evaluate_model
            }
            
            for step_name in steps:
                if step_name in pipeline_steps:
                    if not self.controller.is_running:
                        self.logger.warning("Pipeline interrupted by user")
                        break
                    
                    with Timer(f"pipeline_step_{step_name}", self.logger.main_logger):
                        pipeline_steps[step_name]()
                else:
                    self.logger.warning(f"Unknown pipeline step: {step_name}")
            
            # Generate final report
            self._generate_report()
            
            self.logger.log_experiment_end("completed")
            
        except Exception as e:
            self.logger.main_logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.logger.log_experiment_end("failed")
            raise
    
    def _download_data(self):
        """Download marine data from Google Drive."""
        self.logger.main_logger.info("Step 1: Downloading marine data")
        
        try:
            self.data_manager = MarineDataManager(data_dir=self.config.data.raw_dir.parent)
            
            download_report = self.data_manager.download_all_datasets(
                force_redownload=False  # Don't re-download existing files
            )
            
            # Log download results
            self.logger.data_logger.log_data_info(download_report, "download")
            
            # Create data inventory
            inventory = self.data_manager.create_data_inventory()
            self.logger.data_logger.log_data_info(inventory, "inventory")
            
            self.state['data_downloaded'] = True
            self.logger.main_logger.info("Data download completed successfully")
            
        except Exception as e:
            self.logger.main_logger.error(f"Data download failed: {e}")
            raise
    
    def _process_data(self):
        """Process and prepare data for modeling."""
        self.logger.main_logger.info("Step 2: Processing marine data")
        
        try:
            # Initialize data loader
            from src.data.loader import MarineDataLoader
            self.data_loader = MarineDataLoader(
                data_dir=self.config.data.raw_dir,
                config=self.config.data
            )
            
            # Load all NetCDF files
            dataset = self.data_loader.load_all_datasets()
            
            # Initialize preprocessor
            from src.data.preprocessor import DataPreprocessor
            self.preprocessor = DataPreprocessor(config=self.config.data)
            
            # Process data
            processed_data = self.preprocessor.process(dataset)
            
            # Prepare features and target
            X, y, feature_names = self.preprocessor.prepare_features(
                processed_data,
                target_var=self.config.data.target_variable,
                feature_vars=self.config.data.feature_variables
            )
            
            # Create train/val/test splits
            from src.data.splitter import DataSplitter
            splitter = DataSplitter(
                test_size=self.config.training.test_size,
                val_size=self.config.training.validation_size,
                random_state=self.config.training.random_state
            )
            
            splits = splitter.split(X, y, temporal=True)
            
            # Save processed data
            output_dir = self.config.data.processed_dir / self.config.experiment_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import joblib
            joblib.dump({
                'X_train': splits['X_train'],
                'X_val': splits['X_val'],
                'X_test': splits['X_test'],
                'y_train': splits['y_train'],
                'y_val': splits['y_val'],
                'y_test': splits['y_test'],
                'feature_names': feature_names
            }, output_dir / "processed_data.joblib")
            
            self.state['data_processed'] = True
            self.logger.main_logger.info(f"Data processing completed. Features: {len(feature_names)}")
            
            # Store processed data for next steps
            self.processed_data = {
                'splits': splits,
                'feature_names': feature_names,
                'output_dir': output_dir
            }
            
        except Exception as e:
            self.logger.main_logger.error(f"Data processing failed: {e}")
            raise
    
    def _train_model(self):
        """Train machine learning model."""
        self.logger.main_logger.info("Step 3: Training prediction model")
        
        if not self.state['data_processed']:
            self.logger.main_logger.error("Data not processed. Run process step first.")
            return
        
        try:
            # Check if ModelTrainer is available
            if ModelTrainer is None:
                self.logger.main_logger.error(
                    "ModelTrainer module not available. "
                    "This may be because data hasn't been downloaded yet. "
                    "Please run the download step first, or check if src/models/trainer.py exists."
                )
                return
            
            # Initialize trainer
            self.trainer = ModelTrainer(
                config=self.config,
                output_dir=self.config.output_dir / self.config.experiment_id
            )
            
            # Get processed data
            splits = self.processed_data['splits']
            feature_names = self.processed_data['feature_names']
            
            # Train model
            training_results = self.trainer.train(
                X_train=splits['X_train'],
                y_train=splits['y_train'],
                X_val=splits['X_val'],
                y_val=splits['y_val'],
                feature_names=feature_names
            )
            
            # Log training results
            self.logger.metrics_logger.log_metrics(training_results['metrics'], 'training')
            
            self.state['model_trained'] = True
            self.logger.main_logger.info("Model training completed successfully")
            
            # Store trainer for evaluation
            self.training_results = training_results
            
        except Exception as e:
            self.logger.main_logger.error(f"Model training failed: {e}")
            raise
    
    def _evaluate_model(self):
        """Evaluate trained model."""
        self.logger.main_logger.info("Step 4: Evaluating model performance")
        
        if not self.state['model_trained']:
            self.logger.main_logger.error("Model not trained. Run train step first.")
            return
        
        try:
            # Get test data
            splits = self.processed_data['splits']
            
            # Initialize analyzer
            from src.evaluation.analyzer import ModelAnalyzer
            self.analyzer = ModelAnalyzer(
                model=self.trainer.model,
                config=self.config
            )
            
            # Make predictions
            predictions, uncertainty = self.trainer.predict(
                splits['X_test'],
                return_uncertainty=True
            )
            
            # Evaluate
            evaluation_results = self.analyzer.evaluate(
                y_true=splits['y_test'],
                y_pred=predictions,
                uncertainty=uncertainty
            )
            
            # Generate comprehensive analysis
            analysis_report = self.analyzer.analyze(
                X_test=splits['X_test'],
                y_test=splits['y_test'],
                predictions=predictions,
                feature_names=self.processed_data['feature_names']
            )
            
            # Log evaluation results
            self.logger.metrics_logger.log_metrics(evaluation_results, 'evaluation')
            
            # Save evaluation report
            report_dir = self.config.output_dir / self.config.experiment_id / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(report_dir / "evaluation_report.json", 'w') as f:
                json.dump(analysis_report, f, indent=2, default=str)
            
            self.state['model_evaluated'] = True
            self.logger.main_logger.info("Model evaluation completed")
            
        except Exception as e:
            self.logger.main_logger.error(f"Model evaluation failed: {e}")
            raise
    
    def _generate_report(self):
        """Generate final pipeline report."""
        self.logger.main_logger.info("Generating final pipeline report...")
        
        report = {
            'experiment_id': self.config.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'pipeline_state': self.state,
            'config_summary': {
                'data_config': {
                    'target_variable': self.config.data.target_variable,
                    'feature_count': len(self.config.data.feature_variables),
                    'data_source': self.config.data.source.value
                },
                'model_config': {
                    'model_type': self.config.model.model_type,
                    'uncertainty_method': self.config.model.uncertainty_method
                },
                'training_config': {
                    'test_size': self.config.training.test_size,
                    'validation_size': self.config.training.validation_size
                }
            }
        }
        
        # Add results if available
        if hasattr(self, 'training_results'):
            report['training_results'] = self.training_results.get('metrics', {})
        
        if hasattr(self, 'analyzer'):
            report['evaluation_results'] = getattr(self.analyzer, 'last_evaluation', {})
        
        # Save report
        report_dir = self.config.output_dir / self.config.experiment_id
        report_file = report_dir / "pipeline_report.json"
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.main_logger.info(f"Pipeline report saved to: {report_file}")


def main():
    """Main entry point for pipeline execution."""
    
    parser = argparse.ArgumentParser(
        description='Marine Pollution Prediction Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=Path, default=None,
                       help='Path to configuration file')
    parser.add_argument('--experiment-name', type=str, default='marine_pollution',
                       help='Name of the experiment')
    parser.add_argument('--output-dir', type=Path, default='results',
                       help='Output directory for results')
    parser.add_argument('--data-dir', type=Path, default='data',
                       help='Data directory')
    parser.add_argument('--steps', nargs='+',
                       choices=['download', 'process', 'train', 'evaluate'],
                       default=['download', 'process', 'train', 'evaluate'],
                       help='Pipeline steps to execute')
    parser.add_argument('--log-dir', type=Path, default='logs',
                       help='Log directory')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        
        if args.config:
            config = config_manager.load_config(args.config)
        else:
            # Create config from arguments
            config = ExperimentConfig()
            config.experiment_name = args.experiment_name
            config.output_dir = args.output_dir
            config.data.raw_dir = args.data_dir / "raw"
            config.data.processed_dir = args.data_dir / "processed"
        
        # Update config with command line arguments
        if args.debug:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize logging
        init_experiment_logger(
            experiment_id=config.experiment_id,
            log_dir=args.log_dir
        )
        
        # Create and run pipeline
        pipeline = MarinePollutionPipeline(config)
        pipeline.run(steps=args.steps)
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Experiment ID: {config.experiment_id}")
        print(f"Results directory: {config.output_dir / config.experiment_id}")
        print("="*80)
        
    except Exception as e:
        print(f"\nERROR: Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
