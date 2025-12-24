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

# Import available modules with graceful fallbacks
available_modules = {}

# Try to import logger
try:
    from src.utils.logger import init_experiment_logger, get_experiment_logger, Timer
    available_modules['logger'] = True
except ImportError as e:
    print(f"Warning: Logger module not available: {e}")
    available_modules['logger'] = False
    # Create minimal logger replacement
    class MinimalLogger:
        def __init__(self):
            import logging
            self.main_logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.data_logger = self
            self.metrics_logger = self
        
        def info(self, msg, *args, **kwargs):
            print(f"INFO: {msg}")
        
        def warning(self, msg, *args, **kwargs):
            print(f"WARNING: {msg}")
        
        def error(self, msg, *args, **kwargs):
            print(f"ERROR: {msg}")
        
        def debug(self, msg, *args, **kwargs):
            print(f"DEBUG: {msg}")
        
        def log_config(self, config):
            print(f"CONFIG: {config}")
        
        def log_experiment_end(self, status):
            print(f"EXPERIMENT END: {status}")
        
        def log_data_info(self, info, info_type):
            print(f"DATA {info_type.upper()}: {info}")
        
        def log_metrics(self, metrics, stage):
            print(f"METRICS {stage.upper()}: {metrics}")

# Try to import config
try:
    from src.utils.config import ConfigManager, ExperimentConfig
    available_modules['config'] = True
except ImportError as e:
    print(f"Warning: Config module not available: {e}")
    available_modules['config'] = False
    # Create minimal config replacement
    class ExperimentConfig:
        def __init__(self):
            self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.experiment_name = "marine_pollution"
            self.output_dir = Path("results")
            
            class DataConfig:
                def __init__(self):
                    self.raw_dir = Path("data/raw")
                    self.processed_dir = Path("data/processed")
                    self.target_variable = "pollution_level"
                    self.feature_variables = ["temperature", "salinity", "ph", "turbidity"]
                    self.source = type('obj', (object,), {'value': 'local'})()
            
            class ModelConfig:
                def __init__(self):
                    self.model_type = "random_forest"
                    self.uncertainty_method = "quantile"
            
            class TrainingConfig:
                def __init__(self):
                    self.test_size = 0.2
                    self.validation_size = 0.1
                    self.random_state = 42
            
            self.data = DataConfig()
            self.model = ModelConfig()
            self.training = TrainingConfig()
    
    class ConfigManager:
        def load_config(self, path):
            return ExperimentConfig()

# Try to import data modules
try:
    from src.data.downloader import MarineDataManager
    available_modules['downloader'] = True
except ImportError as e:
    print(f"Warning: Downloader module not available: {e}")
    available_modules['downloader'] = False

try:
    from src.data.loader import MarineDataLoader
    from src.data.preprocessor import DataPreprocessor
    available_modules['data_loader'] = True
except ImportError as e:
    print(f"Warning: Data loader/preprocessor not available: {e}")
    available_modules['data_loader'] = False

# Try to import splitter
try:
    from src.data.splitter import DataSplitter
    available_modules['splitter'] = True
except ImportError as e:
    print(f"Warning: DataSplitter module not available: {e}")
    available_modules['splitter'] = False

# Try to import trainer
try:
    from src.models.trainer import ModelTrainer
    available_modules['trainer'] = True
except ImportError as e:
    print(f"Warning: ModelTrainer module not available: {e}")
    available_modules['trainer'] = False
    ModelTrainer = None

# Try to import analyzer
try:
    from src.evaluation.analyzer import ModelAnalyzer
    available_modules['analyzer'] = True
except ImportError as e:
    print(f"Warning: ModelAnalyzer module not available: {e}")
    available_modules['analyzer'] = False
    ModelAnalyzer = None


class PipelineController:
    """Control and monitor pipeline execution."""
    
    def __init__(self, config, log_dir=None):
        self.config = config
        self.log_dir = log_dir
        
        # Initialize logger
        if available_modules['logger']:
            try:
                # Try to get existing logger
                self.logger = get_experiment_logger()
            except RuntimeError:
                # Initialize logger if not already initialized
                if log_dir:
                    init_experiment_logger(
                        experiment_id=config.experiment_id,
                        log_dir=log_dir
                    )
                else:
                    init_experiment_logger(experiment_id=config.experiment_id)
                self.logger = get_experiment_logger()
            
            # Check logger interface and create wrapper if needed
            if hasattr(self.logger, 'main_logger'):
                # Logger has sub-loggers
                self._logger_wrapper = self.logger
                self.logger = self.logger.main_logger
            else:
                # Create wrapper for consistent interface
                self._logger_wrapper = type('obj', (object,), {
                    'main_logger': self.logger,
                    'data_logger': self.logger,
                    'metrics_logger': self.logger
                })()
        else:
            self.logger = MinimalLogger()
            self._logger_wrapper = self.logger
        
        self.is_running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        if hasattr(self.logger, 'warning'):
            self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        else:
            print(f"WARNING: Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False
    
    def check_health(self) -> bool:
        """Check system health and dependencies."""
        self._log_message("info", "Performing system health check...")
        
        checks = []
        
        # Check Python version
        import platform
        python_version = platform.python_version()
        checks.append(('Python version', python_version, '3.8+', python_version >= '3.8'))
        
        # Check available memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            checks.append(('Available memory', f"{memory_gb:.1f} GB", '>= 8 GB', memory_gb >= 8))
        except ImportError:
            checks.append(('Available memory', 'Unknown', '>= 8 GB', True))
        
        # Check disk space
        try:
            disk = psutil.disk_usage(str(project_root))
            disk_gb = disk.free / (1024 ** 3)
            checks.append(('Free disk space', f"{disk_gb:.1f} GB", '>= 10 GB', disk_gb >= 10))
        except:
            checks.append(('Free disk space', 'Unknown', '>= 10 GB', True))
        
        # Log all checks
        all_passed = True
        for name, actual, required, passed in checks:
            status = "PASS" if passed else "FAIL"
            self._log_message("info", f"  {name}: {actual} (required: {required}) [{status}]")
            if not passed:
                all_passed = False
        
        if not all_passed:
            self._log_message("warning", "Some health checks failed. Pipeline may not run optimally.")
        
        return all_passed
    
    def _log_message(self, level, msg):
        """Universal logging method that works with any logger interface."""
        if hasattr(self.logger, level):
            getattr(self.logger, level)(msg)
        else:
            print(f"{level.upper()}: {msg}")


class MarinePollutionPipeline:
    """Main pipeline for marine pollution prediction."""
    
    def __init__(self, config, log_dir=None):
        self.config = config
        self.controller = PipelineController(config, log_dir)
        self.logger = self.controller.logger
        
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
            self._log_message("info", "=" * 80)
            self._log_message("info", "MARINE POLLUTION PREDICTION PIPELINE")
            self._log_message("info", "=" * 80)
            self._log_message("info", f"Experiment ID: {self.config.experiment_id}")
            self._log_message("info", f"Start time: {datetime.now().isoformat()}")
            
            # Perform health check
            self.controller.check_health()
            
            # Determine steps to run
            if steps is None:
                steps = ['download', 'process', 'train', 'evaluate']
            
            # Remove unavailable steps
            available_steps = []
            for step in steps:
                if step == 'download' and not available_modules['downloader']:
                    self._log_message("warning", "Download step skipped - downloader module not available")
                elif step == 'process' and not available_modules['data_loader']:
                    self._log_message("warning", "Process step skipped - data loader/preprocessor modules not available")
                elif step == 'train' and not available_modules['trainer']:
                    self._log_message("warning", "Train step skipped - trainer module not available")
                elif step == 'evaluate' and not available_modules['analyzer']:
                    self._log_message("warning", "Evaluate step skipped - analyzer module not available")
                else:
                    available_steps.append(step)
            
            if not available_steps:
                self._log_message("error", "No available steps to execute. Check module availability.")
                return
            
            self._log_message("info", f"Pipeline steps to execute: {', '.join(available_steps)}")
            
            # Execute pipeline steps
            pipeline_steps = {
                'download': self._download_data,
                'process': self._process_data,
                'train': self._train_model,
                'evaluate': self._evaluate_model
            }
            
            for step_name in available_steps:
                if step_name in pipeline_steps:
                    if not self.controller.is_running:
                        self._log_message("warning", "Pipeline interrupted by user")
                        break
                    
                    self._log_message("info", f"Executing step: {step_name}")
                    pipeline_steps[step_name]()
                else:
                    self._log_message("warning", f"Unknown pipeline step: {step_name}")
            
            # Generate final report
            self._generate_report()
            
            self._log_message("info", "Pipeline execution completed successfully")
            
        except Exception as e:
            self._log_message("error", f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _download_data(self):
        """Download marine data from Google Drive."""
        self._log_message("info", "Step 1: Downloading marine data")
        
        if not available_modules['downloader']:
            self._log_message("error", "Downloader module not available. Cannot download data.")
            return
        
        try:
            self.data_manager = MarineDataManager(data_dir=self.config.data.raw_dir.parent)
            
            download_report = self.data_manager.download_all_datasets(
                force_redownload=False
            )
            
            self.state['data_downloaded'] = True
            self._log_message("info", "Data download completed successfully")
            
        except Exception as e:
            self._log_message("error", f"Data download failed: {e}")
            raise
    
    def _process_data(self):
        """Process and prepare data for modeling."""
        self._log_message("info", "Step 2: Processing marine data")
        
        if not available_modules['data_loader']:
            self._log_message("error", "Data loader/preprocessor modules not available. Cannot process data.")
            return
        
        try:
            self.data_loader = MarineDataLoader(
                data_dir=self.config.data.raw_dir,
                config=self.config.data
            )
            
            # Load all NetCDF files
            dataset = self.data_loader.load_all_datasets()
            
            self.preprocessor = DataPreprocessor(config=self.config.data)
            
            # Process data
            processed_data = self.preprocessor.process(dataset)
            
            # Prepare features and target
            X, y, feature_names = self.preprocessor.prepare_features(
                processed_data,
                target_var=self.config.data.target_variable,
                feature_vars=self.config.data.feature_variables
            )
            
            # Check if splitter is available
            if available_modules['splitter']:
                splitter = DataSplitter(
                    test_size=self.config.training.test_size,
                    val_size=self.config.training.validation_size,
                    random_state=self.config.training.random_state
                )
                splits = splitter.split(X, y, temporal=True)
            else:
                # Simple split if splitter not available
                from sklearn.model_selection import train_test_split
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=self.config.training.test_size, random_state=self.config.training.random_state
                )
                val_size = self.config.training.validation_size / (1 - self.config.training.test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size, random_state=self.config.training.random_state
                )
                splits = {
                    'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                    'y_train': y_train, 'y_val': y_val, 'y_test': y_test
                }
            
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
            self._log_message("info", f"Data processing completed. Features: {len(feature_names)}")
            
            # Store processed data for next steps
            self.processed_data = {
                'splits': splits,
                'feature_names': feature_names,
                'output_dir': output_dir
            }
            
        except Exception as e:
            self._log_message("error", f"Data processing failed: {e}")
            raise
    
    def _train_model(self):
        """Train machine learning model."""
        self._log_message("info", "Step 3: Training prediction model")
        
        if not self.state['data_processed']:
            self._log_message("error", "Data not processed. Run process step first.")
            return
        
        if not available_modules['trainer']:
            self._log_message("error", "ModelTrainer module not available. Cannot train model.")
            return
        
        try:
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
            
            self.state['model_trained'] = True
            self._log_message("info", "Model training completed successfully")
            
            # Store trainer for evaluation
            self.training_results = training_results
            
        except Exception as e:
            self._log_message("error", f"Model training failed: {e}")
            raise
    
    def _evaluate_model(self):
        """Evaluate trained model."""
        self._log_message("info", "Step 4: Evaluating model performance")
        
        if not self.state['model_trained']:
            self._log_message("error", "Model not trained. Run train step first.")
            return
        
        if not available_modules['analyzer']:
            self._log_message("error", "ModelAnalyzer module not available. Cannot evaluate model.")
            return
        
        try:
            # Get test data
            splits = self.processed_data['splits']
            
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
            
            self.state['model_evaluated'] = True
            self._log_message("info", "Model evaluation completed")
            
        except Exception as e:
            self._log_message("error", f"Model evaluation failed: {e}")
            raise
    
    def _generate_report(self):
        """Generate final pipeline report."""
        self._log_message("info", "Generating final pipeline report...")
        
        report = {
            'experiment_id': self.config.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'pipeline_state': self.state,
            'available_modules': available_modules
        }
        
        # Save report
        report_dir = self.config.output_dir / self.config.experiment_id
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / "pipeline_report.json"
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self._log_message("info", f"Pipeline report saved to: {report_file}")
    
    def _log_message(self, level, msg):
        """Universal logging method that works with any logger interface."""
        if hasattr(self.logger, level):
            getattr(self.logger, level)(msg)
        else:
            print(f"{level.upper()}: {msg}")


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
        # Load or create configuration
        if available_modules['config']:
            config_manager = ConfigManager()
            if args.config:
                config = config_manager.load_config(args.config)
            else:
                config = ExperimentConfig()
                config.experiment_name = args.experiment_name
                config.output_dir = args.output_dir
                config.data.raw_dir = args.data_dir / "raw"
                config.data.processed_dir = args.data_dir / "processed"
        else:
            config = ExperimentConfig()
            config.experiment_name = args.experiment_name
            config.output_dir = args.output_dir
            config.data.raw_dir = args.data_dir / "raw"
            config.data.processed_dir = args.data_dir / "processed"
        
        # Update config with command line arguments
        if args.debug:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Create and run pipeline
        pipeline = MarinePollutionPipeline(config, args.log_dir)
        pipeline.run(steps=args.steps)
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETED")
        print("="*80)
        print(f"Experiment ID: {config.experiment_id}")
        print(f"Results directory: {config.output_dir / config.experiment_id}")
        print("="*80)
        
    except Exception as e:
        print(f"\nERROR: Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
