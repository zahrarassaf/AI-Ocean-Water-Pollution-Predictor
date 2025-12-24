"""
Configuration management with validation and type checking.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DataSource(Enum):
    GOOGLE_DRIVE = "google_drive"
    CMEMS = "cmems"
    LOCAL = "local"


@dataclass
class DataConfig:
    """Data configuration."""
    source: DataSource = DataSource.GOOGLE_DRIVE
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    interim_dir: Path = Path("data/interim")
    
    # Download settings
    download_parallel: int = 4
    download_chunk_size: int = 1024 * 1024  # 1MB
    download_timeout: int = 300
    validate_downloads: bool = True
    
    # Processing settings
    chunk_size: Dict[str, int] = field(default_factory=lambda: {
        'time': 30,
        'lat': 100,
        'lon': 100
    })
    interpolation_method: str = "linear"
    outlier_method: str = "iqr"
    outlier_threshold: float = 3.0
    
    # Target variable
    target_variable: str = "pp"  # Primary productivity
    
    # Feature variables
    feature_variables: List[str] = field(default_factory=lambda: [
        'chl',           # Chlorophyll concentration
        'kd490',         # Diffuse attenuation at 490nm
        'zsd',           # Secchi disk depth
        'sst',           # Sea surface temperature
        'sss',           # Sea surface salinity
        'current_speed',
        'current_direction'
    ])
    
    # Derived features
    create_derived_features: bool = True
    derived_features: List[str] = field(default_factory=lambda: [
        'euphotic_depth',
        'clarity_index',
        'current_energy',
        'chl_per_depth'
    ])
    
    # Temporal features
    include_temporal_features: bool = True
    temporal_features: List[str] = field(default_factory=lambda: [
        'day_of_year',
        'month',
        'season',
        'season_sin',
        'season_cos'
    ])
    
    # Validation
    required_variables: List[str] = field(default_factory=lambda: [
        'lat', 'lon', 'time'
    ])
    max_missing_percentage: float = 50.0
    
    def __post_init__(self):
        # Convert string paths to Path objects
        self.raw_dir = Path(self.raw_dir)
        self.processed_dir = Path(self.processed_dir)
        self.interim_dir = Path(self.interim_dir)
        
        # Ensure DataSource enum
        if isinstance(self.source, str):
            self.source = DataSource(self.source)


@dataclass
class ModelConfig:
    """Model configuration."""
    
    @dataclass
    class RandomForestConfig:
        n_estimators: int = 200
        max_depth: Optional[int] = None
        min_samples_split: int = 2
        min_samples_leaf: int = 1
        max_features: str = "sqrt"
        bootstrap: bool = True
        random_state: int = 42
    
    @dataclass
    class GradientBoostingConfig:
        n_estimators: int = 200
        learning_rate: float = 0.05
        max_depth: int = 5
        subsample: float = 0.8
        random_state: int = 42
    
    @dataclass
    class NeuralNetworkConfig:
        hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
        dropout_rate: float = 0.3
        learning_rate: float = 0.001
        batch_size: int = 64
        epochs: int = 100
        patience: int = 10
    
    # Model selection
    model_type: str = "ensemble"  # random_forest, gradient_boosting, neural_network, ensemble
    ensemble_method: str = "weighted_average"  # weighted_average, stacking
    
    # Individual model configurations
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    gradient_boosting: GradientBoostingConfig = field(default_factory=GradientBoostingConfig)
    neural_network: NeuralNetworkConfig = field(default_factory=NeuralNetworkConfig)
    
    # Uncertainty quantification
    uncertainty_method: str = "ensemble"  # ensemble, dropout, bayesian
    n_ensemble_models: int = 10
    mc_dropout_samples: int = 50
    
    # Feature importance
    calculate_feature_importance: bool = True
    importance_method: str = "permutation"  # permutation, shap, builtin


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Data splitting
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    shuffle: bool = True
    
    # Cross-validation
    cv_method: str = "timeseries"  # timeseries, kfold, stratified
    n_folds: int = 5
    cv_random_state: int = 42
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    
    # Optimization
    optimize_hyperparameters: bool = True
    optimization_method: str = "bayesian"  # grid, random, bayesian
    n_trials: int = 100
    
    # Metrics
    primary_metric: str = "rmse"  # rmse, mae, r2, mape
    secondary_metrics: List[str] = field(default_factory=lambda: [
        'mae', 'r2', 'explained_variance'
    ])
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: Path = Path("models/checkpoints")
    save_best_only: bool = True
    
    # Logging
    log_frequency: int = 10  # Log every N batches/epochs
    tensorboard_logging: bool = False
    wandb_logging: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Experiment metadata
    experiment_name: str = "marine_pollution_prediction"
    experiment_id: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Configuration sections
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Runtime settings
    use_gpu: bool = True
    num_workers: int = 4
    seed: int = 42
    
    # Output settings
    output_dir: Path = Path("results")
    save_predictions: bool = True
    save_model: bool = True
    save_config: bool = True
    
    def __post_init__(self):
        # Generate experiment ID if not provided
        if not self.experiment_id:
            from datetime import datetime
            self.experiment_id = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert paths
        self.output_dir = Path(self.output_dir)
        self.training.checkpoint_dir = Path(self.training.checkpoint_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Manage configuration loading, saving, and validation."""
    
    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_yaml(self, config_file: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_file)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return config_dict or {}
    
    def save_yaml(self, config: ExperimentConfig, config_file: Union[str, Path]):
        """Save configuration to YAML file."""
        config_file = Path(config_file)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self._config_to_dict(config)
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {config_file}")
    
    def load_config(self, config_file: Optional[Union[str, Path]] = None) -> ExperimentConfig:
        """
        Load configuration from file or create default.
        
        Args:
            config_file: Path to configuration file (YAML)
            
        Returns:
            ExperimentConfig object
        """
        if config_file:
            config_dict = self.load_yaml(config_file)
            config = self._dict_to_config(config_dict)
        else:
            config = ExperimentConfig()
        
        # Validate configuration
        self.validate_config(config)
        
        logger.info(f"Loaded configuration for experiment: {config.experiment_id}")
        
        return config
    
    def _config_to_dict(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Convert ExperimentConfig to dictionary."""
        def serialize(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, '__dataclass_fields__'):
                return {k: serialize(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            else:
                return obj
        
        return serialize(config)
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig."""
        
        def deserialize(data, target_type):
            if target_type == Path:
                return Path(data)
            elif hasattr(target_type, '__dataclass_fields__'):
                # Handle nested dataclasses
                field_types = {f.name: f.type for f in target_type.__dataclass_fields__.values()}
                kwargs = {}
                for field_name, field_type in field_types.items():
                    if field_name in data:
                        kwargs[field_name] = deserialize(data[field_name], field_type)
                return target_type(**kwargs)
            elif isinstance(data, dict) and 'value' in data and '__enum__' in data:
                # Handle enums
                enum_class = globals().get(data['__enum__'])
                if enum_class:
                    return enum_class(data['value'])
                return data
            else:
                return data
        
        # Start with default config
        config = ExperimentConfig()
        
        # Update with loaded values
        for key, value in config_dict.items():
            if hasattr(config, key):
                current_value = getattr(config, key)
                if hasattr(current_value, '__dataclass_fields__'):
                    # Nested dataclass
                    setattr(config, key, deserialize(value, type(current_value)))
                else:
                    setattr(config, key, value)
        
        return config
    
    def validate_config(self, config: ExperimentConfig):
        """Validate configuration parameters."""
        errors = []
        
        # Data validation
        if config.data.test_size + config.data.validation_size >= 1.0:
            errors.append("test_size + validation_size must be less than 1.0")
        
        if config.data.download_parallel < 1:
            errors.append("download_parallel must be at least 1")
        
        if config.data.max_missing_percentage < 0 or config.data.max_missing_percentage > 100:
            errors.append("max_missing_percentage must be between 0 and 100")
        
        # Model validation
        valid_model_types = ['random_forest', 'gradient_boosting', 'neural_network', 'ensemble']
        if config.model.model_type not in valid_model_types:
            errors.append(f"model_type must be one of {valid_model_types}")
        
        if config.model.n_ensemble_models < 1:
            errors.append("n_ensemble_models must be at least 1")
        
        # Training validation
        if config.training.test_size <= 0 or config.training.test_size >= 1:
            errors.append("test_size must be between 0 and 1")
        
        if config.training.validation_size < 0 or config.training.validation_size >= 1:
            errors.append("validation_size must be between 0 and 1")
        
        if config.training.n_folds < 2:
            errors.append("n_folds must be at least 2")
        
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def create_default_configs(self):
        """Create default configuration files."""
        default_configs = {
            'data_config.yaml': DataConfig(),
            'model_config.yaml': ModelConfig(),
            'training_config.yaml': TrainingConfig(),
            'experiment_config.yaml': ExperimentConfig()
        }
        
        for filename, config in default_configs.items():
            config_file = self.config_dir / filename
            if not config_file.exists():
                self.save_yaml(config, config_file)
                logger.info(f"Created default config: {config_file}")


# Convenience function
def get_config(config_file: Optional[Union[str, Path]] = None) -> ExperimentConfig:
    """Get configuration object."""
    manager = ConfigManager()
    return manager.load_config(config_file)
