"""
Project-wide constants and configuration
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

# Data constants
DEFAULT_FEATURES = [
    'chl',          # Chlorophyll concentration
    'kd490',        # Diffuse attenuation coefficient at 490nm
    'zsd',          # Secchi disk depth
    'current_speed',
    'current_direction',
    'euphotic_depth',
    'clarity_index'
]

TARGET_VARIABLE = 'pp'  # Primary productivity

# Model constants
DEFAULT_MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 5,
        'random_state': 42
    },
    'neural_network': {
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 100
    }
}

# Uncertainty quantification
UNCERTAINTY_METHODS = [
    'ensemble',
    'bayesian',
    'gaussian_process',
    'mc_dropout',
    'quantile_regression'
]

@dataclass
class TrainingConfig:
    """Training configuration"""
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    n_folds: int = 5
    early_stopping_patience: int = 10
    max_epochs: int = 100
    
@dataclass
class DataConfig:
    """Data processing configuration"""
    chunk_size: Dict[str, int] = None
    interpolation_method: str = 'linear'
    outlier_threshold: float = 3.0
    normalize: bool = True
    scale_method: str = 'standard'
    
    def __post_init__(self):
        if self.chunk_size is None:
            self.chunk_size = {'time': 30, 'lat': 100, 'lon': 100}
