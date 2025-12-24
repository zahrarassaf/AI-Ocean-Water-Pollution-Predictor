#!/usr/bin/env python3
"""
complete_training_pipeline.py
End-to-end training pipeline for marine pollution prediction.
Professional-grade with monitoring, validation, and deployment.
"""

import os
import sys
import argparse
import time
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import xarray as xr
import joblib
from scipy import stats
from sklearn.exceptions import ConvergenceWarning

# ML Imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error
)

# Models
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
rcParams['figure.figsize'] = (12, 8)
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12

class DataValidation:
    """Comprehensive data validation and quality checks."""
    
    def __init__(self):
        self.quality_report = {}
        
    def validate_dataset(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Perform comprehensive data validation."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'dimensions': dict(dataset.dims),
            'variables': list(dataset.data_vars.keys()),
            'quality_metrics': {},
            'issues': []
        }
        
        for var_name in dataset.data_vars:
            var_data = dataset[var_name]
            metrics = self._analyze_variable(var_data)
            report['quality_metrics'][var_name] = metrics
            
            # Check for issues
            if metrics['missing_percentage'] > 50:
                report['issues'].append(f"High missing values in {var_name}: {metrics['missing_percentage']:.1f}%")
            if metrics['constant_check']:
                report['issues'].append(f"Constant variable: {var_name}")
            if metrics['outlier_percentage'] > 10:
                report['issues'].append(f"High outliers in {var_name}: {metrics['outlier_percentage']:.1f}%")
        
        return report
    
    def _analyze_variable(self, var_data: xr.DataArray) -> Dict[str, float]:
        """Analyze individual variable quality."""
        values = var_data.values.flatten()
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) == 0:
            return {
                'missing_percentage': 100.0,
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'constant_check': True,
                'outlier_percentage': 0.0
            }
        
        # Basic statistics
        stats_dict = {
            'missing_percentage': 100 * np.isnan(values).sum() / len(values),
            'mean': float(np.mean(valid_values)),
            'std': float(np.std(valid_values)),
            'min': float(np.min(valid_values)),
            'max': float(np.max(valid_values)),
            'q1': float(np.percentile(valid_values, 25)),
            'median': float(np.median(valid_values)),
            'q3': float(np.percentile(valid_values, 75)),
            'skewness': float(stats.skew(valid_values)),
            'kurtosis': float(stats.kurtosis(valid_values))
        }
        
        # Check for constant values
        stats_dict['constant_check'] = np.std(valid_values) < 1e-10
        
        # Detect outliers using IQR
        Q1 = stats_dict['q1']
        Q3 = stats_dict['q3']
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = valid_values[(valid_values < lower_bound) | (valid_values > upper_bound)]
        stats_dict['outlier_percentage'] = 100 * len(outliers) / len(valid_values)
        
        return stats_dict

class FeatureEngineering:
    """Advanced feature engineering for marine data."""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = RobustScaler()
        
    def create_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Create advanced features for time-series spatial data."""
        
        if X.shape[1] == 0:
            raise ValueError("No input features provided")
        
        print(f"Original features: {len(feature_names)}")
        print(f"Original shape: {X.shape}")
        
        # Store original features
        X_enhanced = X.copy()
        enhanced_names = feature_names.copy()
        
        # Create interaction features for key variables
        key_features = ['CHL', 'PP', 'KD490', 'CDM']
        available_key_features = [f for f in key_features if f in feature_names]
        
        if len(available_key_features) >= 2:
            for i, f1 in enumerate(available_key_features):
                for f2 in available_key_features[i+1:]:
                    idx1 = feature_names.index(f1)
                    idx2 = feature_names.index(f2)
                    
                    # Create ratio
                    ratio_feature = X[:, idx1] / (X[:, idx2] + 1e-10)
                    X_enhanced = np.column_stack([X_enhanced, ratio_feature])
                    enhanced_names.append(f"{f1}_to_{f2}_ratio")
                    
                    # Create product
                    product_feature = X[:, idx1] * X[:, idx2]
                    X_enhanced = np.column_stack([X_enhanced, product_feature])
                    enhanced_names.append(f"{f1}_{f2}_product")
        
        # Create statistical features
        statistical_features = []
        if X.shape[0] > 100:  # Only if we have enough samples
            # Rolling statistics (simulated)
            for i, name in enumerate(feature_names):
                # Moving average
                window_size = min(100, X.shape[0])
                if X.shape[0] > window_size:
                    ma_feature = np.convolve(X[:, i], np.ones(window_size)/window_size, mode='valid')
                    # Pad to match length
                    ma_feature = np.pad(ma_feature, (X.shape[0] - len(ma_feature), 0), 'constant')
                    X_enhanced = np.column_stack([X_enhanced, ma_feature])
                    enhanced_names.append(f"{name}_ma_{window_size}")
                    statistical_features.append(ma_feature)
        
        # Add polynomial features for important variables
        important_vars = ['CHL', 'PP', 'KD490']
        for var in important_vars:
            if var in feature_names:
                idx = feature_names.index(var)
                # Square
                X_enhanced = np.column_stack([X_enhanced, X[:, idx] ** 2])
                enhanced_names.append(f"{var}_squared")
                # Cube root
                X_enhanced = np.column_stack([X_enhanced, np.cbrt(np.abs(X[:, idx] + 1e-10))])
                enhanced_names.append(f"{var}_cube_root")
        
        # Create spatial-temporal features (simplified)
        if 'lat' in feature_names and 'lon' in feature_names:
            lat_idx = feature_names.index('lat')
            lon_idx = feature_names.index('lon')
            
            # Calculate approximate distance from center
            center_lat = np.mean(X[:, lat_idx])
            center_lon = np.mean(X[:, lon_idx])
            distance = np.sqrt((X[:, lat_idx] - center_lat)**2 + 
                             (X[:, lon_idx] - center_lon)**2)
            
            X_enhanced = np.column_stack([X_enhanced, distance])
            enhanced_names.append('distance_from_center')
        
        print(f"Enhanced features: {len(enhanced_names)}")
        print(f"Enhanced shape: {X_enhanced.shape}")
        
        # Remove any columns with NaN or Inf
        valid_mask = ~np.any(np.isnan(X_enhanced) | np.isinf(X_enhanced), axis=0)
        X_enhanced = X_enhanced[:, valid_mask]
        enhanced_names = [name for name, valid in zip(enhanced_names, valid_mask) if valid]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_enhanced)
        
        self.feature_names = enhanced_names
        return X_scaled, enhanced_names
    
    def get_feature_importance(self, model, top_n: int = 20) -> pd.DataFrame:
        """Extract and sort feature importances."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            return pd.DataFrame()
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(importances)],
            'importance': importances
        })
        
        # Sort and select top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        importance_df['importance_normalized'] = importance_df['importance'] / importance_df['importance'].sum()
        
        return importance_df

class ModelRegistry:
    """Registry for managing multiple ML models."""
    
    def __init__(self):
        self.models = {}
        self.performance_history = []
        
    def register_model(self, name: str, model: Any, description: str = ""):
        """Register a model in the registry."""
        self.models[name] = {
            'model': model,
            'description': description,
            'created': datetime.now().isoformat(),
            'performance': None
        }
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train all registered models."""
        results = {}
        
        for name, model_info in self.models.items():
            print(f"Training {name}...")
            start_time = time.time()
            
            try:
                model = model_info['model']
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                val_score = model.score(X_val, y_val) if X_val is not None else None
                
                results[name] = {
                    'training_time': training_time,
                    'train_score': train_score,
                    'val_score': val_score,
                    'model': model
                }
                
                # Update registry
                self.models[name]['performance'] = results[name]
                self.models[name]['last_trained'] = datetime.now().isoformat()
                
                print(f"  ✓ Training time: {training_time:.2f}s")
                print(f"  ✓ Train R²: {train_score:.4f}")
                if val_score is not None:
                    print(f"  ✓ Validation R²: {val_score:.4f}")
                    
            except Exception as e:
                print(f"  ✗ Failed to train {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
        
        return results
    
    def get_best_model(self, metric: str = 'val_score') -> Tuple[str, Any]:
        """Get the best performing model based on specified metric."""
        if not self.models:
            raise ValueError("No models in registry")
        
        best_score = -float('inf')
        best_name = None
        best_model = None
        
        for name, info in self.models.items():
            if info['performance'] and metric in info['performance']:
                score = info['performance'][metric]
                if score is not None and score > best_score:
                    best_score = score
                    best_name = name
                    best_model = info['model']
        
        return best_name, best_model

class AdvancedMetrics:
    """Comprehensive model evaluation metrics."""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_uncertainty: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) if np.all(y_true != 0) else np.nan
        }
        
        # Additional custom metrics
        residuals = y_true - y_pred
        metrics['mean_residual'] = float(np.mean(residuals))
        metrics['std_residual'] = float(np.std(residuals))
        metrics['residual_skewness'] = float(stats.skew(residuals))
        metrics['residual_kurtosis'] = float(stats.kurtosis(residuals))
        
        # Calculate prediction intervals coverage (if uncertainty provided)
        if y_uncertainty is not None:
            z_score = 1.96  # 95% confidence
            lower_bound = y_pred - z_score * y_uncertainty
            upper_bound = y_pred + z_score * y_uncertainty
            coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
            metrics['coverage_95'] = coverage
            metrics['mean_uncertainty'] = float(np.mean(y_uncertainty))
        
        # Calculate efficiency metrics
        y_mean = np.mean(y_true)
        sst = np.sum((y_true - y_mean) ** 2)
        ssr = np.sum(residuals ** 2)
        metrics['nash_sutcliffe'] = 1 - (ssr / sst) if sst != 0 else np.nan
        
        return metrics
    
    @staticmethod
    def calculate_quantile_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                  quantiles: List[float] = None) -> Dict[str, float]:
        """Calculate metrics at different quantiles."""
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        quantile_metrics = {}
        for q in quantiles:
            q_pred = np.percentile(y_pred, q * 100)
            q_true = np.percentile(y_true, q * 100)
            quantile_metrics[f'quantile_{q}_error'] = abs(q_pred - q_true)
            quantile_metrics[f'quantile_{q}_ratio'] = q_pred / (q_true + 1e-10)
        
        return quantile_metrics

class VisualizationEngine:
    """Advanced visualization for model analysis."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_prediction_plot(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              title: str = "Predictions vs Actual"):
        """Create prediction vs actual plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                       'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title(f'{title} - Scatter Plot')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution comparison
        axes[1, 0].hist(y_true, bins=50, alpha=0.5, density=True, label='Actual')
        axes[1, 0].hist(y_pred, bins=50, alpha=0.5, density=True, label='Predicted')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error distribution
        axes[1, 1].hist(residuals, bins=50, density=True)
        axes[1, 1].set_xlabel('Residual')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_importance_plot(self, importance_df: pd.DataFrame, 
                                      title: str = "Feature Importance"):
        """Create feature importance visualization."""
        if importance_df.empty:
            print("No feature importance data available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Bar plot
        importance_df_sorted = importance_df.sort_values('importance_normalized')
        axes[0].barh(range(len(importance_df_sorted)), 
                    importance_df_sorted['importance_normalized'].values)
        axes[0].set_yticks(range(len(importance_df_sorted)))
        axes[0].set_yticklabels(importance_df_sorted['feature'].values)
        axes[0].set_xlabel('Normalized Importance')
        axes[0].set_title(f'{title} - Top {len(importance_df)} Features')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Cumulative importance
        importance_df_sorted = importance_df_sorted.sort_values('importance_normalized', 
                                                               ascending=False)
        cumulative_importance = importance_df_sorted['importance_normalized'].cumsum()
        
        axes[1].plot(range(1, len(cumulative_importance) + 1), 
                    cumulative_importance.values, 
                    marker='o', linewidth=2)
        axes[1].axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(y=0.9, color='g', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Number of Features')
        axes[1].set_ylabel('Cumulative Importance')
        axes[1].set_title('Cumulative Feature Importance')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.1])
        
        # Annotate 80% and 90% thresholds
        idx_80 = np.where(cumulative_importance >= 0.8)[0][0]
        idx_90 = np.where(cumulative_importance >= 0.9)[0][0]
        
        axes[1].annotate(f'80%: {idx_80 + 1} features', 
                        xy=(idx_80 + 1, 0.8), 
                        xytext=(idx_80 + 1, 0.6),
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        axes[1].annotate(f'90%: {idx_90 + 1} features', 
                        xy=(idx_90 + 1, 0.9), 
                        xytext=(idx_90 + 1, 0.7),
                        arrowprops=dict(arrowstyle='->', color='green'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_correlation_matrix(self, X: np.ndarray, feature_names: List[str],
                                 top_n: int = 20):
        """Create correlation matrix visualization."""
        if len(feature_names) > top_n:
            # Select most variable features
            variances = np.var(X, axis=0)
            top_indices = np.argsort(variances)[-top_n:]
            X_subset = X[:, top_indices]
            feature_names_subset = [feature_names[i] for i in top_indices]
        else:
            X_subset = X
            feature_names_subset = feature_names
        
        corr_matrix = np.corrcoef(X_subset, rowvar=False)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, 
                   xticklabels=feature_names_subset,
                   yticklabels=feature_names_subset,
                   cmap='RdBu_r', center=0,
                   square=True, 
                   cbar_kws={"shrink": 0.8})
        
        plt.title(f'Correlation Matrix (Top {len(feature_names_subset)} Features)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

class ModelPersistence:
    """Handle model serialization and versioning."""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def save_model(self, model: Any, model_name: str, metadata: Dict[str, Any]):
        """Save model with comprehensive metadata."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{model_name}_{timestamp}.joblib"
        model_path = self.model_dir / model_filename
        
        # Create save dictionary
        save_dict = {
            'model': model,
            'metadata': metadata,
            'timestamp': timestamp,
            'python_version': sys.version,
            'dependencies': self._get_dependencies()
        }
        
        # Save model
        joblib.dump(save_dict, model_path, compress=3)
        
        # Save metadata separately
        metadata_path = self.model_dir / f"{model_name}_{timestamp}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    def load_model(self, model_path: Path) -> Tuple[Any, Dict[str, Any]]:
        """Load model and metadata."""
        save_dict = joblib.load(model_path)
        return save_dict['model'], save_dict['metadata']
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get current dependency versions."""
        import importlib.metadata
        
        dependencies = {}
        packages = ['scikit-learn', 'numpy', 'pandas', 'scipy', 'joblib', 'xarray']
        
        for package in packages:
            try:
                version = importlib.metadata.version(package)
                dependencies[package] = version
            except:
                dependencies[package] = 'unknown'
        
        return dependencies

class MarinePollutionTrainer:
    """Main training pipeline orchestrator."""
    
    def __init__(self, config_path: Path = None):
        self.config = self._load_config(config_path)
        self.experiment_id = self._generate_experiment_id()
        self.output_dir = Path(self.config.get('output_dir', 'results')) / self.experiment_id
        
        # Initialize components
        self.validator = DataValidation()
        self.feature_engineer = FeatureEngineering()
        self.model_registry = ModelRegistry()
        self.metrics_calculator = AdvancedMetrics()
        self.visualizer = VisualizationEngine(self.output_dir)
        self.model_persistor = ModelPersistence(self.output_dir / 'models')
        
        # Training state
        self.training_history = []
        self.best_model = None
        self.best_score = -float('inf')
        
        # Create output directories
        (self.output_dir / 'plots').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'reports').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: Path = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'random_state': 42,
            'test_size': 0.2,
            'validation_size': 0.1,
            'models': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'n_jobs': -1
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8
                },
                'lightgbm': {
                    'n_estimators': 100,
                    'max_depth': -1,
                    'learning_rate': 0.05,
                    'num_leaves': 31
                }
            },
            'feature_engineering': {
                'create_interactions': True,
                'create_polynomials': True,
                'create_statistical': True
            },
            'output_dir': 'results',
            'cross_validation': {
                'n_splits': 5,
                'shuffle': True
            }
        }
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"marine_pollution_{timestamp}"
    
    def load_processed_data(self, data_path: Path) -> Dict[str, Any]:
        """Load preprocessed data from pipeline."""
        print(f"Loading processed data from: {data_path}")
        
        data = joblib.load(data_path)
        
        if 'splits' not in data:
            raise ValueError("Processed data must contain 'splits' key")
        
        splits = data['splits']
        feature_names = data.get('feature_names', [])
        
        return {
            'X_train': splits['X_train'],
            'X_test': splits['X_test'],
            'X_val': splits.get('X_val', None),
            'y_train': splits['y_train'],
            'y_test': splits['y_test'],
            'y_val': splits.get('y_val', None),
            'feature_names': feature_names
        }
    
    def prepare_data(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for training with feature engineering."""
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        X_val = data_dict.get('X_val', None)
        feature_names = data_dict['feature_names']
        
        print(f"Original training shape: {X_train.shape}")
        print(f"Original test shape: {X_test.shape}")
        print(f"Number of features: {len(feature_names)}")
        
        # Apply feature engineering
        X_train_enhanced, enhanced_names = self.feature_engineer.create_features(
            X_train, feature_names
        )
        
        X_test_enhanced, _ = self.feature_engineer.create_features(
            X_test, feature_names
        )
        
        if X_val is not None:
            X_val_enhanced, _ = self.feature_engineer.create_features(
                X_val, feature_names
            )
        else:
            X_val_enhanced = None
        
        print(f"Enhanced training shape: {X_train_enhanced.shape}")
        print(f"Enhanced test shape: {X_test_enhanced.shape}")
        
        return {
            'X_train': X_train_enhanced,
            'X_test': X_test_enhanced,
            'X_val': X_val_enhanced,
            'y_train': data_dict['y_train'],
            'y_test': data_dict['y_test'],
            'y_val': data_dict.get('y_val', None),
            'feature_names': enhanced_names
        }
    
    def initialize_models(self):
        """Initialize and register ML models."""
        print("\n" + "="*60)
        print("MODEL INITIALIZATION")
        print("="*60)
        
        # Random Forest
        rf_params = self.config['models'].get('random_forest', {})
        rf_model = RandomForestRegressor(
            n_estimators=rf_params.get('n_estimators', 100),
            max_depth=rf_params.get('max_depth', 20),
            min_samples_split=rf_params.get('min_samples_split', 5),
            random_state=self.config['random_state'],
            n_jobs=rf_params.get('n_jobs', -1),
            verbose=1
        )
        self.model_registry.register_model('RandomForest', rf_model, 
                                         "Random Forest Regressor")
        
        # XGBoost
        xgb_params = self.config['models'].get('xgboost', {})
        xgb_model = xgb.XGBRegressor(
            n_estimators=xgb_params.get('n_estimators', 100),
            max_depth=xgb_params.get('max_depth', 6),
            learning_rate=xgb_params.get('learning_rate', 0.1),
            subsample=xgb_params.get('subsample', 0.8),
            random_state=self.config['random_state'],
            verbosity=1
        )
        self.model_registry.register_model('XGBoost', xgb_model,
                                         "Gradient Boosting with XGBoost")
        
        # LightGBM
        lgb_params = self.config['models'].get('lightgbm', {})
        lgb_model = lgb.LGBMRegressor(
            n_estimators=lgb_params.get('n_estimators', 100),
            max_depth=lgb_params.get('max_depth', -1),
            learning_rate=lgb_params.get('learning_rate', 0.05),
            num_leaves=lgb_params.get('num_leaves', 31),
            random_state=self.config['random_state'],
            verbose=-1
        )
        self.model_registry.register_model('LightGBM', lgb_model,
                                         "Light Gradient Boosting Machine")
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.config['random_state'],
            verbose=1
        )
        self.model_registry.register_model('GradientBoosting', gb_model,
                                         "Gradient Boosting Regressor")
        
        # Extra Trees
        et_model = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=self.config['random_state'],
            n_jobs=-1,
            verbose=1
        )
        self.model_registry.register_model('ExtraTrees', et_model,
                                         "Extra Trees Regressor")
        
        print(f"Registered {len(self.model_registry.models)} models")
    
    def train_models(self, data_dict: Dict[str, Any]):
        """Train all registered models."""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_val = data_dict.get('X_val')
        y_val = data_dict.get('y_val')
        
        # Train all models
        training_results = self.model_registry.train_all(X_train, y_train, X_val, y_val)
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'data_shape': X_train.shape,
            'results': training_results
        })
        
        return training_results
    
    def evaluate_models(self, data_dict: Dict[str, Any], training_results: Dict[str, Any]):
        """Comprehensive model evaluation."""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        evaluation_report = {
            'experiment_id': self.experiment_id,
            'evaluation_time': datetime.now().isoformat(),
            'test_set_size': len(y_test),
            'models': {}
        }
        
        best_model_name = None
        best_score = -float('inf')
        
        for model_name, model_info in training_results.items():
            if 'error' in model_info:
                print(f"Skipping evaluation for {model_name}: {model_info['error']}")
                continue
            
            model = model_info['model']
            
            print(f"\nEvaluating {model_name}:")
            print("-" * 40)
            
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_all_metrics(y_test, y_pred)
                
                # Calculate uncertainty (for models that support it)
                y_uncertainty = None
                if hasattr(model, 'estimators_'):
                    tree_predictions = np.array([tree.predict(X_test) for tree in model.estimators_])
                    y_uncertainty = np.std(tree_predictions, axis=0)
                    metrics.update({
                        'mean_uncertainty': float(np.mean(y_uncertainty)),
                        'uncertainty_std': float(np.std(y_uncertainty))
                    })
                
                # Store results
                evaluation_report['models'][model_name] = {
                    'metrics': metrics,
                    'training_time': model_info.get('training_time', None),
                    'train_score': model_info.get('train_score', None),
                    'val_score': model_info.get('val_score', None)
                }
                
                # Print results
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  MAE: {metrics['mae']:.4f}")
                print(f"  R²: {metrics['r2']:.4f}")
                print(f"  Explained Variance: {metrics['explained_variance']:.4f}")
                
                if y_uncertainty is not None:
                    print(f"  Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")
                
                # Update best model
                if metrics['r2'] > best_score:
                    best_score = metrics['r2']
                    best_model_name = model_name
                    self.best_model = model
                    self.best_score = best_score
                    
                    # Create visualizations for best model
                    self.visualizer.create_prediction_plot(
                        y_test, y_pred, 
                        title=f"{model_name} - Predictions vs Actual"
                    )
                    
                    # Create feature importance plot
                    if hasattr(model, 'feature_importances_'):
                        importance_df = self.feature_engineer.get_feature_importance(
                            model, top_n=20
                        )
                        if not importance_df.empty:
                            self.visualizer.create_feature_importance_plot(
                                importance_df, 
                                title=f"{model_name} - Feature Importance"
                            )
                
            except Exception as e:
                print(f"  ✗ Evaluation failed: {e}")
                evaluation_report['models'][model_name] = {'error': str(e)}
        
        # Record best model
        if best_model_name:
            evaluation_report['best_model'] = {
                'name': best_model_name,
                'score': best_score,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"\n{'='*60}")
            print(f"BEST MODEL: {best_model_name}")
            print(f"R² Score: {best_score:.4f}")
            print(f"{'='*60}")
        
        # Save evaluation report
        report_path = self.output_dir / 'reports' / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        print(f"\nEvaluation report saved to: {report_path}")
        
        return evaluation_report
    
    def save_best_model(self, evaluation_report: Dict[str, Any], 
                       data_dict: Dict[str, Any]):
        """Save the best performing model with metadata."""
        if self.best_model is None:
            print("No best model to save")
            return
        
        # Get best model info
        best_model_info = evaluation_report.get('best_model', {})
        model_name = best_model_info.get('name', 'unknown')
        
        # Prepare metadata
        metadata = {
            'experiment_id': self.experiment_id,
            'model_name': model_name,
            'performance': best_model_info,
            'feature_names': data_dict['feature_names'],
            'feature_count': len(data_dict['feature_names']),
            'training_samples': len(data_dict['y_train']),
            'test_samples': len(data_dict['y_test']),
            'config': self.config,
            'creation_time': datetime.now().isoformat()
        }
        
        # Save model
        model_path = self.model_persistor.save_model(
            self.best_model, model_name, metadata
        )
        
        print(f"\nBest model saved to: {model_path}")
        
        # Create model card
        self._create_model_card(model_name, metadata, evaluation_report)
        
        return model_path
    
    def _create_model_card(self, model_name: str, metadata: Dict[str, Any], 
                          evaluation_report: Dict[str, Any]):
        """Create comprehensive model documentation."""
        model_card = {
            'model_name': model_name,
            'version': '1.0.0',
            'description': f'Marine Pollution Prediction Model - {model_name}',
            'authors': ['AI Ocean Team'],
            'date_created': datetime.now().isoformat(),
            'license': 'MIT',
            'intended_use': 'Research and environmental monitoring',
            'limitations': 'Model trained on specific time period and region',
            'performance': evaluation_report['models'].get(model_name, {}),
            'training_data': {
                'samples': metadata['training_samples'],
                'features': metadata['feature_count'],
                'period': 'Not specified',
                'region': 'Global oceans'
            },
            'evaluation_metrics': {
                'primary': 'R² Score',
                'secondary': ['RMSE', 'MAE', 'Explained Variance']
            },
            'ethical_considerations': {
                'bias': 'Potential geographical bias in training data',
                'fairness': 'Model should be validated across diverse regions',
                'environmental_impact': 'Positive impact through pollution monitoring'
            }
        }
        
        model_card_path = self.output_dir / 'reports' / 'model_card.json'
        with open(model_card_path, 'w') as f:
            json.dump(model_card, f, indent=2, default=str)
        
        print(f"Model card created: {model_card_path}")
    
    def create_summary_report(self, data_dict: Dict[str, Any], 
                            evaluation_report: Dict[str, Any]):
        """Create comprehensive summary report."""
        print("\n" + "="*60)
        print("GENERATING SUMMARY REPORT")
        print("="*60)
        
        summary = {
            'experiment_summary': {
                'id': self.experiment_id,
                'start_time': self.training_history[0]['timestamp'] if self.training_history else None,
                'end_time': datetime.now().isoformat(),
                'duration': 'Not tracked',
                'output_directory': str(self.output_dir)
            },
            'data_summary': {
                'training_samples': len(data_dict['y_train']),
                'test_samples': len(data_dict['y_test']),
                'validation_samples': len(data_dict['y_val']) if data_dict.get('y_val') is not None else 0,
                'total_features': len(data_dict['feature_names']),
                'feature_engineering_applied': True
            },
            'model_performance': {},
            'best_model': evaluation_report.get('best_model', {}),
            'recommendations': []
        }
        
        # Compile model performances
        for model_name, model_info in evaluation_report.get('models', {}).items():
            if 'metrics' in model_info:
                summary['model_performance'][model_name] = {
                    'r2': model_info['metrics']['r2'],
                    'rmse': model_info['metrics']['rmse'],
                    'mae': model_info['metrics']['mae'],
                    'training_time': model_info.get('training_time')
                }
        
        # Generate recommendations
        best_r2 = summary['best_model'].get('score', 0)
        
        if best_r2 > 0.8:
            summary['recommendations'].append("Excellent model performance - ready for deployment")
        elif best_r2 > 0.6:
            summary['recommendations'].append("Good model performance - consider fine-tuning")
        elif best_r2 > 0.4:
            summary['recommendations'].append("Moderate performance - investigate feature engineering")
        else:
            summary['recommendations'].append("Poor performance - revisit data preprocessing and feature selection")
        
        summary['recommendations'].extend([
            "Validate model on unseen temporal and spatial data",
            "Monitor model performance over time",
            "Consider ensemble methods for improved robustness"
        ])
        
        # Save summary
        summary_path = self.output_dir / 'reports' / 'summary_report.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        print(f"\nEXPERIMENT SUMMARY")
        print(f"{'='*40}")
        print(f"Experiment ID: {summary['experiment_summary']['id']}")
        print(f"Output Directory: {summary['experiment_summary']['output_directory']}")
        print(f"\nDATA SUMMARY")
        print(f"  Training Samples: {summary['data_summary']['training_samples']:,}")
        print(f"  Test Samples: {summary['data_summary']['test_samples']:,}")
        print(f"  Features: {summary['data_summary']['total_features']}")
        print(f"\nBEST MODEL")
        print(f"  Model: {summary['best_model'].get('name', 'N/A')}")
        print(f"  R² Score: {summary['best_model'].get('score', 0):.4f}")
        print(f"\nRECOMMENDATIONS")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"\nSummary report saved to: {summary_path}")
        
        return summary
    
    def run(self, data_path: Path, steps: List[str] = None):
        """Run the complete training pipeline."""
        start_time = time.time()
        
        try:
            print("\n" + "="*80)
            print("MARINE POLLUTION PREDICTION - TRAINING PIPELINE")
            print("="*80)
            print(f"Experiment ID: {self.experiment_id}")
            print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Output Directory: {self.output_dir}")
            print("="*80)
            
            # Define default steps
            if steps is None:
                steps = ['load_data', 'prepare_data', 'initialize_models', 
                        'train_models', 'evaluate_models', 'save_model', 
                        'create_summary']
            
            # Load data
            if 'load_data' in steps:
                print("\n[STEP 1] Loading processed data...")
                data_dict = self.load_processed_data(data_path)
            
            # Prepare data
            if 'prepare_data' in steps:
                print("\n[STEP 2] Preparing data with feature engineering...")
                data_dict = self.prepare_data(data_dict)
                
                # Create correlation matrix visualization
                self.visualizer.create_correlation_matrix(
                    data_dict['X_train'], 
                    data_dict['feature_names']
                )
            
            # Initialize models
            if 'initialize_models' in steps:
                print("\n[STEP 3] Initializing machine learning models...")
                self.initialize_models()
            
            # Train models
            if 'train_models' in steps:
                print("\n[STEP 4] Training models...")
                training_results = self.train_models(data_dict)
            
            # Evaluate models
            if 'evaluate_models' in steps:
                print("\n[STEP 5] Evaluating model performance...")
                evaluation_report = self.evaluate_models(data_dict, training_results)
            
            # Save best model
            if 'save_model' in steps:
                print("\n[STEP 6] Saving best model...")
                model_path = self.save_best_model(evaluation_report, data_dict)
            
            # Create summary
            if 'create_summary' in steps:
                print("\n[STEP 7] Creating summary report...")
                summary = self.create_summary_report(data_dict, evaluation_report)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            print("\n" + "="*80)
            print("PIPELINE EXECUTION COMPLETE")
            print("="*80)
            print(f"Total Execution Time: {execution_time:.2f} seconds")
            print(f"Results Directory: {self.output_dir}")
            print("="*80)
            
            return {
                'success': True,
                'experiment_id': self.experiment_id,
                'output_dir': str(self.output_dir),
                'execution_time': execution_time,
                'best_model_score': self.best_score
            }
            
        except Exception as e:
            print(f"\nERROR: Pipeline execution failed")
            print(f"Error: {e}")
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'experiment_id': self.experiment_id
            }

def main():
    """Main entry point for the training pipeline."""
    parser = argparse.ArgumentParser(
        description='Marine Pollution Prediction - Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s data/processed/processed_data.joblib
  %(prog)s data/processed/processed_data.joblib --config config/training_config.json
  %(prog)s data/processed/processed_data.joblib --skip-steps load_data,prepare_data
        '''
    )
    
    parser.add_argument('data_path', 
                       type=Path, 
                       help='Path to processed data file (.joblib)')
    
    parser.add_argument('--config', 
                       type=Path, 
                       default=None,
                       help='Path to configuration file (JSON)')
    
    parser.add_argument('--output-dir', 
                       type=Path, 
                       default='results',
                       help='Output directory for results')
    
    parser.add_argument('--skip-steps',
                       type=str,
                       default='',
                       help='Comma-separated list of steps to skip')
    
    parser.add_argument('--only-steps',
                       type=str,
                       default='',
                       help='Comma-separated list of steps to execute (overrides skip-steps)')
    
    parser.add_argument('--experiment-id',
                       type=str,
                       default=None,
                       help='Custom experiment ID')
    
    args = parser.parse_args()
    
    # Validate data path
    if not args.data_path.exists():
        print(f"Error: Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Create trainer
    trainer = MarinePollutionTrainer(args.config)
    
    # Override experiment ID if provided
    if args.experiment_id:
        trainer.experiment_id = args.experiment_id
    
    # Override output directory
    if args.output_dir:
        trainer.config['output_dir'] = str(args.output_dir)
        trainer.output_dir = args.output_dir / trainer.experiment_id
    
    # Determine steps to execute
    all_steps = ['load_data', 'prepare_data', 'initialize_models', 
                 'train_models', 'evaluate_models', 'save_model', 'create_summary']
    
    if args.only_steps:
        steps_to_run = [s.strip() for s in args.only_steps.split(',') if s.strip()]
        # Validate steps
        invalid_steps = [s for s in steps_to_run if s not in all_steps]
        if invalid_steps:
            print(f"Error: Invalid steps specified: {invalid_steps}")
            print(f"Valid steps are: {', '.join(all_steps)}")
            sys.exit(1)
    else:
        steps_to_run = all_steps.copy()
        if args.skip_steps:
            steps_to_skip = [s.strip() for s in args.skip_steps.split(',') if s.strip()]
            steps_to_run = [s for s in steps_to_run if s not in steps_to_skip]
    
    print(f"Steps to execute: {', '.join(steps_to_run)}")
    
    # Run pipeline
    result = trainer.run(args.data_path, steps_to_run)
    
    if result['success']:
        print(f"\n✅ Training pipeline completed successfully!")
        print(f"   Experiment ID: {result['experiment_id']}")
        print(f"   Best Model R²: {result.get('best_model_score', 0):.4f}")
        sys.exit(0)
    else:
        print(f"\n❌ Training pipeline failed!")
        print(f"   Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == '__main__':
    main()
