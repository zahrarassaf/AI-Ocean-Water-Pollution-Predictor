

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, Optional, Union
import joblib
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.logger import get_experiment_logger
    logger = get_experiment_logger()
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Model training will be limited.")


class ModelTrainer:
    """Simple model trainer for marine pollution prediction."""
    
    def __init__(self, config=None, output_dir=None):
        self.config = config or {}
        self.output_dir = Path(output_dir) if output_dir else Path("results/models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_names = None
        self.training_history = {}
        
        logger.info(f"ModelTrainer initialized with output_dir: {self.output_dir}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """Train a machine learning model."""
        logger.info("Starting model training...")
        logger.info(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
        
        if X_val is not None and y_val is not None:
            logger.info(f"Validation data shape: X_val={X_val.shape}, y_val={y_val.shape}")
        
        # Store feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        logger.info(f"Number of features: {len(self.feature_names)}")
        
        # Get model type from config
        model_type = getattr(self.config.model, 'model_type', 'random_forest') if hasattr(self.config, 'model') else 'random_forest'
        logger.info(f"Model type: {model_type}")
        
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn is required for model training but not available.")
            raise ImportError("scikit-learn is not installed. Please install it with: pip install scikit-learn")
        
        # Train model based on type
        if model_type.lower() == 'random_forest':
            self.model = self._train_random_forest(X_train, y_train, X_val, y_val)
        else:
            logger.warning(f"Unknown model type: {model_type}. Using Random Forest as default.")
            self.model = self._train_random_forest(X_train, y_train, X_val, y_val)
        
        # Make predictions
        train_pred = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred, "train")
        
        metrics = {
            'train': train_metrics,
            'model_type': model_type,
            'feature_count': X_train.shape[1]
        }
        
        # Evaluate on validation set if available
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred, "validation")
            metrics['validation'] = val_metrics
        
        # Save the trained model
        self._save_model(metrics)
        
        self.training_history = metrics
        logger.info("Model training completed successfully")
        
        return {
            'model': self.model,
            'metrics': metrics,
            'feature_names': self.feature_names,
            'output_dir': self.output_dir
        }
    
    def _train_random_forest(self, X_train, y_train, X_val=None, y_val=None):
        """Train a Random Forest model."""
        logger.info("Training Random Forest model...")
        
        # Basic hyperparameters
        n_estimators = 100
        max_depth = None
        min_samples_split = 2
        
        # Try to get hyperparameters from config
        try:
            if hasattr(self.config, 'model'):
                n_estimators = getattr(self.config.model, 'n_estimators', n_estimators)
                max_depth = getattr(self.config.model, 'max_depth', max_depth)
                min_samples_split = getattr(self.config.model, 'min_samples_split', min_samples_split)
        except:
            pass
        
        logger.info(f"Random Forest parameters: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
        
        # Create and train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(X_train, y_train)
        
        # Log feature importance if available
        if hasattr(model, 'feature_importances_') and self.feature_names:
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-10:]  # Top 10 features
            logger.info("Top 10 feature importances:")
            for idx in reversed(top_indices):
                logger.info(f"  {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        return model
    
    def _calculate_metrics(self, y_true, y_pred, dataset_name):
        """Calculate evaluation metrics."""
        metrics = {}
        
        try:
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2'] = float(r2_score(y_true, y_pred))
            
            # Calculate additional statistics
            metrics['samples'] = len(y_true)
            metrics['y_mean'] = float(np.mean(y_true))
            metrics['y_std'] = float(np.std(y_true))
            metrics['pred_mean'] = float(np.mean(y_pred))
            metrics['pred_std'] = float(np.std(y_pred))
            
            logger.info(f"{dataset_name} metrics - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
            
        except Exception as e:
            logger.warning(f"Could not calculate all metrics for {dataset_name}: {e}")
            # Basic metrics as fallback
            metrics['samples'] = len(y_true)
            if len(y_true) > 0:
                errors = y_true - y_pred
                metrics['mse'] = float(np.mean(errors ** 2))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
                metrics['mae'] = float(np.mean(np.abs(errors)))
        
        return metrics
    
    def predict(self, X, return_uncertainty=False):
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        logger.info(f"Making predictions for {len(X)} samples")
        
        predictions = self.model.predict(X)
        
        if return_uncertainty and hasattr(self.model, 'predict'):
            # For Random Forest, we can estimate uncertainty using individual trees
            try:
                all_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
                uncertainty = np.std(all_predictions, axis=0)
                logger.info(f"Uncertainty estimated (std across trees)")
                return predictions, uncertainty
            except:
                logger.warning("Could not estimate uncertainty")
                return predictions, None
        
        return predictions if not return_uncertainty else (predictions, None)
    
    def _save_model(self, metrics):
        """Save the trained model and metadata."""
        # Save model
        model_path = self.output_dir / "trained_model.joblib"
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config,
            'metrics': metrics
        }, model_path)
        
        # Save metrics separately
        metrics_path = self.output_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metrics saved to: {metrics_path}")
    
    def load_model(self, model_path):
        """Load a trained model."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        data = joblib.load(model_path)
        self.model = data['model']
        self.feature_names = data.get('feature_names')
        self.config = data.get('config', {})
        
        logger.info(f"Model loaded from: {model_path}")
        return self
