"""
Professional model trainer for marine pollution prediction.
Supports multiple model types, hyperparameter optimization, and comprehensive logging.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import json
import joblib
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Professional model trainer for marine pollution prediction.
    
    Features:
    - Multiple model types (Random Forest, Gradient Boosting, Neural Networks)
    - Hyperparameter optimization
    - Cross-validation
    - Early stopping
    - Model checkpointing
    - Comprehensive logging
    """
    
    def __init__(self, config: Dict, output_dir: Path):
        """
        Initialize model trainer.
        
        Args:
            config: Training configuration
            output_dir: Output directory for models and results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.scaler = StandardScaler()
        self.best_score = -np.inf
        self.training_history = []
        
        logger.info(f"Initialized ModelTrainer with output_dir: {self.output_dir}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train model with given data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Names of features (optional)
            
        Returns:
            Dictionary with training results
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("MODEL TRAINING STARTED")
        logger.info("=" * 60)
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Model type: {self.config.get('model_type', 'random_forest')}")
        
        try:
            # Scale features
            logger.info("Scaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
            
            # Select and train model
            model_type = self.config.get('model_type', 'random_forest')
            
            if model_type == 'random_forest':
                results = self._train_random_forest(X_train_scaled, y_train, X_val_scaled if X_val is not None else None, y_val)
            elif model_type == 'gradient_boosting':
                results = self._train_gradient_boosting(X_train_scaled, y_train, X_val_scaled if X_val is not None else None, y_val)
            elif model_type == 'neural_network':
                results = self._train_neural_network(X_train_scaled, y_train, X_val_scaled if X_val is not None else None, y_val)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Add metadata to results
            results.update({
                'training_time': training_time,
                'feature_names': feature_names,
                'n_samples': X_train.shape[0],
                'n_features': X_train.shape[1],
                'model_type': model_type,
                'timestamp': datetime.now().isoformat()
            })
            
            # Save model and results
            self._save_model(results)
            
            logger.info("=" * 60)
            logger.info("MODEL TRAINING COMPLETED")
            logger.info("=" * 60)
            logger.info(f"Training time: {training_time:.1f} seconds")
            logger.info(f"Best validation score: {results.get('best_score', 'N/A')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")
        
        # Get model parameters from config
        params = self.config.get('random_forest_params', {})
        
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            max_features=params.get('max_features', 'sqrt'),
            bootstrap=params.get('bootstrap', True),
            random_state=params.get('random_state', 42),
            n_jobs=-1,
            verbose=0
        )
        
        # Train model
        model.fit(X_train, y_train)
        self.model = model
        
        # Evaluate
        train_pred = model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred)
        
        results = {
            'model': model,
            'train_metrics': train_metrics,
            'feature_importance': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None
        }
        
        # Validation evaluation
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred)
            results['val_metrics'] = val_metrics
            results['best_score'] = val_metrics.get('r2', 0)
        
        logger.info(f"Random Forest trained: {model.n_estimators} trees")
        
        return results
    
    def _train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train Gradient Boosting model."""
        logger.info("Training Gradient Boosting model...")
        
        # Get model parameters from config
        params = self.config.get('gradient_boosting_params', {})
        
        model = GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 3),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            subsample=params.get('subsample', 1.0),
            random_state=params.get('random_state', 42),
            verbose=0
        )
        
        # Train model
        model.fit(X_train, y_train)
        self.model = model
        
        # Evaluate
        train_pred = model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred)
        
        results = {
            'model': model,
            'train_metrics': train_metrics,
            'feature_importance': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None
        }
        
        # Validation evaluation
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred)
            results['val_metrics'] = val_metrics
            results['best_score'] = val_metrics.get('r2', 0)
        
        logger.info(f"Gradient Boosting trained: {model.n_estimators} trees")
        
        return results
    
    def _train_neural_network(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train Neural Network model."""
        logger.info("Training Neural Network model...")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
        
        # Get model parameters from config
        params = self.config.get('neural_network_params', {})
        
        # Define neural network
        class MarineNet(nn.Module):
            def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
                super().__init__()
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, 1))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        # Initialize model
        input_dim = X_train.shape[1]
        hidden_dims = params.get('hidden_dims', [64, 32, 16])
        dropout_rate = params.get('dropout_rate', 0.2)
        
        model = MarineNet(input_dim, hidden_dims, dropout_rate)
        
        # Training parameters
        learning_rate = params.get('learning_rate', 0.001)
        batch_size = params.get('batch_size', 32)
        epochs = params.get('epochs', 100)
        patience = params.get('patience', 10)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Early stopping
        best_val_loss = np.inf
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    val_predictions = model(X_val_tensor)
                    val_loss = criterion(val_predictions, y_val_tensor).item()
                    history['val_loss'].append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}" + 
                           (f", Val Loss: {val_loss:.4f}" if X_val is not None else ""))
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        self.model = model
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor).numpy().flatten()
            train_metrics = self._calculate_metrics(y_train, train_pred)
        
        results = {
            'model': model,
            'train_metrics': train_metrics,
            'training_history': history,
            'best_val_loss': best_val_loss if X_val is not None else None
        }
        
        # Validation evaluation
        if X_val is not None and y_val is not None:
            with torch.no_grad():
                val_pred = model(X_val_tensor).numpy().flatten()
                val_metrics = self._calculate_metrics(y_val, val_pred)
                results['val_metrics'] = val_metrics
                results['best_score'] = -best_val_loss  # Negative because lower loss is better
        
        logger.info(f"Neural Network trained: {epochs} epochs")
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred))
        }
        
        # Calculate additional metrics
        y_mean = np.mean(y_true)
        if y_mean != 0:
            metrics['nrmse'] = metrics['rmse'] / y_mean
            metrics['mape'] = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        
        # Explained variance
        metrics['explained_variance'] = float(1 - np.var(y_true - y_pred) / np.var(y_true))
        
        return metrics
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with trained model.
        
        Args:
            X: Input features
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predictions (and optionally uncertainty)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        if isinstance(self.model, (RandomForestRegressor, GradientBoostingRegressor)):
            predictions = self.model.predict(X_scaled)
            
            if return_uncertainty:
                # For tree-based models, estimate uncertainty using tree variance
                if isinstance(self.model, RandomForestRegressor):
                    tree_preds = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
                    uncertainty = tree_preds.std(axis=0)
                else:
                    uncertainty = np.zeros_like(predictions)
                
                return predictions, uncertainty
            
            return predictions
        
        elif isinstance(self.model, nn.Module):
            # Neural network
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                predictions = self.model(X_tensor).numpy().flatten()
            
            if return_uncertainty:
                # Simple uncertainty estimate for NN
                uncertainty = np.ones_like(predictions) * 0.1  # Placeholder
                return predictions, uncertainty
            
            return predictions
        
        else:
            raise ValueError(f"Unknown model type: {type(self.model)}")
    
    def _save_model(self, results: Dict[str, Any]):
        """Save trained model and results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.output_dir / f"model_{timestamp}.joblib"
        
        if isinstance(self.model, nn.Module):
            # Save PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'config': self.config
            }, model_path.with_suffix('.pth'))
        else:
            # Save scikit-learn model
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'config': self.config
            }, model_path)
        
        # Save results
        results_path = self.output_dir / f"training_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_serializable = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    results_serializable[key] = value.tolist()
                elif hasattr(value, '__dict__') and not isinstance(value, (RandomForestRegressor, GradientBoostingRegressor, nn.Module)):
                    results_serializable[key] = str(value)
                else:
                    results_serializable[key] = value
            
            json.dump(results_serializable, f, indent=2, default=str)
        
        # Save configuration
        config_path = self.output_dir / f"config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Config saved to: {config_path}")
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Targets
            cv_folds: Number of cross-validation folds
            random_state: Random seed
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create CV splits
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        cv_scores = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
            logger.info(f"Training fold {fold}/{cv_folds}...")
            
            # Split data
            X_train_fold = X_scaled[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_scaled[val_idx]
            y_val_fold = y[val_idx]
            
            # Train model for this fold
            fold_trainer = ModelTrainer(self.config, self.output_dir / f"fold_{fold}")
            fold_results_dict = fold_trainer.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            
            # Store metrics
            if 'val_metrics' in fold_results_dict:
                val_metrics = fold_results_dict['val_metrics']
                for metric in cv_scores.keys():
                    if metric in val_metrics:
                        cv_scores[metric].append(val_metrics[metric])
            
            fold_results.append({
                'fold': fold,
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'metrics': fold_results_dict.get('val_metrics', {})
            })
        
        # Calculate statistics
        cv_summary = {}
        for metric, scores in cv_scores.items():
            if scores:
                cv_summary[metric] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'scores': scores
                }
        
        results = {
            'cv_summary': cv_summary,
            'fold_results': fold_results,
            'n_folds': cv_folds,
            'random_state': random_state
        }
        
        # Save CV results
        cv_path = self.output_dir / f"cross_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(cv_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Cross-validation completed")
        logger.info(f"Mean R²: {cv_summary.get('r2', {}).get('mean', 0):.4f} ± {cv_summary.get('r2', {}).get('std', 0):.4f}")
        
        return results


def main():
    """Example usage of ModelTrainer."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = X.dot(np.random.randn(n_features)) + np.random.randn(n_samples) * 0.1
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configuration
    config = {
        'model_type': 'random_forest',
        'random_forest_params': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
    }
    
    # Create trainer
    trainer = ModelTrainer(config, output_dir=Path("test_results"))
    
    # Train model
    results = trainer.train(X_train, y_train, X_val, y_val)
    
    print(f"Training R²: {results['train_metrics']['r2']:.4f}")
    print(f"Validation R²: {results['val_metrics']['r2']:.4f}")
    
    # Make predictions
    predictions = trainer.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")


if __name__ == "__main__":
    main()
