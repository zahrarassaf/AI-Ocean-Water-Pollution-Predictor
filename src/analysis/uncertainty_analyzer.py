"""
Uncertainty analysis for primary productivity estimates using ensemble methods
"""

import numpy as np
import xarray as xr
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from ..utils.config_loader import ConfigLoader

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class PrimaryProductivityUncertaintyAnalyzer:
    """Analyze uncertainty in primary productivity estimates"""
    
    def __init__(self, dataset: xr.Dataset, config: Optional[Dict] = None):
        """
        Initialize uncertainty analyzer
        
        Parameters
        ----------
        dataset : xr.Dataset
            Environmental dataset
        config : dict, optional
            Configuration dictionary
        """
        self.dataset = dataset
        self.config = config or ConfigLoader().get('models', {})
        
        # Results storage
        self.models = {}
        self.results = {}
        self.uncertainty_maps = None
        
        # Check for required variables
        self._validate_dataset()
    
    def _validate_dataset(self) -> None:
        """Validate dataset has required variables"""
        required_vars = ['pp']  # Primary productivity is required
        missing_vars = [var for var in required_vars if var not in self.dataset.data_vars]
        
        if missing_vars:
            raise ValueError(f"Dataset missing required variables: {missing_vars}")
    
    def prepare_features(self, feature_vars: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and target vector
        
        Parameters
        ----------
        feature_vars : list, optional
            List of feature variable names
            
        Returns
        -------
        tuple
            (X, y) feature matrix and target vector
        """
        if feature_vars is None:
            # Default feature variables
            feature_vars = [
                'chl', 'kd490', 'zsd', 'bbp', 'cdm',
                'current_speed', 'current_direction'
            ]
        
        # Filter available variables
        available_vars = [var for var in feature_vars if var in self.dataset.data_vars]
        logger.info(f"Using features: {available_vars}")
        
        # Collect feature data
        X_list = []
        for var in available_vars:
            data = self.dataset[var].values.flatten()
            X_list.append(data.reshape(-1, 1))
        
        # Target variable
        y = self.dataset['pp'].values.flatten()
        
        # Stack features
        X = np.hstack(X_list) if X_list else np.ones((len(y), 1))
        
        # Remove NaN values
        valid_mask = ~np.isnan(y)
        if X.shape[1] > 1:
            valid_mask = valid_mask & ~np.any(np.isnan(X), axis=1)
        
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        logger.info(f"Prepared data: {len(y_clean)} samples, {X_clean.shape[1]} features")
        
        return X_clean, y_clean
    
    def train_random_forest_ensemble(self, X: np.ndarray, y: np.ndarray,
                                   n_models: int = 10) -> Dict:
        """
        Train ensemble of Random Forest models
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        n_models : int
            Number of models in ensemble
            
        Returns
        -------
        dict
            Training results
        """
        logger.info(f"Training Random Forest ensemble with {n_models} models")
        
        config = self.config.get('random_forest', {})
        models = []
        predictions = []
        
        for i in range(n_models):
            # Create model with varied parameters
            model = RandomForestRegressor(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', None),
                min_samples_split=config.get('min_samples_split', 2) + i % 3,
                min_samples_leaf=config.get('min_samples_leaf', 1) + i % 2,
                random_state=config.get('random_state', 42) + i,
                n_jobs=config.get('n_jobs', -1)
            )
            
            # Train model
            model.fit(X, y)
            models.append(model)
            
            # Collect predictions from all trees
            tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
            predictions.append(tree_preds)
        
        # Combine predictions
        all_predictions = np.vstack(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)
        
        # Store results
        results = {
            'models': models,
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'predictions': all_predictions,
            'feature_importance': self._calculate_feature_importance(models, X, y)
        }
        
        self.models['random_forest'] = results
        return results
    
    def train_gaussian_process(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train Gaussian Process model
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
            
        Returns
        -------
        dict
            Training results
        """
        logger.info("Training Gaussian Process model")
        
        config = self.config.get('gaussian_process', {})
        
        # Define kernel
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        
        # Create and train model
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=config.get('n_restarts_optimizer', 10),
            random_state=config.get('random_state', 42),
            normalize_y=True
        )
        
        model.fit(X, y)
        
        # Get predictions with uncertainty
        y_pred, y_std = model.predict(X, return_std=True)
        
        results = {
            'model': model,
            'mean_prediction': y_pred,
            'std_prediction': y_std,
            'kernel': str(model.kernel_),
            'log_marginal_likelihood': model.log_marginal_likelihood()
        }
        
        self.models['gaussian_process'] = results
        return results
    
    class NeuralNetwork(nn.Module):
        """Neural network for uncertainty estimation"""
        
        def __init__(self, input_dim: int, hidden_dims: List[int] = None,
                    dropout_rate: float = 0.2):
            super().__init__()
            
            if hidden_dims is None:
                hidden_dims = [64, 32, 16]
            
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
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)
    
    def train_neural_network_ensemble(self, X: np.ndarray, y: np.ndarray,
                                     n_models: int = 5) -> Dict:
        """
        Train ensemble of neural networks
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        n_models : int
            Number of models in ensemble
            
        Returns
        -------
        dict
            Training results
        """
        logger.info(f"Training Neural Network ensemble with {n_models} models")
        
        config = self.config.get('neural_network', {})
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        models = []
        predictions = []
        
        for i in range(n_models):
            # Create model
            model = self.NeuralNetwork(
                input_dim=X.shape[1],
                hidden_dims=config.get('hidden_layers', [64, 32, 16]),
                dropout_rate=config.get('dropout_rate', 0.2)
            )
            
            # Define optimizer and loss
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.get('learning_rate', 0.001)
            )
            criterion = nn.MSELoss()
            
            # Training loop
            n_epochs = config.get('epochs', 100)
            batch_size = config.get('batch_size', 32)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            model.train()
            for epoch in range(n_epochs):
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Store model
            models.append(model)
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                preds = model(X_tensor).numpy().flatten()
                predictions.append(preds)
        
        # Combine predictions
        predictions_array = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions_array, axis=0)
        std_pred = np.std(predictions_array, axis=0)
        
        results = {
            'models': models,
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'predictions': predictions_array
        }
        
        self.models['neural_network'] = results
        return results
    
    def calculate_bootstrap_uncertainty(self, X: np.ndarray, y: np.ndarray,
                                      n_bootstrap: int = 100) -> Dict:
        """
        Calculate uncertainty using bootstrap sampling
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        n_bootstrap : int
            Number of bootstrap samples
            
        Returns
        -------
        dict
            Bootstrap results
        """
        logger.info(f"Calculating bootstrap uncertainty with {n_bootstrap} iterations")
        
        n_samples = len(y)
        predictions = np.zeros((n_bootstrap, n_samples))
        
        for i in range(n_bootstrap):
            # Create bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=50,
                random_state=i,
                n_jobs=-1
            )
            model.fit(X_boot, y_boot)
            
            # Predict on original data
            predictions[i] = model.predict(X)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        results = {
            'mean': mean_pred,
            'std': std_pred,
            'predictions': predictions
        }
        
        self.models['bootstrap'] = results
        return results
    
    def cross_validation_uncertainty(self, X: np.ndarray, y: np.ndarray,
                                   n_folds: int = 5) -> Dict:
        """
        Calculate uncertainty using cross-validation
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        n_folds : int
            Number of CV folds
            
        Returns
        -------
        dict
            CV results
        """
        logger.info(f"Calculating {n_folds}-fold cross-validation uncertainty")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        oof_predictions = np.zeros_like(y)
        fold_uncertainties = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=fold,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            oof_predictions[val_idx] = y_pred
            
            # Calculate uncertainty from tree predictions
            tree_preds = np.array([tree.predict(X_val) for tree in model.estimators_])
            fold_uncertainty = np.std(tree_preds, axis=0)
            fold_uncertainties.append(fold_uncertainty)
        
        # Combine fold uncertainties
        oof_uncertainty = np.concatenate(fold_uncertainties)
        
        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(y, oof_predictions))
        mae = mean_absolute_error(y, oof_predictions)
        r2 = r2_score(y, oof_predictions)
        
        results = {
            'predictions': oof_predictions,
            'uncertainty': oof_uncertainty,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        self.models['cross_validation'] = results
        return results
    
    def run_ensemble_analysis(self, test_size: float = 0.2) -> Dict:
        """
        Run complete ensemble uncertainty analysis
        
        Parameters
        ----------
        test_size : float
            Proportion of data for testing
            
        Returns
        -------
        dict
            Complete analysis results
        """
        logger.info("Starting ensemble uncertainty analysis")
        
        # Prepare data
        X, y = self.prepare_features()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Run all uncertainty methods
        results = {}
        
        # 1. Random Forest Ensemble
        rf_results = self.train_random_forest_ensemble(X_train_scaled, y_train)
        results['random_forest'] = rf_results
        
        # 2. Gaussian Process
        try:
            gp_results = self.train_gaussian_process(X_train_scaled, y_train)
            results['gaussian_process'] = gp_results
        except Exception as e:
            logger.warning(f"Gaussian Process failed: {e}")
        
        # 3. Neural Network Ensemble
        nn_results = self.train_neural_network_ensemble(X_train_scaled, y_train)
        results['neural_network'] = nn_results
        
        # 4. Bootstrap
        bootstrap_results = self.calculate_bootstrap_uncertainty(X_train_scaled, y_train)
        results['bootstrap'] = bootstrap_results
        
        # 5. Cross-Validation
        cv_results = self.cross_validation_uncertainty(X_train_scaled, y_train)
        results['cross_validation'] = cv_results
        
        # Combine uncertainties
        ensemble_results = self._combine_uncertainties(results)
        results['ensemble'] = ensemble_results
        
        # Evaluate on test set
        test_results = self.evaluate_on_test_set(X_test_scaled, y_test, results)
        results['test_evaluation'] = test_results
        
        # Calculate statistics
        stats = self._calculate_statistics(results)
        results['statistics'] = stats
        
        # Map to spatial grid
        spatial_uncertainty = self._map_to_spatial_grid(results, scaler)
        results['spatial_uncertainty'] = spatial_uncertainty
        
        self.results = results
        return results
    
    def _combine_uncertainties(self, results: Dict) -> Dict:
        """Combine uncertainties from different methods"""
        predictions_list = []
        uncertainties_list = []
        
        for method, method_results in results.items():
            if method == 'ensemble':
                continue
            
            # Extract predictions
            if 'mean_prediction' in method_results:
                predictions_list.append(method_results['mean_prediction'])
            elif 'mean' in method_results:
                predictions_list.append(method_results['mean'])
            elif 'predictions' in method_results and method_results['predictions'].ndim == 1:
                predictions_list.append(method_results['predictions'])
            
            # Extract uncertainties
            if 'std_prediction' in method_results:
                uncertainties_list.append(method_results['std_prediction'])
            elif 'std' in method_results:
                uncertainties_list.append(method_results['std'])
            elif 'uncertainty' in method_results:
                uncertainties_list.append(method_results['uncertainty'])
        
        # Calculate ensemble statistics
        if predictions_list:
            predictions_array = np.array(predictions_list)
            ensemble_mean = np.mean(predictions_array, axis=0)
            
            # Decompose uncertainty
            if uncertainties_list:
                uncertainties_array = np.array(uncertainties_list)
                aleatoric = np.mean(uncertainties_array, axis=0)
                epistemic = np.std(predictions_array, axis=0)
                total = np.sqrt(aleatoric**2 + epistemic**2)
            else:
                aleatoric = np.std(predictions_array, axis=0)
                epistemic = np.zeros_like(aleatoric)
                total = aleatoric
            
            relative = total / (np.abs(ensemble_mean) + 1e-10)
            
            return {
                'mean': ensemble_mean,
                'aleatoric_uncertainty': aleatoric,
                'epistemic_uncertainty': epistemic,
                'total_uncertainty': total,
                'relative_uncertainty': relative
            }
        
        return {}
    
    def evaluate_on_test_set(self, X_test: np.ndarray, y_test: np.ndarray,
                           results: Dict) -> Dict:
        """Evaluate ensemble on test set"""
        predictions_list = []
        
        for method, method_results in results.items():
            if method == 'ensemble':
                continue
            
            if 'models' in method_results:
                models = method_results['models']
                
                if isinstance(models, list):
                    # Average predictions from ensemble
                    method_preds = np.mean([
                        self._predict_model(m, X_test, method)
                        for m in models
                    ], axis=0)
                    predictions_list.append(method_preds)
                
                elif hasattr(models, 'predict'):
                    # Single model
                    predictions_list.append(models.predict(X_test))
        
        if predictions_list:
            ensemble_pred = np.mean(predictions_list, axis=0)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            mae = mean_absolute_error(y_test, ensemble_pred)
            r2 = r2_score(y_test, ensemble_pred)
            
            return {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': ensemble_pred,
                'true_values': y_test
            }
        
        return {}
    
    def _predict_model(self, model, X: np.ndarray, method: str) -> np.ndarray:
        """Make predictions from a model"""
        if method == 'neural_network':
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                preds = model(X_tensor).numpy().flatten()
            return preds
        else:
            return model.predict(X)
    
    def _calculate_feature_importance(self, models: List, X: np.ndarray, y: np.ndarray) -> Dict:
        """Calculate feature importance from ensemble"""
        if not models:
            return {}
        
        # Calculate permutation importance
        importance_scores = []
        for model in models[:5]:  # Use first 5 models for speed
            result = permutation_importance(
                model, X, y,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            importance_scores.append(result.importances_mean)
        
        if importance_scores:
            mean_importance = np.mean(importance_scores, axis=0)
            std_importance = np.std(importance_scores, axis=0)
            
            # Sort features by importance
            sorted_idx = np.argsort(mean_importance)[::-1]
            
            return {
                'mean_importance': mean_importance,
                'std_importance': std_importance,
                'sorted_indices': sorted_idx
            }
        
        return {}
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """Calculate comprehensive statistics"""
        stats = {}
        
        if 'ensemble' in results:
            ensemble = results['ensemble']
            
            for key in ['mean', 'aleatoric_uncertainty', 'epistemic_uncertainty', 'total_uncertainty']:
                if key in ensemble:
                    data = ensemble[key]
                    stats[key] = {
                        'mean': float(np.nanmean(data)),
                        'std': float(np.nanstd(data)),
                        'min': float(np.nanmin(data)),
                        'max': float(np.nanmax(data)),
                        'median': float(np.nanmedian(data))
                    }
        
        if 'test_evaluation' in results:
            test_eval = results['test_evaluation']
            stats['test_performance'] = {
                'r2': test_eval.get('r2', np.nan),
                'rmse': test_eval.get('rmse', np.nan),
                'mae': test_eval.get('mae', np.nan)
            }
        
        return stats
    
    def _map_to_spatial_grid(self, results: Dict, scaler: StandardScaler) -> xr.Dataset:
        """Map uncertainty results to spatial grid"""
        if 'ensemble' not in results:
            raise ValueError("Ensemble results not found")
        
        ensemble = results['ensemble']
        
        # Get original grid information
        pp_data = self.dataset['pp'].values
        n_time, n_lat, n_lon = pp_data.shape
        
        # Prepare features for entire grid
        X_full, _ = self.prepare_features()
        X_full_scaled = scaler.transform(X_full)
        
        # Create full array for each metric
        uncertainty_maps = {}
        
        for metric in ['mean', 'aleatoric_uncertainty', 'epistemic_uncertainty', 'total_uncertainty']:
            if metric in ensemble:
                # Predict for entire grid
                metric_data = ensemble[metric]
                
                # Reshape to match grid (simplified - in practice would need spatial prediction)
                if len(metric_data) == X_full_scaled.shape[0]:
                    # Create full spatial array
                    full_array = np.full(pp_data.shape[1:], np.nan)
                    
                    # For simplicity, take mean across time
                    # In actual implementation, would need proper spatial-temporal mapping
                    uncertainty_maps[metric] = full_array
        
        # Create dataset
        coords = {
            'lat': self.dataset.lat.values,
            'lon': self.dataset.lon.values
        }
        
        data_vars = {}
        for metric, data in uncertainty_maps.items():
            if data is not None:
                data_vars[metric] = (['lat', 'lon'], data)
        
        uncertainty_ds = xr.Dataset(data_vars, coords=coords)
        
        return uncertainty_ds
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """Save analysis results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save spatial uncertainty
        if self.uncertainty_maps is not None:
            nc_file = output_dir / "spatial_uncertainty.nc"
            self.uncertainty_maps.to_netcdf(nc_file)
        
        # Save statistics
        if 'statistics' in self.results:
            import json
            stats_file = output_dir / "uncertainty_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(self.results['statistics'], f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
