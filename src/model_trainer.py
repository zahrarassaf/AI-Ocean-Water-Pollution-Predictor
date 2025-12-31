import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

class PollutionModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        
    def prepare_features(self, X_train, X_val, X_test):
        """Scale and prepare features."""
        # Scale numerical features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def prepare_target(self, y_train, y_val, y_test):
        """Encode target labels."""
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        return y_train_encoded, y_val_encoded, y_test_encoded
    
    def initialize_models(self) -> Dict:
        """Initialize multiple ML models."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.05,
                random_state=self.random_state
            ),
            'catboost': CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=False
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=self.random_state,
                probability=True
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            )
        }
        
        return models
    
    def train_models(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train all models and select the best one."""
        models = self.initialize_models()
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted')
                recall = recall_score(y_val, y_pred, average='weighted')
                f1 = f1_score(y_val, y_pred, average='weighted')
                
                # Store results
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Select best model based on F1 score
        if results:
            self.best_model_name = max(results, 
                                     key=lambda x: results[x]['f1'])
            self.best_model = results[self.best_model_name]['model']
            logger.info(f"Best model: {self.best_model_name}")
        
        self.models = results
        return results
    
    def evaluate_best_model(self, X_test, y_test) -> Dict:
        """Evaluate the best model on test set."""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=self.label_encoder.classes_
            )
        }
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test F1 Score: {metrics['f1']:.4f}")
        
        return metrics
    
    def cross_validate(self, X, y, cv: int = 5) -> Dict:
        """Perform cross-validation on all models."""
        cv_results = {}
        
        for name, model in self.initialize_models().items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, 
                                       scoring='f1_weighted', n_jobs=-1)
                cv_results[name] = {
                    'mean_f1': scores.mean(),
                    'std_f1': scores.std(),
                    'scores': scores
                }
                logger.info(f"{name} CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
            except:
                continue
        
        return cv_results
    
    def hyperparameter_tuning(self, X_train, y_train) -> Dict:
        """Perform hyperparameter tuning for best model."""
        if self.best_model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestClassifier(random_state=self.random_state)
        
        elif self.best_model_name == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3]
            }
            base_model = xgb.XGBClassifier(random_state=self.random_state, 
                                         use_label_encoder=False)
        
        else:
            return {}
        
        grid_search = GridSearchCV(
            base_model, param_grid, 
            cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        return grid_search.best_params_
    
    def save_model(self, model_path: str = "models/"):
        """Save the trained model and scaler."""
        from pathlib import Path
        
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        joblib.dump(self.best_model, model_dir / "pollution_model.pkl")
        
        # Save scaler and encoder
        joblib.dump(self.scaler, model_dir / "scaler.pkl")
        joblib.dump(self.label_encoder, model_dir / "label_encoder.pkl")
        
        # Save model metadata
        metadata = {
            'best_model': self.best_model_name,
            'features': self.scaler.feature_names_in_.tolist() 
                        if hasattr(self.scaler, 'feature_names_in_') else [],
            'classes': self.label_encoder.classes_.tolist()
        }
        
        joblib.dump(metadata, model_dir / "model_metadata.pkl")
        
        logger.info(f"Model saved to {model_dir}")
