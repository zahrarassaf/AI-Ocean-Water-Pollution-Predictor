# فایل model_trainer.py را با این نسخه جایگزین کن:

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
                           f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

# Import optional models with fallbacks
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not available")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("⚠️  LightGBM not available")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available")

logger = logging.getLogger(__name__)

class PollutionModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        
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
        """Initialize multiple ML models with fallbacks."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            )
        }
        
        # Add SVM only if we have enough samples per class
        models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=self.random_state,
            probability=True,
            class_weight='balanced'
        )
        
        # Add optional models if available
        if XGB_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        
        if LGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.05,
                random_state=self.random_state
            )
        
        if CATBOOST_AVAILABLE:
            models['catboost'] = CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=False
            )
        
        logger.info(f"Initialized {len(models)} models")
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
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                
                # Store results
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'predictions': y_pred
                }
                
                print(f"  {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                print(f"  ❌ {name}: Failed - {str(e)[:50]}...")
                continue
        
        # Select best model based on F1 score
        if results:
            self.best_model_name = max(results, 
                                     key=lambda x: results[x]['f1'])
            self.best_model = results[self.best_model_name]['model']
            logger.info(f"Best model: {self.best_model_name} (F1={results[self.best_model_name]['f1']:.4f})")
            print(f"\n🏆 Best model: {self.best_model_name}")
        else:
            logger.warning("No models trained successfully")
            print("❌ No models were trained successfully")
        
        self.models = results
        return results
    
    def evaluate_best_model(self, X_test, y_test) -> Dict:
        """Evaluate the best model on test set."""
        if self.best_model is None:
            raise ValueError("No model trained yet. Run train_models() first.")
        
        y_pred = self.best_model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=self.label_encoder.classes_,
                zero_division=0
            )
        }
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test F1 Score: {metrics['f1']:.4f}")
        
        return metrics
    
    def cross_validate(self, X, y, cv: int = 3) -> Dict:
        """Perform cross-validation on all models."""
        cv_results = {}
        models = self.initialize_models()
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, 
                                       scoring='f1_weighted', n_jobs=-1)
                cv_results[name] = {
                    'mean_f1': scores.mean(),
                    'std_f1': scores.std(),
                    'scores': scores
                }
                print(f"{name}: CV F1 = {scores.mean():.4f} ± {scores.std():.4f}")
            except Exception as e:
                print(f"{name}: CV failed - {str(e)[:50]}...")
                continue
        
        return cv_results
    
    def hyperparameter_tuning(self, X_train, y_train) -> Dict:
        """Perform hyperparameter tuning for best model."""
        if self.best_model is None:
            print("No best model found for tuning")
            return {}
        
        if self.best_model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
                'min_samples_split': [2, 5]
            }
            base_model = RandomForestClassifier(random_state=self.random_state, class_weight='balanced')
        
        elif self.best_model_name == 'xgboost' and XGB_AVAILABLE:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
            base_model = xgb.XGBClassifier(random_state=self.random_state, 
                                         use_label_encoder=False)
        
        elif self.best_model_name == 'logistic_regression':
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['liblinear']
            }
            base_model = LogisticRegression(max_iter=1000, random_state=self.random_state, class_weight='balanced')
        
        else:
            print(f"Hyperparameter tuning not implemented for {self.best_model_name}")
            return {}
        
        print(f"\n🔧 Tuning {self.best_model_name}...")
        try:
            grid_search = GridSearchCV(
                base_model, param_grid, 
                cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"✅ Best params: {grid_search.best_params_}")
            print(f"✅ Best score: {grid_search.best_score_:.4f}")
            
            self.best_model = grid_search.best_estimator_
            return grid_search.best_params_
        except Exception as e:
            print(f"❌ Tuning failed: {e}")
            return {}
    
    def save_model(self, model_path: str = "models/"):
        """Save the trained model and scaler."""
        from pathlib import Path
        
        if self.best_model is None:
            raise ValueError("No model to save. Train a model first.")
        
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
            'model_type': type(self.best_model).__name__,
            'feature_count': self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else 'unknown',
            'classes': self.label_encoder.classes_.tolist(),
            'random_state': self.random_state
        }
        
        joblib.dump(metadata, model_dir / "model_metadata.pkl")
        
        print(f"💾 Model saved to {model_dir}/")
        print(f"   - pollution_model.pkl")
        print(f"   - scaler.pkl")
        print(f"   - label_encoder.pkl")
        print(f"   - model_metadata.pkl")
        
        return str(model_dir)

# =============== DEMO EXECUTION ===============
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 OCEAN POLLUTION MODEL TRAINER - DEMO")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create BETTER sample data with balanced classes
        print("\n📊 Creating balanced sample data...")
        np.random.seed(42)
        n_samples = 1200  # More samples
        n_features = 5
        
        # Create balanced classes
        n_per_class = n_samples // 3  # 3 classes
        
        # Class 1: LOW (chlorophyll <= 1)
        X_low = np.random.randn(n_per_class, n_features) * 0.5 + 0.5
        y_low = ['LOW'] * n_per_class
        
        # Class 2: MEDIUM (chlorophyll 1-5)
        X_medium = np.random.randn(n_per_class, n_features) * 0.8 + 2.0
        y_medium = ['MEDIUM'] * n_per_class
        
        # Class 3: HIGH (chlorophyll > 5)
        X_high = np.random.randn(n_per_class, n_features) * 1.0 + 4.0
        y_high = ['HIGH'] * n_per_class
        
        # Combine
        X = np.vstack([X_low, X_medium, X_high])
        y = np.array(y_low + y_medium + y_high)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        print(f"   Total samples: {len(X)}")
        print(f"   Features: {n_features}")
        print(f"   Classes: {np.unique(y)}")
        
        # Show class distribution
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"   {cls}: {count} samples ({count/len(y)*100:.1f}%)")
        
        # Split data WITHOUT stratify for demo
        print("\n📈 Splitting data...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f"   Train: {X_train.shape[0]} samples")
        print(f"   Validation: {X_val.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")
        
        # Show test distribution
        print("\n📊 Test set class distribution:")
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        for cls, count in zip(unique_test, counts_test):
            print(f"   {cls}: {count} samples")
        
        # Initialize trainer
        print("\n🤖 Initializing model trainer...")
        trainer = PollutionModelTrainer(random_state=42)
        
        # Prepare data
        print("\n🔧 Preparing data...")
        X_train_scaled, X_val_scaled, X_test_scaled = trainer.prepare_features(
            X_train, X_val, X_test
        )
        
        y_train_encoded, y_val_encoded, y_test_encoded = trainer.prepare_target(
            y_train, y_val, y_test
        )
        
        # Train models
        print("\n🚀 Training models...")
        results = trainer.train_models(
            X_train_scaled, y_train_encoded,
            X_val_scaled, y_val_encoded
        )
        
        if trainer.best_model:
            # Evaluate
            print("\n📊 Evaluating best model...")
            metrics = trainer.evaluate_best_model(X_test_scaled, y_test_encoded)
            
            print(f"\n✅ Final Results:")
            print(f"   Best Model: {trainer.best_model_name}")
            print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Test F1 Score: {metrics['f1']:.4f}")
            
            # Show classification report
            print("\n📋 Classification Report:")
            print(metrics['classification_report'])
            
            # Cross-validation
            print("\n🔍 Cross-validation...")
            X_combined = np.vstack([X_train_scaled, X_val_scaled])
            y_combined = np.concatenate([y_train_encoded, y_val_encoded])
            cv_results = trainer.cross_validate(X_combined, y_combined, cv=3)
            
            # Save model
            print("\n💾 Saving model...")
            save_path = trainer.save_model()
            
            print("\n" + "=" * 60)
            print("🎉 DEMO COMPLETED SUCCESSFULLY!")
            print(f"Model saved to: {save_path}")
            
            # Test loading the model
            print("\n🧪 Testing model loading...")
            try:
                loaded_model = joblib.load('models/pollution_model.pkl')
                loaded_scaler = joblib.load('models/scaler.pkl')
                loaded_encoder = joblib.load('models/label_encoder.pkl')
                print("✅ Model loaded successfully!")
                
                # Make a prediction
                sample = X_test_scaled[0:1]
                prediction = loaded_model.predict(sample)
                decoded = loaded_encoder.inverse_transform(prediction)
                print(f"   Sample prediction: {decoded[0]}")
                
            except Exception as e:
                print(f"❌ Error loading model: {e}")
            
        else:
            print("\n❌ No model was trained successfully")
            
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
