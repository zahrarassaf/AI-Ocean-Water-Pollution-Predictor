# File: final_test_fixed.py
import pandas as pd
import numpy as np
import joblib
import os
import glob
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def get_latest_processed_data():
    """Find the latest processed data file"""
    processed_files = glob.glob('data/processed/marine_pollution_prediction_*/processed_data.joblib')
    if not processed_files:
        raise FileNotFoundError("No processed data files found")
    
    # Sort by modification time (newest first)
    processed_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = processed_files[0]
    
    if not os.path.exists(latest_file):
        raise FileNotFoundError(f"Processed data file not found: {latest_file}")
    
    return latest_file

def train_final_model():
    """Train final model with optimized parameters"""
    print("=" * 60)
    print("FINAL MODEL TRAINING")
    print("=" * 60)
    
    try:
        # Get the latest processed data
        data_path = get_latest_processed_data()
        print(f"Loading data from: {data_path}")
        
        # Load processed data
        data = joblib.load(data_path)
        
        print("\nAvailable keys in data:")
        for key in data.keys():
            print(f"  {key}: {type(data[key])}")
        
        # Check if data has 'splits' key (new format)
        if 'splits' in data:
            print("\nDetected new data format (using 'splits' key)")
            splits = data['splits']
            
            # Extract data from splits
            X_train = splits['X_train']
            X_val = splits.get('X_val', None)
            X_test = splits['X_test']
            y_train = splits['y_train']
            y_val = splits.get('y_val', None)
            y_test = splits['y_test']
            
            # If no validation set, create one
            if X_val is None:
                print("No validation set found, using 20% of training for validation")
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
            
            feature_names = data.get('feature_names', [f'feature_{i}' for i in range(X_train.shape[1])])
            
        else:
            # Old format
            print("\nDetected old data format")
            X_train = data['X_train']
            X_val = data['X_val']
            X_test = data['X_test']
            y_train = data['y_train']
            y_val = data['y_val']
            y_test = data['y_test']
            feature_names = data['feature_names']
        
        print(f"\nX_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"Number of features: {len(feature_names)}")
        print(f"First 10 feature names: {feature_names[:10]}")
        
        # Combine train and validation sets for final training
        print("\nCombining train and validation sets for final training...")
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        print(f"Full training set: X shape={X_train_full.shape}, y shape={y_train_full.shape}")
        
        # Scale features
        print("\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled = scaler.transform(X_test)
        
        print("\nTraining Random Forest model...")
        
        # Train Random Forest with optimized parameters
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        rf_model.fit(X_train_scaled, y_train_full)
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred = rf_model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE")
        print("=" * 60)
        print(f"Test Set Size: {len(y_test)} samples")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Also evaluate on train set for comparison
        y_train_pred = rf_model.predict(scaler.transform(X_train))
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        print(f"\nTraining Set Performance:")
        print(f"  Training RMSE: {train_rmse:.4f}")
        print(f"  Training R²: {train_r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 most important features:")
        for i, row in feature_importance.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"models/final_model_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        model_data = {
            'model': rf_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': {
                'test_rmse': rmse,
                'test_mae': mae,
                'test_r2': r2,
                'train_rmse': train_rmse,
                'train_r2': train_r2
            },
            'training_date': timestamp,
            'data_source': data_path,
            'config': data.get('config', {})
        }
        
        model_path = os.path.join(model_dir, 'final_model.joblib')
        joblib.dump(model_data, model_path, compress=3)
        
        # Save feature importance
        feature_importance_path = os.path.join(model_dir, 'feature_importance.csv')
        feature_importance.to_csv(feature_importance_path, index=False)
        
        # Save predictions for analysis
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'error': y_test - y_pred,
            'absolute_error': np.abs(y_test - y_pred)
        })
        
        predictions_path = os.path.join(model_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        
        # Calculate additional statistics
        error_stats = predictions_df['absolute_error'].describe()
        
        print(f"\nError Statistics:")
        print(f"  Mean Absolute Error: {error_stats['mean']:.4f}")
        print(f"  Std of Errors: {error_stats['std']:.4f}")
        print(f"  Min Error: {error_stats['min']:.4f}")
        print(f"  25th Percentile: {error_stats['25%']:.4f}")
        print(f"  50th Percentile (Median): {error_stats['50%']:.4f}")
        print(f"  75th Percentile: {error_stats['75%']:.4f}")
        print(f"  Max Error: {error_stats['max']:.4f}")
        
        print(f"\nModel saved to: {model_path}")
        print(f"Feature importance saved to: {feature_importance_path}")
        print(f"Predictions saved to: {predictions_path}")
        
        # Create summary report
        report_path = os.path.join(model_dir, 'training_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("FINAL MODEL TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Training Date: {timestamp}\n")
            f.write(f"Data Source: {data_path}\n\n")
            f.write("DATA INFORMATION:\n")
            f.write(f"  Total training samples: {X_train_full.shape[0]}\n")
            f.write(f"  Test samples: {X_test.shape[0]}\n")
            f.write(f"  Number of features: {len(feature_names)}\n\n")
            f.write("MODEL PARAMETERS:\n")
            f.write(f"  Model: Random Forest Regressor\n")
            f.write(f"  n_estimators: {rf_model.n_estimators}\n")
            f.write(f"  max_depth: {rf_model.max_depth}\n")
            f.write(f"  min_samples_split: {rf_model.min_samples_split}\n")
            f.write(f"  min_samples_leaf: {rf_model.min_samples_leaf}\n")
            f.write(f"  max_features: {rf_model.max_features}\n\n")
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"  Test RMSE: {rmse:.4f}\n")
            f.write(f"  Test MAE: {mae:.4f}\n")
            f.write(f"  Test R² Score: {r2:.4f}\n")
            f.write(f"  Training RMSE: {train_rmse:.4f}\n")
            f.write(f"  Training R²: {train_r2:.4f}\n\n")
            f.write("ERROR STATISTICS:\n")
            for stat, value in error_stats.items():
                f.write(f"  {stat}: {value:.4f}\n")
            f.write("\nTOP 15 FEATURES:\n")
            for i, row in feature_importance.head(15).iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
            f.write("\nALL FEATURES:\n")
            for i, row in feature_importance.iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
        
        print(f"\nTraining report saved to: {report_path}")
        
        # Create a quick visualization of predictions vs actual
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test[:1000], y_pred[:1000], alpha=0.5, s=10)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Predictions vs Actual (R²={r2:.3f})')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(model_dir, 'predictions_vs_actual.png')
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"Prediction plot saved to: {plot_path}")
        except ImportError:
            print("Matplotlib not available for plotting")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return model_data
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nAvailable processed data files:")
        processed_files = glob.glob('data/processed/marine_pollution_prediction_*/processed_data.joblib')
        for file in processed_files:
            print(f"  {file}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    
    # Train the model
    train_final_model()
