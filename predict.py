"""
Ocean Pollution Prediction System
ML model for predicting ocean pollution levels from satellite data
"""

import joblib
import pandas as pd
import numpy as np
import os
import sys
import warnings
from datetime import datetime
import glob
import traceback
import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings('ignore')


class OceanPollutionPredictor:
    """Machine learning model for ocean pollution prediction"""
    
    def __init__(self, data_path="data/raw"):
        self.model = None
        self.scaler = None
        self.features = []
        self.classes = ['LOW', 'MEDIUM', 'HIGH']
        self.data_path = data_path
        self.datasets = {}
        self.feature_statistics = {}
    
    def _remove_duplicates(self, features_list):
        """Remove duplicate features while preserving order"""
        unique_features = []
        seen = set()
        for feat in features_list:
            if feat not in seen:
                unique_features.append(feat)
                seen.add(feat)
        return unique_features
    
    def _calculate_feature_statistics(self, X, feature_names):
        """Calculate statistics for each feature"""
        for i, feat in enumerate(feature_names):
            if feat not in self.feature_statistics:
                self.feature_statistics[feat] = {
                    'mean': np.mean(X[:, i]) if X.shape[0] > 0 else 0,
                    'std': np.std(X[:, i]) if X.shape[0] > 0 else 1,
                    'min': np.min(X[:, i]) if X.shape[0] > 0 else 0,
                    'max': np.max(X[:, i]) if X.shape[0] > 0 else 1,
                    'median': np.median(X[:, i]) if X.shape[0] > 0 else 0
                }
    
    def load_datasets(self):
        """Load all NetCDF files from data directory"""
        try:
            nc_files = glob.glob(os.path.join(self.data_path, "*.nc"))
            if not nc_files:
                print("No NetCDF files found in data/raw/")
                return False
            
            print(f"Found {len(nc_files)} NetCDF files:")
            for filepath in nc_files:
                filename = os.path.basename(filepath)
                try:
                    ds = xr.open_dataset(filepath)
                    self.datasets[filename] = ds
                    print(f"  Loaded: {filename} ({len(ds.data_vars)} variables)")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")
                    continue
            
            # Display what's in each file
            print("\nDataset contents:")
            for filename, ds in self.datasets.items():
                vars_list = list(ds.data_vars)
                print(f"  {filename}: {vars_list}")
            
            return bool(self.datasets)
            
        except Exception as e:
            print(f"Error in load_datasets: {e}")
            traceback.print_exc()
            return False
    
    def find_chlorophyll_dataset(self):
        """Find which dataset contains chlorophyll data"""
        chl_dataset = None
        chl_filename = None
        chl_variable = None
        
        for filename, ds in self.datasets.items():
            for var_name in ds.data_vars:
                var_upper = var_name.upper()
                # Look for chlorophyll-related variable names
                if any(chl_keyword in var_upper for chl_keyword in ['CHL', 'CHLOROPHYLL']):
                    chl_dataset = ds
                    chl_filename = filename
                    chl_variable = var_name
                    print(f"Found chlorophyll in: {filename} (variable: {var_name})")
                    break
            if chl_dataset:
                break
        
        if chl_dataset is None:
            print("\nWARNING: No chlorophyll data found!")
            print("Available variables across all files:")
            for filename, ds in self.datasets.items():
                print(f"  {filename}: {list(ds.data_vars)}")
        
        return chl_dataset, chl_filename, chl_variable
    
    def create_pollution_index(self, X, feature_names):
        """Create pollution index from multiple indicators"""
        pollution_score = np.zeros(X.shape[0])
        
        # Updated pollution indicators based on available data
        pollution_indicators = {
            'CHL': 0.30,
            'KD490': 0.25,
            'PP': 0.20,
            'CDM': 0.15,
            'BBP': 0.10
        }
        
        # Try to find available indicators
        available_indicators = {}
        for indicator, weight in pollution_indicators.items():
            # Check if exact match
            if indicator in feature_names:
                available_indicators[indicator] = weight
            else:
                # Check for similar names
                for feat in feature_names:
                    if indicator in feat.upper():
                        available_indicators[feat] = weight
                        break
        
        # If no exact matches found, use whatever is available
        if not available_indicators:
            for i, feat in enumerate(feature_names):
                available_indicators[feat] = 1.0 / len(feature_names)
        
        # Normalize and combine indicators
        total_weight = sum(available_indicators.values())
        if total_weight > 0:
            for indicator, weight in available_indicators.items():
                if indicator in feature_names:
                    idx = feature_names.index(indicator)
                    data = X[:, idx]
                    if np.max(data) > np.min(data):
                        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
                        pollution_score += norm_data * (weight / total_weight)
        
        return pollution_score
    
    def extract_features_from_datasets(self, sampled_points):
        """Extract features from all datasets for sampled points"""
        feature_data = {}
        feature_names = []
        added_features = set()
        
        # First pass: collect all available features
        all_features = []
        for filename, ds in self.datasets.items():
            for var_name in ds.data_vars:
                if var_name.lower() not in ['time', 'latitude', 'longitude', 'lat', 'lon']:
                    all_features.append((filename, var_name))
        
        print(f"\nTotal features available: {len(all_features)}")
        
        # Process each feature
        for filename, var_name in all_features:
            ds = self.datasets[filename]
            
            if var_name in added_features:
                continue
            
            try:
                var_data = ds[var_name].values
                if var_data.ndim != 3:
                    print(f"  Skipping {var_name}: Not 3D data")
                    continue
                
                var_values = []
                valid_count = 0
                
                for point in sampled_points:
                    try:
                        value = var_data[point['time_idx'], point['lat_idx'], point['lon_idx']]
                        if not np.isnan(value):
                            var_values.append(value)
                            valid_count += 1
                        else:
                            var_values.append(np.nan)
                    except (IndexError, ValueError):
                        var_values.append(np.nan)
                
                valid_ratio = valid_count / len(sampled_points)
                if valid_ratio > 0.3:  # Keep if more than 30% valid
                    var_array = np.array(var_values)
                    median_val = np.nanmedian(var_array)
                    var_array[np.isnan(var_array)] = median_val
                    
                    feature_data[var_name] = var_array
                    feature_names.append(var_name)
                    added_features.add(var_name)
                    print(f"  Added: {var_name} ({valid_ratio:.1%} valid)")
                else:
                    print(f"  Skipped: {var_name} (only {valid_ratio:.1%} valid)")
                    
            except Exception as e:
                print(f"  Error extracting {var_name}: {e}")
                continue
        
        return feature_data, feature_names
    
    def create_dataset(self, sample_limit=50000):
        """Create dataset from loaded NetCDF files"""
        try:
            # Find dataset with chlorophyll or use first available
            chl_ds, chl_filename, chl_variable = self.find_chlorophyll_dataset()
            
            if chl_ds is None:
                print("Using first available dataset as reference")
                chl_ds = list(self.datasets.values())[0]
                chl_filename = list(self.datasets.keys())[0]
                chl_variable = list(chl_ds.data_vars)[0]
            
            print(f"\nUsing reference dataset: {chl_filename}")
            print(f"Reference variable: {chl_variable}")
            
            # Get data from reference variable
            ref_data = chl_ds[chl_variable].values
            ref_flat = ref_data.flatten()
            valid_mask = ~np.isnan(ref_flat)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > sample_limit:
                sample_indices = np.random.choice(valid_indices, sample_limit, replace=False)
                print(f"Sampled to: {sample_limit} points")
            else:
                sample_indices = valid_indices
            
            if len(sample_indices) == 0:
                print("ERROR: No valid data points found!")
                return None, None, None
            
            # Calculate indices for 3D array
            time_size, lat_size, lon_size = ref_data.shape
            sampled_points = []
            
            for idx in sample_indices:
                time_idx = idx // (lat_size * lon_size)
                temp = idx % (lat_size * lon_size)
                lat_idx = temp // lon_size
                lon_idx = temp % lon_size
                
                sampled_points.append({
                    'time_idx': time_idx,
                    'lat_idx': lat_idx,
                    'lon_idx': lon_idx,
                    'ref_value': ref_flat[idx]
                })
            
            # Extract features from all datasets
            print("\nExtracting features from all datasets...")
            feature_data, feature_names = self.extract_features_from_datasets(sampled_points)
            
            if not feature_names:
                print("ERROR: No features extracted!")
                return None, None, None
            
            # Create feature matrix
            X_list = []
            final_features = []
            
            for feat_name in feature_names:
                if feat_name in feature_data:
                    X_list.append(feature_data[feat_name])
                    final_features.append(feat_name)
            
            X = np.column_stack(X_list)
            
            # Calculate statistics
            self._calculate_feature_statistics(X, final_features)
            
            # Create pollution index
            print("\nCreating pollution labels...")
            pollution_score = self.create_pollution_index(X, final_features)
            
            # Define thresholds
            low_threshold = np.percentile(pollution_score, 33)
            medium_threshold = np.percentile(pollution_score, 66)
            
            # Create labels
            y = np.zeros(len(pollution_score))
            y[pollution_score > medium_threshold] = 2
            y[(pollution_score > low_threshold) & (pollution_score <= medium_threshold)] = 1
            
            # Print statistics
            print(f"\nPollution Label Distribution:")
            print(f"  LOW: {np.sum(y == 0):,} samples")
            print(f"  MEDIUM: {np.sum(y == 1):,} samples")
            print(f"  HIGH: {np.sum(y == 2):,} samples")
            
            # Calculate correlation with reference variable
            if chl_variable in final_features:
                ref_idx = final_features.index(chl_variable)
                ref_values = X[:, ref_idx]
                ref_corr = np.corrcoef(ref_values, y)[0, 1]
                print(f"  Correlation with {chl_variable}: {ref_corr:.3f}")
            
            # Close datasets
            for ds in self.datasets.values():
                ds.close()
            
            # Remove duplicates
            final_features = self._remove_duplicates(final_features)
            
            print(f"\nDataset created successfully:")
            print(f"  Samples: {X.shape[0]:,}")
            print(f"  Features: {len(final_features)}")
            print(f"  Pollution score range: {pollution_score.min():.3f} to {pollution_score.max():.3f}")
            
            return X, y, final_features
            
        except Exception as e:
            print(f"Error creating dataset: {e}")
            traceback.print_exc()
            return None, None, None
    
    def train_model(self):
        """Train Random Forest model"""
        try:
            print("\n" + "="*60)
            print("TRAINING OCEAN POLLUTION PREDICTION MODEL")
            print("="*60)
            
            # Load datasets
            if not self.load_datasets():
                return False
            
            # Create dataset
            X, y, feature_names = self.create_dataset()
            
            if X is None:
                return False
            
            self.features = self._remove_duplicates(feature_names)
            
            # Print class distribution
            print(f"\nClass distribution:")
            unique, counts = np.unique(y, return_counts=True)
            for cls, count in zip(unique, counts):
                print(f"  Class {self.classes[int(cls)]}: {count:,} samples ({count/len(y)*100:.1f}%)")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"\nData split:")
            print(f"  Training samples: {X_train.shape[0]:,}")
            print(f"  Testing samples: {X_test.shape[0]:,}")
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Determine optimal parameters
            n_estimators = min(200, X_train.shape[0] // 100)
            n_estimators = max(50, n_estimators)
            
            # Train Random Forest
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                oob_score=True,
                max_features='sqrt'
            )
            
            print(f"\nTraining Random Forest model...")
            print(f"  Number of trees: {n_estimators}")
            print(f"  Features: {len(self.features)}")
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
            
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, n_jobs=-1)
            
            print(f"\nModel Evaluation:")
            print(f"  Training Accuracy: {train_accuracy:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  CV Mean Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            if hasattr(self.model, 'oob_score_'):
                print(f"  OOB Score: {self.model.oob_score_:.4f}")
            
            print("\nClassification Report (Test Set):")
            print(classification_report(y_test, y_pred_test, target_names=self.classes))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred_test)
            print(f"\nConfusion Matrix:")
            print(f"           Predicted")
            print(f"           LOW  MEDIUM HIGH")
            for i, actual in enumerate(self.classes):
                row = f"Actual {actual:6s}"
                for j in range(3):
                    row += f" {cm[i, j]:6d}"
                print(row)
            
            # Feature importance
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"\nTop 10 Feature Importance:")
            for i, idx in enumerate(indices[:10]):
                if idx < len(self.features):
                    print(f"  {i+1:2d}. {self.features[idx]:25s}: {importances[idx]:.4f}")
            
            # Overfitting check
            if train_accuracy - test_accuracy > 0.05:
                print(f"\nWARNING: Possible overfitting detected!")
                print(f"  Train-Test Difference: {(train_accuracy - test_accuracy):.4f}")
            
            # Save model and statistics
            self._save_model(test_accuracy, cv_scores.mean(), X.shape[0])
            self._save_feature_statistics()
            
            print("\n" + "="*60)
            print("MODEL TRAINING COMPLETED SUCCESSFULLY")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            traceback.print_exc()
            return False
    
    def _save_feature_statistics(self):
        """Save feature statistics to JSON file"""
        if self.feature_statistics:
            import json
            stats_path = 'models/feature_statistics.json'
            with open(stats_path, 'w') as f:
                stats_dict = {}
                for feat, stats in self.feature_statistics.items():
                    stats_dict[feat] = {k: float(v) for k, v in stats.items()}
                json.dump(stats_dict, f, indent=2)
            print(f"  Feature statistics saved to {stats_path}")
    
    def _load_feature_statistics(self):
        """Load feature statistics from JSON file"""
        stats_path = 'models/feature_statistics.json'
        if os.path.exists(stats_path):
            import json
            with open(stats_path, 'r') as f:
                self.feature_statistics = json.load(f)
    
    def _save_model(self, accuracy, cv_score, sample_count):
        """Save model and metadata"""
        os.makedirs('models', exist_ok=True)
        
        # Save model and scaler
        joblib.dump(self.model, 'models/ocean_model.pkl')
        joblib.dump(self.scaler, 'models/ocean_scaler.pkl')
        
        # Save features list
        with open('models/features.txt', 'w') as f:
            for feat in self.features:
                f.write(f"{feat}\n")
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'features': self.features,
            'feature_count': len(self.features),
            'accuracy': float(accuracy),
            'cv_score': float(cv_score),
            'samples': int(sample_count),
            'classes': self.classes
        }
        
        import json
        with open('models/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nModel saved to models/ directory")
        print(f"  Features: {len(self.features)} variables")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Samples: {sample_count:,}")
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load('models/ocean_model.pkl')
            self.scaler = joblib.load('models/ocean_scaler.pkl')
            
            with open('models/features.txt', 'r') as f:
                features = [line.strip() for line in f.readlines()]
            
            self.features = self._remove_duplicates(features)
            
            self._load_feature_statistics()
            
            print(f"Model loaded with {len(self.features)} features")
            
            if os.path.exists('models/metadata.json'):
                import json
                with open('models/metadata.json', 'r') as f:
                    metadata = json.load(f)
                    print(f"  Accuracy: {metadata.get('accuracy', 'N/A'):.4f}")
                    print(f"  Samples: {metadata.get('samples', 'N/A'):,}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return False
    
    def predict(self, input_values):
        """Make prediction for input values"""
        try:
            # Find CHL value from input
            chl_value = None
            for key, value in input_values.items():
                if 'CHL' in key.upper() and 'UNCERTAINTY' not in key.upper():
                    try:
                        chl_value = float(value)
                        break
                    except:
                        continue
            
            # If not found, try any CHL value
            if chl_value is None:
                for key, value in input_values.items():
                    if 'CHL' in key.upper():
                        try:
                            chl_value = float(value)
                            break
                        except:
                            continue
            
            if chl_value is None:
                return {'error': 'CHL value is required'}
            
            # Prepare features
            prepared = {}
            
            for feature in self.features:
                matched = False
                for input_key, input_val in input_values.items():
                    if feature.upper() == input_key.upper():
                        try:
                            prepared[feature] = float(input_val)
                            matched = True
                            break
                        except:
                            continue
                
                if not matched:
                    if feature in self.feature_statistics:
                        stats = self.feature_statistics[feature]
                        prepared[feature] = stats['median']
                    else:
                        feature_upper = feature.upper()
                        if 'CHL' in feature_upper and 'UNCERTAINTY' not in feature_upper:
                            prepared[feature] = chl_value
                        elif 'CHL' in feature_upper and 'UNCERTAINTY' in feature_upper:
                            prepared[feature] = chl_value * 0.1
                        elif feature == 'PP' or 'PP_' in feature_upper:
                            prepared[feature] = chl_value * 100
                        elif 'KD' in feature_upper:
                            prepared[feature] = 0.1 + (chl_value * 0.05)
                        elif any(phyto in feature_upper for phyto in ['DIATO', 'DINO', 'GREEN', 'HAPTO', 'MICRO', 'NANO', 'PICO', 'PROCHLO', 'PROKAR']):
                            if 'UNCERTAINTY' in feature_upper:
                                prepared[feature] = chl_value * 0.005
                            else:
                                prepared[feature] = chl_value * 0.01
                        elif feature == 'CDM':
                            prepared[feature] = 0.01 + (chl_value * 0.002)
                        elif feature == 'BBP':
                            prepared[feature] = 0.001 + (chl_value * 0.0003)
                        else:
                            prepared[feature] = 0.0
            
            # Create DataFrame with all features
            df = pd.DataFrame([prepared])
            
            # Add missing features
            missing_features = set(self.features) - set(df.columns)
            for feat in missing_features:
                df[feat] = 0.0
            
            # Ensure correct feature order
            df = df[self.features]
            
            # Scale and predict
            scaled = self.scaler.transform(df)
            pred = self.model.predict(scaled)[0]
            proba = self.model.predict_proba(scaled)[0]
            
            pred_int = int(pred)
            confidence = float(np.max(proba))
            
            result = {
                'level': pred_int,
                'level_name': self.classes[pred_int],
                'confidence': confidence,
                'probabilities': {
                    self.classes[i]: float(proba[i]) for i in range(len(self.classes))
                },
                'chl_value': chl_value,
                'features_provided': len(input_values),
                'features_total': len(self.features)
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e), 'traceback': traceback.format_exc()}
    
    def interactive_mode(self):
        """Interactive prediction mode"""
        print("\n" + "=" * 60)
        print("OCEAN POLLUTION PREDICTION SYSTEM")
        print("=" * 60)
        print("Enter 'quit' to exit")
        print("=" * 60)
        
        while True:
            try:
                print(f"\nRequired: Chlorophyll (CHL)")
                print(f"Optional: PP, KD490, DIATO, DINO, GREEN, CDM, BBP")
                print("-" * 50)
                
                chl_input = input("Chlorophyll (CHL) in mg/m³: ").strip()
                if chl_input.lower() == 'quit':
                    print("\nExiting...")
                    break
                
                try:
                    chl_value = float(chl_input)
                    input_data = {'CHL': chl_value}
                except ValueError:
                    print("Please enter a valid number for CHL")
                    continue
                
                print(f"\nAdd more features for better accuracy.")
                add_more = input("Add more features? (y/n): ").strip().lower()
                
                if add_more == 'y':
                    print("\nEnter additional features (comma-separated, KEY=VALUE):")
                    print("Example: PP=300, KD490=0.1, DIATO=0.02")
                    additional = input("Additional features: ").strip()
                    
                    if additional:
                        for pair in additional.split(','):
                            pair = pair.strip()
                            if '=' in pair:
                                key, value = pair.split('=', 1)
                                key = key.strip().upper()
                                try:
                                    input_data[key] = float(value.strip())
                                except ValueError:
                                    print(f"  Skipping {pair}: invalid number")
                
                print(f"\nMaking prediction...")
                result = self.predict(input_data)
                
                if 'error' in result:
                    print(f"\nERROR: {result['error']}")
                else:
                    print(f"\n" + "=" * 50)
                    print(f"PREDICTION RESULTS")
                    print("=" * 50)
                    
                    level_emoji = {'LOW': '✅', 'MEDIUM': '⚠️', 'HIGH': '🚨'}
                    emoji = level_emoji.get(result['level_name'], '📊')
                    
                    print(f"{emoji} Pollution Level: {result['level_name']}")
                    print(f"Confidence: {result['confidence']:.1%}")
                    print(f"Chlorophyll: {result['chl_value']} mg/m³")
                    
                    print(f"\nProbability Distribution:")
                    probs = result['probabilities']
                    for level, prob in probs.items():
                        bar_length = int(prob * 30)
                        bar = '█' * bar_length + '░' * (30 - bar_length)
                        print(f"  {level:7s} [{bar}] {prob:.1%}")
                    
                    print(f"\nRecommendations:")
                    if result['level_name'] == 'LOW':
                        print("  • Water quality is good")
                        print("  • Continue normal monitoring")
                    elif result['level_name'] == 'MEDIUM':
                        print("  • Moderate pollution detected")
                        print("  • Increase monitoring frequency")
                    else:
                        print("  • HIGH POLLUTION LEVEL")
                        print("  • Immediate action required")
                        print("  • Notify authorities")
                
                print(f"\n" + "-" * 50)
                continue_choice = input("Make another prediction? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
                
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue


def main():
    """Main function"""
    print("=" * 70)
    print("OCEAN POLLUTION PREDICTION SYSTEM")
    print("=" * 70)
    
    predictor = OceanPollutionPredictor()
    
    args = sys.argv[1:]
    
    if '--train' in args or '-t' in args:
        print("Training new model...")
        if not predictor.train_model():
            print("Model training failed!")
            sys.exit(1)
        print("\nModel training completed!")
    
    elif os.path.exists('models/ocean_model.pkl'):
        print("Loading existing model...")
        if not predictor.load_model():
            print("Failed to load model. Use --train to train new model.")
            sys.exit(1)
        print("Model loaded successfully!")
    else:
        print("No model found. Use --train to train new model.")
        print("\nUsage: python predict.py --train")
        sys.exit(1)
    
    if '--interactive' in args or '-i' in args:
        predictor.interactive_mode()
    
    elif '--demo' in args or '-d' in args:
        print("\n" + "=" * 60)
        print("DEMO PREDICTIONS")
        print("=" * 60)
        
        test_cases = [
            {'name': 'Open Ocean', 'data': {'CHL': 0.3}},
            {'name': 'Coastal Bay', 'data': {'CHL': 2.5, 'PP': 400, 'KD490': 0.2}},
            {'name': 'Polluted Estuary', 'data': {'CHL': 8.0, 'PP': 800, 'KD490': 0.5, 'DIATO': 0.2}},
            {'name': 'Clear Coastal', 'data': {'CHL': 0.8}},
            {'name': 'Algal Bloom', 'data': {'CHL': 15.0, 'PP': 1200, 'GREEN': 0.5}}
        ]
        
        for test in test_cases:
            print(f"\n{test['name']}:")
            result = predictor.predict(test['data'])
            
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                emoji = {'LOW': '✅', 'MEDIUM': '⚠️', 'HIGH': '🚨'}.get(result['level_name'], '📊')
                print(f"  {emoji} Prediction: {result['level_name']}")
                print(f"  Confidence: {result['confidence']:.1%}")
                print(f"  CHL: {result['chl_value']} mg/m³")
        
        print(f"\n" + "=" * 60)
        print("Demo completed!")
    
    elif '--predict' in args or '-p' in args:
        print("\nQuick Prediction Mode")
        print("Enter feature values as KEY=VALUE pairs")
        print("Example: CHL=2.5 PP=300 KD490=0.2")
        
        try:
            input_str = input("\nEnter features: ").strip()
            if input_str:
                input_data = {}
                for pair in input_str.split():
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        try:
                            input_data[key.upper()] = float(value)
                        except:
                            input_data[key.upper()] = value
                
                if input_data:
                    result = predictor.predict(input_data)
                    if 'error' in result:
                        print(f"Error: {result['error']}")
                    else:
                        print(f"\nResult: {result['level_name']}")
                        print(f"Confidence: {result['confidence']:.1%}")
        except KeyboardInterrupt:
            print("\nExiting...")
    
    else:
        print("\n" + "=" * 70)
        print("USAGE OPTIONS")
        print("=" * 70)
        print("  --train, -t       : Train a new model")
        print("  --interactive, -i : Interactive prediction mode")
        print("  --demo, -d        : Run demo predictions")
        print("  --predict, -p     : Quick prediction from command line")
        print("\n" + "=" * 70)
        print("EXAMPLES")
        print("=" * 70)
        print("  python predict.py --train")
        print("  python predict.py --interactive")
        print("  python predict.py --demo")


if __name__ == "__main__":
    main()
