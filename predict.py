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
from sklearn.metrics import accuracy_score, classification_report
warnings.filterwarnings('ignore')

class OceanPollutionPredictor:
    def __init__(self, data_path="data/raw"):
        self.model = None
        self.scaler = None
        self.features = []
        self.classes = ['LOW', 'MEDIUM', 'HIGH']
        self.data_path = data_path
        self.datasets = {}
    
    def _remove_duplicates(self, features_list):
        """Remove duplicate features while preserving order"""
        unique_features = []
        seen = set()
        for feat in features_list:
            if feat not in seen:
                unique_features.append(feat)
                seen.add(feat)
        return unique_features
    
    def load_datasets(self):
        try:
            nc_files = glob.glob(os.path.join(self.data_path, "*.nc"))
            if not nc_files:
                print("No NetCDF files found in data/raw/")
                return False
            
            for filepath in nc_files:
                filename = os.path.basename(filepath)
                try:
                    ds = xr.open_dataset(filepath)
                    self.datasets[filename] = ds
                    print(f"Loaded: {filename}")
                except:
                    print(f"Error loading: {filename}")
                    continue
            
            return bool(self.datasets)
        except:
            return False
    
    def create_dataset(self, sample_limit=100000):
        try:
            reference_ds = None
            reference_filename = None
            
            for filename, ds in self.datasets.items():
                if 'CHL' in ds.variables:
                    reference_ds = ds
                    reference_filename = filename
                    break
            
            if reference_ds is None:
                print("No CHL data found")
                return None, None, None
            
            print(f"Using reference dataset: {reference_filename}")
            
            chl_data = reference_ds['CHL'].values
            chl_flat = chl_data.flatten()
            valid_mask = ~np.isnan(chl_flat)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > sample_limit:
                sample_indices = np.random.choice(valid_indices, sample_limit, replace=False)
                print(f"Sampled to: {sample_limit} points")
            else:
                sample_indices = valid_indices
            
            time_size, lat_size, lon_size = chl_data.shape
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
                    'chl_value': chl_flat[idx]
                })
            
            # Track added features to avoid duplicates
            feature_data = {'CHL': np.array([p['chl_value'] for p in sampled_points])}
            feature_names = ['CHL']
            added_features = set(['CHL'])  # Track what we've already added
            
            for filename, ds in self.datasets.items():
                print(f"Extracting variables from: {filename}")
                
                for var_name in ds.variables:
                    if var_name.lower() in ['time', 'latitude', 'longitude', 'lat', 'lon']:
                        continue
                    
                    # Skip if already added from another file
                    if var_name in added_features:
                        print(f"  Skipped: {var_name} (already exists)")
                        continue
                    
                    # Skip CHL from reference dataset (already added)
                    if var_name == 'CHL' and filename == reference_filename:
                        continue
                    
                    try:
                        var_data = ds[var_name].values
                        if var_data.ndim != 3:
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
                            except IndexError:
                                var_values.append(np.nan)
                        
                        valid_ratio = valid_count / len(sampled_points)
                        if valid_ratio > 0.3:
                            var_array = np.array(var_values)
                            median_val = np.nanmedian(var_array)
                            var_array[np.isnan(var_array)] = median_val
                            
                            # Add to features
                            feature_data[var_name] = var_array
                            feature_names.append(var_name)
                            added_features.add(var_name)
                            print(f"  Added: {var_name} ({valid_ratio:.1%} valid)")
                        
                    except Exception as e:
                        print(f"  Error extracting {var_name}: {e}")
                        continue
            
            # Create feature matrix
            X_list = []
            final_features = []
            
            for feat_name in feature_names:
                if feat_name in feature_data:
                    X_list.append(feature_data[feat_name])
                    final_features.append(feat_name)
            
            X = np.column_stack(X_list)
            
            # Find CHL index
            try:
                chl_idx = final_features.index('CHL')
                chl_values = X[:, chl_idx]
            except ValueError:
                print("Warning: CHL not found in final features, using first column")
                chl_values = X[:, 0]
            
            # Create labels
            y = np.zeros(len(chl_values))
            y[chl_values > 5.0] = 2
            y[(chl_values > 1.0) & (chl_values <= 5.0)] = 1
            
            # Close datasets
            for ds in self.datasets.values():
                ds.close()
            
            # Remove any remaining duplicates
            final_features = self._remove_duplicates(final_features)
            
            print(f"\n✅ Dataset created:")
            print(f"  Samples: {X.shape[0]}")
            print(f"  Features: {len(final_features)} (unique)")
            print(f"  Features list: {final_features[:10]}{'...' if len(final_features) > 10 else ''}")
            
            return X, y, final_features
            
        except Exception as e:
            print(f"Error creating dataset: {e}")
            traceback.print_exc()
            return None, None, None
    
    def train_model(self):
        try:
            if not self.load_datasets():
                return False
            
            X, y, feature_names = self.create_dataset()
            
            if X is None:
                return False
            
            # Ensure no duplicates
            self.features = self._remove_duplicates(feature_names)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                oob_score=True
            )
            
            print("\nTraining Random Forest model...")
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, n_jobs=-1)
            
            print(f"\n📊 Model Evaluation:")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Samples: {len(y_test)}")
            print(f"CV Mean Score: {cv_scores.mean():.4f}")
            
            if hasattr(self.model, 'oob_score_'):
                print(f"OOB Score: {self.model.oob_score_:.4f}")
            
            print("\n📋 Classification Report:")
            print(classification_report(y_test, y_pred, target_names=self.classes))
            
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\n⭐ Top 10 Feature Importance:")
            for i, idx in enumerate(indices[:10]):
                print(f"  {i+1:2d}. {self.features[idx]}: {importances[idx]:.4f}")
            
            self._save_model(accuracy, cv_scores.mean(), X.shape[0])
            
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            traceback.print_exc()
            return False
    
    def _save_model(self, accuracy, cv_score, sample_count):
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, 'models/ocean_model.pkl')
        joblib.dump(self.scaler, 'models/ocean_scaler.pkl')
        
        with open('models/features.txt', 'w') as f:
            for feat in self.features:
                f.write(f"{feat}\n")
        
        metadata = {
            'training_date': datetime.now().isoformat(),
            'features': self.features,
            'unique_features': True,
            'feature_count': len(self.features),
            'accuracy': float(accuracy),
            'cv_score': float(cv_score),
            'samples': int(sample_count)
        }
        
        import json
        with open('models/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Model saved to models/ directory")
        print(f"   Features: {len(self.features)} unique variables")
        print(f"   Accuracy: {accuracy:.4f}")
    
    def load_model(self):
        try:
            self.model = joblib.load('models/ocean_model.pkl')
            self.scaler = joblib.load('models/ocean_scaler.pkl')
            
            # Load features and remove duplicates
            with open('models/features.txt', 'r') as f:
                features = [line.strip() for line in f.readlines()]
            
            self.features = self._remove_duplicates(features)
            
            print(f"✅ Model loaded with {len(self.features)} unique features")
            
            # Load metadata if exists
            if os.path.exists('models/metadata.json'):
                import json
                with open('models/metadata.json', 'r') as f:
                    metadata = json.load(f)
                    print(f"   Accuracy: {metadata.get('accuracy', 'N/A')}")
                    print(f"   Samples: {metadata.get('samples', 'N/A')}")
                    print(f"   Features: {metadata.get('feature_count', len(self.features))}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, input_values):
        try:
            # Check for chlorophyll
            has_chl = any('CHL' in key.upper() for key in input_values.keys())
            if not has_chl:
                return {'error': 'Chlorophyll (CHL) value is required'}
            
            # Prepare input with all required features
            prepared = {}
            
            for feature in self.features:
                # Try to find matching input (case insensitive)
                matched = False
                for input_key, input_val in input_values.items():
                    if feature.upper() == input_key.upper():
                        try:
                            prepared[feature] = float(input_val)
                            matched = True
                            break
                        except:
                            continue
                
                # If not provided, use reasonable default
                if not matched:
                    feature_upper = feature.upper()
                    if 'CHL' in feature_upper:
                        prepared[feature] = 1.0  # Default chlorophyll
                    elif feature == 'PP':
                        prepared[feature] = 300.0  # Default primary productivity
                    elif 'KD' in feature_upper:
                        prepared[feature] = 0.1  # Default diffuse attenuation
                    elif any(phyto in feature_upper for phyto in ['DIATO', 'DINO', 'GREEN', 'HAPTO', 'MICRO', 'NANO', 'PICO']):
                        prepared[feature] = 0.01  # Default phytoplankton
                    elif 'UNCERTAINTY' in feature_upper:
                        prepared[feature] = 0.1  # Default uncertainty
                    elif feature == 'CDM':
                        prepared[feature] = 0.02  # Default CDM
                    elif feature == 'BBP':
                        prepared[feature] = 0.003  # Default BBP
                    elif feature == 'FLAGS':
                        prepared[feature] = 0.0  # Default flags
                    else:
                        prepared[feature] = 0.0
            
            # Create DataFrame in correct feature order
            df = pd.DataFrame([prepared])
            
            # Ensure all features are present
            missing_features = set(self.features) - set(df.columns)
            for feat in missing_features:
                df[feat] = 0.0
            
            df = df[self.features]
            
            # Scale and predict
            scaled = self.scaler.transform(df)
            pred = self.model.predict(scaled)[0]
            proba = self.model.predict_proba(scaled)[0]
            
            pred_int = int(pred)
            
            result = {
                'level': pred_int,
                'level_name': self.classes[pred_int],
                'confidence': float(np.max(proba)),
                'probabilities': {
                    self.classes[i]: float(proba[i]) for i in range(len(self.classes))
                },
                'features_used': len([v for v in prepared.values() if v != 0.0]),
                'total_features': len(self.features)
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e), 'traceback': traceback.format_exc()}
    
    def interactive_mode(self):
        print("\n" + "=" * 50)
        print("Interactive Prediction Mode")
        print("Enter 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                print(f"\nRequired feature: CHL (Chlorophyll)")
                print("Optional features: PP, KD490, DIATO, DINO, etc.")
                print("-" * 40)
                
                chl = input("Chlorophyll (CHL) in mg/m³: ").strip()
                if chl.lower() == 'quit':
                    break
                
                # Get additional features
                input_data = {'CHL': float(chl)}
                
                add_more = input("Add more features? (y/n): ").strip().lower()
                if add_more == 'y':
                    print("\nEnter additional features (format: FEATURE=VALUE):")
                    print("Example: PP=300, KD490=0.1, DIATO=0.02")
                    additional = input("Additional features: ").strip()
                    
                    if additional:
                        for pair in additional.split(','):
                            pair = pair.strip()
                            if '=' in pair:
                                key, value = pair.split('=', 1)
                                try:
                                    input_data[key.strip().upper()] = float(value.strip())
                                except:
                                    print(f"Invalid format: {pair}")
                
                # Make prediction
                result = self.predict(input_data)
                
                if 'error' in result:
                    print(f"\n❌ Error: {result['error']}")
                else:
                    print(f"\n✅ Prediction Results:")
                    print(f"   Pollution Level: {result['level_name']}")
                    print(f"   Confidence: {result['confidence']:.1%}")
                    print(f"   Features Used: {result['features_used']}/{result['total_features']}")
                    
                    print(f"\n   Probabilities:")
                    for level, prob in result['probabilities'].items():
                        print(f"     {level}: {prob:.1%}")
                    
                    # Recommendation
                    recommendations = {
                        0: "✅ Water quality is good. Continue normal monitoring.",
                        1: "⚠️ Moderate pollution detected. Increase monitoring frequency.",
                        2: "🚨 High pollution level! Immediate action required."
                    }
                    print(f"\n   Recommendation: {recommendations[result['level']]}")
                
            except ValueError:
                print("Please enter valid numbers")
            except KeyboardInterrupt:
                print("\nExiting interactive mode")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    print("=" * 60)
    print("Ocean Pollution Prediction System")
    print("=" * 60)
    
    predictor = OceanPollutionPredictor()
    
    args = sys.argv[1:]
    
    if '--train' in args or '-t' in args:
        print("Training new model...")
        if not predictor.train_model():
            return
    elif os.path.exists('models/ocean_model.pkl'):
        print("Loading existing model...")
        if not predictor.load_model():
            print("Failed to load model. Use --train to train new model.")
            return
    else:
        print("No model found. Use --train to train new model.")
        return
    
    if '--interactive' in args or '-i' in args:
        predictor.interactive_mode()
    elif '--demo' in args or '-d' in args:
        print("\nDemo Predictions:")
        print("-" * 40)
        
        test_cases = [
            {'name': 'Open Ocean', 'CHL': 0.3},
            {'name': 'Coastal Bay', 'CHL': 2.5, 'PP': 400},
            {'name': 'Polluted Estuary', 'CHL': 8.0, 'PP': 800, 'DIATO': 0.2}
        ]
        
        for test in test_cases:
            print(f"\n📍 {test['name']}:")
            result = predictor.predict(test)
            
            if 'error' not in result:
                print(f"  Chlorophyll: {test['CHL']} mg/m³")
                print(f"  Prediction: {result['level_name']}")
                print(f"  Confidence: {result['confidence']:.1%}")
            else:
                print(f"  Error: {result['error']}")
    
    else:
        print("\nUsage options:")
        print("  --interactive, -i : Interactive prediction mode")
        print("  --demo, -d        : Run demo predictions")
        print("  --train, -t       : Train new model")
        print("\nExamples:")
        print("  python predict.py --demo")
        print("  python predict.py --interactive")
        print("  python predict.py --train")


if __name__ == "__main__":
    main()
