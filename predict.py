"""
Ocean Pollution Prediction System - Complete with Real Time Series
"""

import joblib
import pandas as pd
import numpy as np
import os
import sys
import warnings
from datetime import datetime, timedelta
import glob
warnings.filterwarnings('ignore')

class OceanPollutionPredictor:
    def __init__(self, auto_train=True):
        self.model = None
        self.scaler = None
        self.features = []
        self.classes = ['LOW', 'MEDIUM', 'HIGH']
        
        if auto_train:
            self._initialize_system()
        else:
            self._load_existing_model()
    
    def _initialize_system(self):
        print("Initializing ocean pollution prediction system...")
        
        if self._check_existing_model():
            print("Found existing model, loading...")
            self._load_existing_model()
            return
        
        if not self._check_data_exists():
            print("No ocean data found in data/raw/")
            print("Please add NetCDF files to data/raw/ folder")
            sys.exit(1)
        
        print("Training new model with your ocean data...")
        if self._train_model():
            print("Model trained successfully!")
            self._save_model()
        else:
            print("Model training failed")
            sys.exit(1)
    
    def _check_existing_model(self):
        required_files = [
            'models/smart_ocean_model.pkl',
            'models/smart_scaler.pkl',
            'models/smart_features.txt'
        ]
        return all(os.path.exists(f) for f in required_files)
    
    def _check_data_exists(self):
        data_dir = "data/raw"
        if not os.path.exists(data_dir):
            return False
        
        nc_files = glob.glob(os.path.join(data_dir, "*.nc"))
        return len(nc_files) > 0
    
    def _load_existing_model(self):
        try:
            self.model = joblib.load('models/smart_ocean_model.pkl')
            self.scaler = joblib.load('models/smart_scaler.pkl')
            
            with open('models/smart_features.txt', 'r') as f:
                self.features = [line.strip() for line in f.readlines()]
            
            print(f"Model loaded: {type(self.model).__name__}")
            print(f"Features: {self.features}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        return True
    
    def _train_model(self):
        try:
            import xarray as xr
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            
            chl_file = None
            for f in os.listdir('data/raw'):
                if f.endswith('.nc'):
                    filepath = os.path.join('data/raw', f)
                    try:
                        ds = xr.open_dataset(filepath)
                        if 'CHL' in ds.variables:
                            chl_file = filepath
                            ds.close()
                            break
                        ds.close()
                    except:
                        continue
            
            if not chl_file:
                print("No chlorophyll data found in NetCDF files")
                return False
            
            print(f"Using data from: {os.path.basename(chl_file)}")
            ds = xr.open_dataset(chl_file)
            chl_data = ds['CHL'].values.flatten()
            ds.close()
            
            if len(chl_data) > 10000:
                chl_data = np.random.choice(chl_data, 10000, replace=False)
            
            y = np.zeros(len(chl_data))
            y[chl_data > 5.0] = 2
            y[(chl_data > 1.0) & (chl_data <= 5.0)] = 1
            
            print(f"Class distribution:")
            print(f"   LOW (chl <= 1.0): {sum(y == 0)} samples")
            print(f"   MEDIUM (1.0 < chl <= 5.0): {sum(y == 1)} samples")
            print(f"   HIGH (chl > 5.0): {sum(y == 2)} samples")
            
            X = np.column_stack([
                chl_data,
                np.random.normal(300, 100, len(chl_data)),
                np.random.normal(15, 8, len(chl_data))
            ])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Model accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=self.classes))
            
            self.features = ['CHL', 'PP', 'TRANS']
            
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_model(self):
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, 'models/smart_ocean_model.pkl')
        joblib.dump(self.scaler, 'models/smart_scaler.pkl')
        
        with open('models/smart_features.txt', 'w') as f:
            for feat in self.features:
                f.write(f"{feat}\n")
        
        print("Model saved to models/ directory")
    
    def predict(self, chlorophyll, productivity=None, transparency=None):
        try:
            if productivity is None:
                productivity = chlorophyll * 60
            
            if transparency is None:
                transparency = max(1.0, 30 - (chlorophyll * 3))
            
            input_data = {}
            for feat in self.features:
                if 'CHL' in feat.upper():
                    input_data[feat] = float(chlorophyll)
                elif 'PP' in feat.upper():
                    input_data[feat] = float(productivity)
                elif 'TRANS' in feat.upper():
                    input_data[feat] = float(transparency)
                else:
                    input_data[feat] = 0.0
            
            df = pd.DataFrame([input_data])
            
            for feat in self.features:
                if feat not in df.columns:
                    df[feat] = 0.0
            
            df = df[self.features]
            
            scaled = self.scaler.transform(df)
            pred = self.model.predict(scaled)[0]
            proba = self.model.predict_proba(scaled)[0]
            
            pred_int = int(pred)
            
            level_descriptions = {
                0: "Clean water - Low nutrient levels",
                1: "Moderate pollution - Potential algal blooms",
                2: "High pollution - Likely harmful algal blooms"
            }
            
            recommendations = {
                0: "Water quality is excellent. Continue normal monitoring.",
                1: "Moderate pollution detected. Increase monitoring frequency.",
                2: "High pollution level! Immediate action required."
            }
            
            result = {
                'pollution_level': pred_int,
                'level_name': self.classes[pred_int],
                'level_description': level_descriptions[pred_int],
                'confidence': float(np.max(proba)),
                'probabilities': {
                    'low': float(proba[0]),
                    'medium': float(proba[1]),
                    'high': float(proba[2])
                },
                'recommendation': recommendations[pred_int],
                'input_values': {
                    'chlorophyll': chlorophyll,
                    'productivity': productivity,
                    'transparency': transparency
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

class TimeSeriesForecaster:
    def __init__(self):
        self.window_size = 7
    
    def run_time_series_forecast(self, days=7):
        """Run the time series prediction module"""
        try:
            # Import and use the time_series_predictor module
            from time_series_predictor import TimeSeriesPredictionSystem
            
            print("\n" + "=" * 60)
            print("TIME SERIES FORECAST")
            print("=" * 60)
            
            # Initialize system
            ts_system = TimeSeriesPredictionSystem(model_type="simple")
            
            # Try to load real data first
            real_data_files = [
                'ocean_timeseries.csv',
                'historical_data.csv',
                'chlorophyll_data.csv',
                'sample_time_series.csv'  # Created by time_series_predictor.py
            ]
            
            data_loaded = False
            for data_file in real_data_files:
                if os.path.exists(data_file):
                    print(f"Loading data from: {data_file}")
                    data = ts_system.load_historical_data(data_file)
                    if data is not None and len(data) > 20:
                        data_loaded = True
                        break
            
            # If no real data, create sample
            if not data_loaded:
                print("No time series data found, creating sample data...")
                from time_series_predictor import create_sample_time_series_data
                df = create_sample_time_series_data()
                ts_system.load_historical_data('sample_time_series.csv')
            
            # Train and predict
            print("Training time series models...")
            ts_system.train_models()
            
            print(f"Generating {days}-day forecast...")
            predictions = ts_system.predict_future(days=days)
            
            if predictions:
                print(f"\n{days}-DAY POLLUTION FORECAST")
                print("-" * 50)
                
                for pred in predictions:
                    date_str = pred.timestamp.strftime('%Y-%m-%d')
                    print(f"{date_str}:")
                    print(f"  Chlorophyll: {pred.chlorophyll_pred:.2f} mg/m3")
                    print(f"  Pollution: {pred.pollution_level}")
                    print(f"  Confidence: {pred.confidence:.1%}")
                    print(f"  Trend: {pred.trend.upper()}")
                    print()
                
                # Save forecast
                self._save_forecast(predictions)
                return True
            else:
                print("No predictions generated")
                return False
                
        except ImportError as e:
            print(f"Time series module not available: {e}")
            print("Install required packages: pip install scipy")
            return False
        except Exception as e:
            print(f"Error in time series forecast: {e}")
            return False
    
    def _save_forecast(self, predictions):
        """Save forecast to CSV"""
        data = []
        for pred in predictions:
            data.append({
                'date': pred.timestamp,
                'chlorophyll': pred.chlorophyll_pred,
                'pollution_level': pred.pollution_level,
                'confidence': pred.confidence,
                'trend': pred.trend
            })
        
        df = pd.DataFrame(data)
        filename = f"pollution_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Forecast saved to: {filename}")
        return filename
    
    def simple_forecast(self, days=5):
        """Simple forecast if main module fails"""
        print("\n" + "=" * 60)
        print("SIMPLE TIME SERIES FORECAST")
        print("=" * 60)
        
        # Create simple forecast
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        data = []
        
        for i, date in enumerate(dates):
            chl = 1.5 + 0.5 * np.sin(2 * np.pi * i / 7) + i * 0.02
            data.append({
                'date': date,
                'chlorophyll': max(0.1, chl + np.random.normal(0, 0.15))
            })
        
        df = pd.DataFrame(data)
        chl_data = df['chlorophyll'].tolist()
        
        print(f"Sample data: {len(chl_data)} days")
        print(f"Current chlorophyll: {chl_data[-1]:.2f} mg/m3")
        
        # Simple moving average forecast
        forecasts = []
        window = chl_data[-self.window_size:].copy()
        
        for _ in range(days):
            pred = np.mean(window)
            forecasts.append(pred)
            window = window[1:] + [pred]
        
        print(f"\n{days}-Day Simple Forecast:")
        print("-" * 40)
        
        last_date = df['date'].iloc[-1]
        for i, pred in enumerate(forecasts, 1):
            forecast_date = last_date + timedelta(days=i)
            date_str = forecast_date.strftime('%Y-%m-%d')
            
            if pred <= 1.0:
                level = "LOW"
            elif pred <= 5.0:
                level = "MEDIUM"
            else:
                level = "HIGH"
            
            print(f"{date_str}: {pred:.2f} mg/m3 - {level}")
        
        return forecasts

def run_standard_demo(predictor):
    print("\n" + "=" * 60)
    print("DEMONSTRATION PREDICTIONS")
    print("=" * 60)
    
    demo_locations = [
        {"name": "Open Ocean", "chlorophyll": 0.1, "productivity": 50, "transparency": 30},
        {"name": "Remote Coast", "chlorophyll": 0.5, "productivity": 100, "transparency": 25},
        {"name": "Coastal Bay", "chlorophyll": 2.0, "productivity": 300, "transparency": 10},
        {"name": "Estuary", "chlorophyll": 4.0, "productivity": 500, "transparency": 5},
        {"name": "Port Area", "chlorophyll": 8.0, "productivity": 800, "transparency": 2},
        {"name": "Industrial Zone", "chlorophyll": 15.0, "productivity": 1200, "transparency": 1}
    ]
    
    for loc in demo_locations:
        result = predictor.predict(
            loc["chlorophyll"], 
            loc["productivity"], 
            loc["transparency"]
        )
        
        if result:
            print(f"\n{loc['name']}:")
            print(f"  Chlorophyll: {result['input_values']['chlorophyll']} mg/m3")
            print(f"  Productivity: {result['input_values']['productivity']} mg C/m2/day")
            print(f"  Transparency: {result['input_values']['transparency']} m")
            print(f"  Prediction: {result['level_name']}")
            print(f"  Confidence: {result['confidence']:.1%}")
            print(f"  {result['recommendation']}")

def print_usage():
    print("\n" + "=" * 60)
    print("HELP & USAGE")
    print("=" * 60)
    print("""
OCEAN POLLUTION PREDICTION SYSTEM WITH TIME SERIES
===================================================

Usage:
  python predict.py                    # Run demo predictions
  python predict.py --interactive      # Interactive mode
  python predict.py --train           # Retrain model
  python predict.py --timeseries      # Include time series forecast
  python predict.py --forecast DAYS   # Days to forecast (default: 7)

Examples:
  python predict.py                    # Standard demo
  python predict.py --timeseries       # Demo with forecast
  python predict.py --forecast 7       # 7-day forecast
  python predict.py --train            # Retrain model
  python predict.py --interactive      # Interactive prediction

Time Series Features:
  - Uses time_series_predictor.py module
  - Can load real time series data from CSV
  - Generates multi-day pollution forecasts
  - Saves forecasts to CSV files

In your Python code:
  
  from predict import OceanPollutionPredictor, TimeSeriesForecaster
  
  # Real-time prediction
  predictor = OceanPollutionPredictor()
  result = predictor.predict(3.5, 250.0, 12.0)
  print(f"Pollution: {result['level_name']}")
  print(f"Confidence: {result['confidence']:.1%}")
  
  # Time series forecasting
  forecaster = TimeSeriesForecaster()
  success = forecaster.run_time_series_forecast(days=7)
    """)

def main():
    args = sys.argv[1:]
    
    # Parse arguments manually
    interactive_mode = any(arg in ['--interactive', '-i'] for arg in args)
    train_mode = any(arg in ['--train', '-t'] for arg in args)
    timeseries_mode = any(arg in ['--timeseries', '-ts'] for arg in args)
    help_mode = any(arg in ['--help', '-h', '/?'] for arg in args)
    
    # Get forecast days
    forecast_days = 7
    for i, arg in enumerate(args):
        if arg in ['--forecast', '-f'] and i + 1 < len(args):
            try:
                forecast_days = int(args[i + 1])
            except:
                pass
    
    print("=" * 60)
    print("OCEAN POLLUTION PREDICTION SYSTEM")
    print("=" * 60)
    
    print("Initializing Ocean Pollution Prediction System...")
    
    try:
        if train_mode:
            print("\nTraining new model...")
            predictor = OceanPollutionPredictor(auto_train=True)
        else:
            predictor = OceanPollutionPredictor(auto_train=False)
        
        if not predictor.model:
            print("Failed to initialize predictor")
            return
        
        if help_mode:
            print_usage()
            return
        
        if interactive_mode:
            print("\n" + "=" * 60)
            print("INTERACTIVE PREDICTION MODE")
            print("=" * 60)
            
            while True:
                try:
                    print("\nEnter ocean parameters (or 'quit' to exit):")
                    
                    chl = input("Chlorophyll concentration (mg/m3): ").strip()
                    if chl.lower() == 'quit':
                        break
                    
                    pp = input("Primary productivity (mg C/m2/day, press Enter for auto): ").strip()
                    trans = input("Water transparency (m, press Enter for auto): ").strip()
                    
                    chl_val = float(chl)
                    pp_val = float(pp) if pp else None
                    trans_val = float(trans) if trans else None
                    
                    result = predictor.predict(chl_val, pp_val, trans_val)
                    
                    if result:
                        print(f"\nPREDICTION RESULTS:")
                        print(f"  Pollution Level: {result['level_name']}")
                        print(f"  Confidence: {result['confidence']:.1%}")
                        print(f"  {result['recommendation']}")
                        
                        print(f"\n  Probabilities:")
                        print(f"    Low: {result['probabilities']['low']:.1%}")
                        print(f"    Medium: {result['probabilities']['medium']:.1%}")
                        print(f"    High: {result['probabilities']['high']:.1%}")
                    
                except ValueError:
                    print("Please enter valid numbers")
                except KeyboardInterrupt:
                    print("\nExiting interactive mode")
                    break
                except Exception as e:
                    print(f"Error: {e}")
        else:
            # Default mode: run standard demo
            print("\n" + "=" * 60)
            print("SYSTEM INFORMATION")
            print("=" * 60)
            print("""
Ocean Pollution Prediction AI
Model: Random Forest Classifier
Based on: Chlorophyll concentration
Scientific thresholds:
   LOW: <= 1.0 mg/m3 (Clean)
   MEDIUM: 1.0-5.0 mg/m3 (Moderate)
   HIGH: > 5.0 mg/m3 (Polluted)
            """)
            
            run_standard_demo(predictor)
            
            if timeseries_mode:
                forecaster = TimeSeriesForecaster()
                print("\n" + "=" * 60)
                print("TIME SERIES FORECAST INITIALIZATION")
                print("=" * 60)
                
                # Try to run advanced time series
                success = forecaster.run_time_series_forecast(days=forecast_days)
                
                if not success:
                    print("\nFalling back to simple forecast...")
                    forecaster.simple_forecast(days=forecast_days)
            
            # Show usage at the end
            print_usage()
        
        print("\n" + "=" * 60)
        print("END")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Ensure data/raw/ contains NetCDF files")
        print("2. Install requirements: pip install -r requirements.txt")
        print("3. Run with --train to rebuild model")

if __name__ == "__main__":
    main()
