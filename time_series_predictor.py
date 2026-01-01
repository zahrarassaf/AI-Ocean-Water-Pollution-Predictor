"""
Time Series Prediction Module for Ocean Pollution
LSTM-Free Version
"""

import numpy as np
import pandas as pd
import os  # Added import
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

@dataclass
class TimeSeriesPrediction:
    timestamp: datetime
    chlorophyll_pred: float
    productivity_pred: float
    transparency_pred: float
    pollution_level: str
    confidence: float
    trend: str

class SimpleTimeSeriesPredictor:
    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.scalers = {}
        self.models = {}
        
    def create_features(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(series) - self.window_size):
            X.append(series[i:i + self.window_size])
            y.append(series[i + self.window_size])
        return np.array(X), np.array(y)
    
    def prepare_data(self, df: pd.DataFrame) -> Dict:
        features = ['CHL_diffuse_at', 'PP_primary_pr', 'TRANS_secchi_dep']
        prepared_data = {}
        
        for feature in features:
            if feature in df.columns:
                scaler = MinMaxScaler()
                scaled_values = scaler.fit_transform(df[[feature]].values)
                self.scalers[feature] = scaler
                X, y = self.create_features(scaled_values.flatten())
                prepared_data[feature] = {'X': X, 'y': y}
        
        return prepared_data
    
    def train(self, prepared_data: Dict):
        for feature, data in prepared_data.items():
            if len(data['X']) > 0:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                model.fit(data['X'], data['y'])
                self.models[feature] = model
    
    def predict_next_days(self, last_values: Dict[str, np.ndarray], days: int = 7) -> Dict[str, List[float]]:
        predictions = {}
        
        for feature, model in self.models.items():
            if feature in last_values:
                current_seq = last_values[feature].copy()
                feature_preds = []
                
                for _ in range(days):
                    if len(current_seq) >= self.window_size:
                        pred = model.predict(current_seq[-self.window_size:].reshape(1, -1))[0]
                        feature_preds.append(pred)
                        current_seq = np.append(current_seq, pred)
                
                if feature in self.scalers:
                    feature_preds = self.scalers[feature].inverse_transform(
                        np.array(feature_preds).reshape(-1, 1)
                    ).flatten()
                
                predictions[feature] = feature_preds.tolist()
        
        return predictions

class PollutionTrendAnalyzer:
    def __init__(self):
        self.trend_window = 5
    
    def analyze_trend(self, values: List[float]) -> Dict:
        if len(values) < 2:
            return {"trend": "insufficient_data", "slope": 0.0}
        
        x = np.arange(len(values))
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        except:
            return {"trend": "unknown", "slope": 0.0, "r_squared": 0.0, "p_value": 1.0}
        
        if abs(slope) < 0.01:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            "trend": trend,
            "slope": float(slope),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "current_value": float(values[-1]) if values else 0.0
        }

class TimeSeriesPredictionSystem:
    def __init__(self, model_type: str = "simple"):
        self.model_type = model_type
        self.predictor = None
        self.trend_analyzer = PollutionTrendAnalyzer()
        self.historical_data = None
        
    def load_historical_data(self, data_path: str) -> pd.DataFrame:
        try:
            self.historical_data = pd.read_csv(data_path)
            if 'date' in self.historical_data.columns:
                self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
                self.historical_data = self.historical_data.set_index('date')
            return self.historical_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def train_models(self):
        if self.historical_data is None:
            raise ValueError("Load historical data first")
        
        if self.model_type == "simple":
            self.predictor = SimpleTimeSeriesPredictor()
            prepared_data = self.predictor.prepare_data(self.historical_data)
            self.predictor.train(prepared_data)
            return True
        
        return False
    
    def predict_future(self, days: int = 7) -> List[TimeSeriesPrediction]:
        if self.historical_data is None or self.predictor is None:
            return []
        
        try:
            last_sequences = {}
            for feature in ['CHL_diffuse_at', 'PP_primary_pr', 'TRANS_secchi_dep']:
                if feature in self.historical_data.columns:
                    scaler = self.predictor.scalers.get(feature)
                    if scaler:
                        values = self.historical_data[feature].values[-self.predictor.window_size:]
                        scaled = scaler.transform(values.reshape(-1, 1)).flatten()
                        last_sequences[feature] = scaled
            
            predictions = self.predictor.predict_next_days(last_sequences, days)
            
            results = []
            current_date = self.historical_data.index[-1]
            
            if 'CHL_diffuse_at' in predictions:
                chl_predictions = predictions['CHL_diffuse_at']
                
                for i, pred_chl in enumerate(chl_predictions[:days]):
                    pred_date = current_date + timedelta(days=i+1)
                    
                    historical_chl = list(self.historical_data['CHL_diffuse_at'].values[-5:])
                    trend_data = self.trend_analyzer.analyze_trend(historical_chl + [pred_chl])
                    
                    level = "LOW"
                    if pred_chl > 5.0:
                        level = "HIGH"
                    elif pred_chl > 1.0:
                        level = "MEDIUM"
                    
                    confidence = 0.9 - abs(trend_data['slope']) * 10
                    confidence = max(0.5, min(0.95, confidence))
                    
                    results.append(TimeSeriesPrediction(
                        timestamp=pred_date,
                        chlorophyll_pred=float(pred_chl),
                        productivity_pred=float(predictions.get('PP_primary_pr', [0.0])[i] if i < len(predictions.get('PP_primary_pr', [])) else 0.0),
                        transparency_pred=float(predictions.get('TRANS_secchi_dep', [0.0])[i] if i < len(predictions.get('TRANS_secchi_dep', [])) else 0.0),
                        pollution_level=level,
                        confidence=float(confidence),
                        trend=trend_data['trend']
                    ))
            
            return results
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return []

def create_sample_time_series_data():
    """Create sample time series data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    data = []
    base_chl = 1.0
    
    for i, date in enumerate(dates):
        seasonal = 0.5 * np.sin(2 * np.pi * i / 30)
        noise = np.random.normal(0, 0.2)
        
        chl = max(0.1, base_chl + seasonal + noise + i * 0.01)
        pp = chl * 80 + np.random.normal(0, 20)
        trans = max(1.0, 30 - chl * 3 + np.random.normal(0, 2))
        
        data.append({
            'date': date,
            'CHL_diffuse_at': round(chl, 3),
            'PP_primary_pr': round(pp, 1),
            'TRANS_secchi_dep': round(trans, 1)
        })
    
    df = pd.DataFrame(data)
    df.to_csv('sample_time_series.csv', index=False)
    print("Created sample_time_series.csv with 100 days of data")
    return df

def main():
    """Main function to demonstrate time series prediction"""
    print("="*60)
    print("TIME SERIES PREDICTION DEMO")
    print("="*60)
    
    # Create sample data
    print("\n1. Creating sample time series data...")
    df = create_sample_time_series_data()
    
    # Initialize system
    print("\n2. Initializing time series prediction system...")
    ts_system = TimeSeriesPredictionSystem(model_type="simple")
    
    # Load data
    ts_system.load_historical_data('sample_time_series.csv')
    
    # Train models
    print("3. Training prediction models...")
    ts_system.train_models()
    
    # Make predictions
    print("4. Predicting next 7 days...")
    predictions = ts_system.predict_future(days=7)
    
    # Display results
    print("\n" + "-"*60)
    print("7-DAY FORECAST")
    print("-"*60)
    
    for pred in predictions:
        date_str = pred.timestamp.strftime('%Y-%m-%d')
        print(f"{date_str}:")
        print(f"  Chlorophyll: {pred.chlorophyll_pred:.2f} mg/m3")
        print(f"  Pollution: {pred.pollution_level}")
        print(f"  Confidence: {pred.confidence:.1%}")
        print(f"  Trend: {pred.trend}")
        print()
    
    # Save predictions
    output_data = []
    for pred in predictions:
        output_data.append({
            'date': pred.timestamp,
            'chlorophyll': pred.chlorophyll_pred,
            'pollution_level': pred.pollution_level,
            'confidence': pred.confidence,
            'trend': pred.trend
        })
    
    forecast_df = pd.DataFrame(output_data)
    forecast_df.to_csv('7_day_forecast.csv', index=False)
    print("Forecast saved to: 7_day_forecast.csv")
    
    print("="*60)
    print("DEMO COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
