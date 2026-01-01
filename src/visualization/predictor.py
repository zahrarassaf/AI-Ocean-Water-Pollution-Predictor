import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import joblib
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PollutionPredictor:
    def __init__(self, model_path: str = "models/"):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and preprocessing objects."""
        try:
            self.model = joblib.load(self.model_path / "pollution_model.pkl")
            self.scaler = joblib.load(self.model_path / "scaler.pkl")
            self.label_encoder = joblib.load(self.model_path / "label_encoder.pkl")
            self.metadata = joblib.load(self.model_path / "model_metadata.pkl")
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_input(self, input_data: pd.DataFrame) -> np.ndarray:
        """Preprocess input data for prediction."""
        # Ensure all required features are present
        required_features = self.metadata['features']
        
        for feature in required_features:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        # Select and order features correctly
        input_data = input_data[required_features]
        
        # Scale features
        input_scaled = self.scaler.transform(input_data)
        
        return input_scaled
    
    def predict(self, input_data: pd.DataFrame) -> Dict:
        """Make predictions on new data."""
        try:
            # Preprocess input
            X_processed = self.preprocess_input(input_data)
            
            # Make predictions
            predictions = self.model.predict(X_processed)
            probabilities = self.model.predict_proba(X_processed)
            
            # Decode labels
            decoded_predictions = self.label_encoder.inverse_transform(predictions)
            
            # Create detailed results
            results = []
            for i, (pred, prob) in enumerate(zip(decoded_predictions, probabilities)):
                confidence = np.max(prob)
                prob_dict = {
                    cls: float(p) 
                    for cls, p in zip(self.label_encoder.classes_, prob)
                }
                
                results.append({
                    'prediction': pred,
                    'confidence': confidence,
                    'probabilities': prob_dict,
                    'risk_level': self._get_risk_level(pred, confidence)
                })
            
            return {
                'predictions': results,
                'summary': self._create_summary(results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _get_risk_level(self, prediction: str, confidence: float) -> str:
        """Determine risk level based on prediction and confidence."""
        risk_map = {
            'LOW': 'LOW_RISK',
            'MEDIUM': 'MEDIUM_RISK',
            'HIGH': 'HIGH_RISK',
            'CRITICAL': 'CRITICAL_RISK'
        }
        
        base_risk = risk_map.get(prediction, 'UNKNOWN')
        
        if confidence < 0.7:
            return f"{base_risk}_LOW_CONFIDENCE"
        
        return base_risk
    
    def _create_summary(self, results: List[Dict]) -> Dict:
        """Create summary statistics from predictions."""
        predictions = [r['prediction'] for r in results]
        
        summary = {
            'total_samples': len(results),
            'prediction_distribution': pd.Series(predictions).value_counts().to_dict(),
            'average_confidence': np.mean([r['confidence'] for r in results]),
            'high_risk_count': sum(1 for r in results 
                                 if r['prediction'] in ['HIGH', 'CRITICAL']),
            'low_risk_count': sum(1 for r in results 
                                if r['prediction'] in ['LOW', 'MEDIUM']),
            'risk_assessment': self._assess_overall_risk(results)
        }
        
        return summary
    
    def _assess_overall_risk(self, results: List[Dict]) -> str:
        """Assess overall risk based on predictions."""
        high_risk_pct = sum(1 for r in results 
                          if r['prediction'] in ['HIGH', 'CRITICAL']) / len(results)
        
        if high_risk_pct > 0.3:
            return "CRITICAL - Immediate action required"
        elif high_risk_pct > 0.1:
            return "HIGH - Monitor closely"
        else:
            return "NORMAL - Regular monitoring sufficient"
    
    def forecast_timeseries(self, historical_data: pd.DataFrame, 
                          days_ahead: int = 7) -> pd.DataFrame:
        """Generate time series forecast."""
        forecasts = []
        current_date = datetime.now()
        
        for day in range(days_ahead):
            forecast_date = current_date + timedelta(days=day)
            
            # Use historical patterns for forecast
            # This is a simplified version - extend with proper time series model
            base_features = historical_data.iloc[-1].copy()
            
            # Add seasonal/temporal adjustments
            day_of_year = forecast_date.timetuple().tm_yday
            seasonal_factor = np.sin(2 * np.pi * day_of_year / 365) * 0.1
            
            adjusted_features = base_features * (1 + seasonal_factor)
            
            # Make prediction
            prediction_result = self.predict(
                pd.DataFrame([adjusted_features])
            )
            
            forecast = {
                'date': forecast_date.date(),
                'timestamp': forecast_date.isoformat(),
                'predicted_level': prediction_result['predictions'][0]['prediction'],
                'confidence': prediction_result['predictions'][0]['confidence'],
                'risk_level': prediction_result['predictions'][0]['risk_level'],
                'features': adjusted_features.to_dict()
            }
            
            forecasts.append(forecast)
        
        return pd.DataFrame(forecasts)
    
    def save_forecast(self, forecast_df: pd.DataFrame, 
                     output_path: str = "data/forecasts/"):
        """Save forecast to CSV."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"pollution_forecast_{timestamp}.csv"
        
        forecast_df.to_csv(filename, index=False)
        logger.info(f"Forecast saved to {filename}")
        
        return filename
