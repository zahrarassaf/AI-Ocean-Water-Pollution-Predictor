import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.predictor import PollutionPredictor
from src.data_processor import OceanDataProcessor

class TestPollutionPredictor:
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'CHL': np.random.exponential(1, 100),
            'KD490': np.random.uniform(0.02, 1.5, 100),
            'PROCHLO': np.random.uniform(0.001, 1.0, 100),
            'lat': np.random.uniform(-90, 90, 100),
            'lon': np.random.uniform(-180, 180, 100)
        })
        
        return data
    
    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return PollutionPredictor()
    
    def test_predictor_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.label_encoder is not None
        assert predictor.metadata is not None
    
    def test_preprocess_input(self, predictor, sample_data):
        """Test input preprocessing."""
        processed = predictor.preprocess_input(sample_data)
        
        assert isinstance(processed, np.ndarray)
        assert processed.shape[0] == len(sample_data)
        assert processed.shape[1] == len(predictor.metadata['features'])
    
    def test_predict(self, predictor, sample_data):
        """Test prediction functionality."""
        results = predictor.predict(sample_data)
        
        assert 'predictions' in results
        assert 'summary' in results
        assert 'timestamp' in results
        
        predictions = results['predictions']
        assert len(predictions) == len(sample_data)
        
        for pred in predictions:
            assert 'prediction' in pred
            assert 'confidence' in pred
            assert 'probabilities' in pred
            assert 'risk_level' in pred
            
            assert pred['confidence'] >= 0
            assert pred['confidence'] <= 1
    
    def test_forecast_timeseries(self, predictor, sample_data):
        """Test time series forecast."""
        forecast_days = 7
        forecast_df = predictor.forecast_timeseries(
            sample_data, days_ahead=forecast_days
        )
        
        assert len(forecast_df) == forecast_days
        assert 'date' in forecast_df.columns
        assert 'predicted_level' in forecast_df.columns
        assert 'confidence' in forecast_df.columns
        assert 'risk_level' in forecast_df.columns
    
    def test_save_forecast(self, predictor, sample_data, tmp_path):
        """Test saving forecast to file."""
        forecast_df = predictor.forecast_timeseries(sample_data, days_ahead=3)
        
        output_path = tmp_path / "forecasts"
        filename = predictor.save_forecast(forecast_df, str(output_path))
        
        assert Path(filename).exists()
        
        # Verify saved content
        saved_df = pd.read_csv(filename)
        assert len(saved_df) == len(forecast_df)
        assert all(col in saved_df.columns for col in forecast_df.columns)

class TestOceanDataProcessor:
    @pytest.fixture
    def processor(self):
        """Create data processor instance."""
        return OceanDataProcessor("data/raw/")
    
    def test_clean_data(self, processor):
        """Test data cleaning functionality."""
        # Create sample data with outliers
        data = pd.DataFrame({
            'A': np.concatenate([np.random.normal(0, 1, 90), 
                                np.array([100, -100])]),  # Outliers
            'B': np.random.normal(10, 2, 92),
            'C': np.random.exponential(1, 92)
        })
        
        cleaned = processor.clean_data(data)
        
        assert len(cleaned) <= len(data)
        assert not cleaned.isnull().any().any()
        assert not np.isinf(cleaned).any().any()
    
    def test_create_target(self, processor):
        """Test target variable creation."""
        data = pd.DataFrame({
            'CHL': [0.5, 3.0, 10.0, 25.0, 0.1]
        })
        
        target = processor.create_target(data)
        
        expected_classes = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'LOW']
        assert list(target) == expected_classes
    
    def test_split_data(self, processor):
        """Test data splitting."""
        data = pd.DataFrame({
            'CHL': np.random.exponential(1, 1000),
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.uniform(0, 1, 1000)
        })
        
        target = processor.create_target(data)
        features = data[['feature1', 'feature2']]
        
        splits = processor.split_data(features, target, test_size=0.2, val_size=0.1)
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        
        # Check sizes
        total_len = len(features)
        assert len(X_test) == int(total_len * 0.2)
        assert len(X_val) == int(total_len * 0.1)
        assert len(X_train) == total_len - len(X_test) - len(X_val)
        
        # Check no overlap
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        test_indices = set(X_test.index)
        
        assert train_indices.isdisjoint(val_indices)
        assert train_indices.isdisjoint(test_indices)
        assert val_indices.isdisjoint(test_indices)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
