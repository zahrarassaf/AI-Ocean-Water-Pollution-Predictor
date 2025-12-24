#!/usr/bin/env python3
"""
deploy_model.py - Simple model deployment for marine pollution prediction.
Production-ready API with monitoring and logging.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import traceback

import numpy as np
import pandas as pd
import joblib

# Try to import FastAPI, fallback if not available
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not installed. API features disabled.")
    print("Install with: pip install fastapi uvicorn pydantic")

# Try to import monitoring, fallback if not available
try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not installed. Monitoring disabled.")
    print("Install with: pip install prometheus-client")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelServer:
    """Production model server with monitoring and logging."""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self.feature_names = []
        self.scaler = None
        
        # Initialize metrics if available
        if PROMETHEUS_AVAILABLE:
            self._init_metrics()
        else:
            self._init_simple_metrics()
        
        # Load model
        self.load_model()
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        self.request_counter = Counter(
            'model_requests_total',
            'Total number of prediction requests'
        )
        
        self.prediction_histogram = Histogram(
            'model_prediction_seconds',
            'Prediction latency in seconds',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        self.error_counter = Counter(
            'model_errors_total',
            'Total number of prediction errors',
            ['error_type']
        )
        
        self.feature_gauge = Gauge(
            'model_features_total',
            'Number of features in the model'
        )
        
        self.model_version_gauge = Gauge(
            'model_version_info',
            'Model version information',
            ['model_name', 'version']
        )
    
    def _init_simple_metrics(self):
        """Initialize simple metrics when Prometheus is not available."""
        self.request_counter = 0
        self.error_counter = {'invalid_features': 0, 'prediction_error': 0}
        self.prediction_times = []
    
    def load_model(self):
        """Load model and metadata from file."""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load model
            model_data = joblib.load(self.model_path)
            
            self.model = model_data.get('model')
            if self.model is None:
                raise ValueError("Model not found in the loaded data")
            
            self.metadata = model_data.get('metadata', {})
            self.feature_names = self.metadata.get('feature_names', [])
            self.scaler = model_data.get('scaler', None)
            
            # Update metrics if available
            if PROMETHEUS_AVAILABLE:
                self.feature_gauge.set(len(self.feature_names))
                model_name = self.metadata.get('model_name', 'unknown')
                version = self.metadata.get('creation_time', 'unknown')
                self.model_version_gauge.labels(model_name=model_name, version=version)
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Model type: {type(self.model).__name__}")
            logger.info(f"Features: {len(self.feature_names)}")
            logger.info(f"Model created: {self.metadata.get('creation_time', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def validate_features(self, features: Dict[str, float]) -> Tuple[bool, List[str], List[str]]:
        """Validate input features."""
        missing_features = []
        extra_features = []
        
        # Check for missing features
        for feature in self.feature_names:
            if feature not in features:
                missing_features.append(feature)
        
        # Check for extra features (warn but don't fail)
        for feature in features.keys():
            if feature not in self.feature_names:
                extra_features.append(feature)
        
        is_valid = len(missing_features) == 0
        return is_valid, missing_features, extra_features
    
    def preprocess_features(self, features: Dict[str, float]) -> np.ndarray:
        """Preprocess input features for prediction."""
        # Convert to array in correct order
        feature_array = np.array([features.get(f, 0.0) for f in self.feature_names]).reshape(1, -1)
        
        # Apply scaling if available
        if self.scaler is not None:
            try:
                feature_array = self.scaler.transform(feature_array)
            except Exception as e:
                logger.warning(f"Scaling failed: {e}")
        
        return feature_array
    
    def predict(self, features: Dict[str, float], return_uncertainty: bool = False) -> Dict[str, Any]:
        """Make prediction with comprehensive output."""
        start_time = time.time()
        
        try:
            # Update request counter
            if PROMETHEUS_AVAILABLE:
                self.request_counter.inc()
            else:
                self.request_counter += 1
            
            # Validate features
            is_valid, missing, extra = self.validate_features(features)
            if not is_valid:
                if PROMETHEUS_AVAILABLE:
                    self.error_counter.labels(error_type='invalid_features').inc()
                else:
                    self.error_counter['invalid_features'] += 1
                raise ValueError(f"Missing required features: {missing}")
            
            # Warn about extra features
            if extra:
                logger.warning(f"Extra features provided (ignored): {extra}")
            
            # Preprocess features
            X = self.preprocess_features(features)
            
            # Make prediction
            if PROMETHEUS_AVAILABLE:
                with self.prediction_histogram.time():
                    prediction = self.model.predict(X)[0]
            else:
                prediction_start = time.time()
                prediction = self.model.predict(X)[0]
                self.prediction_times.append(time.time() - prediction_start)
            
            # Calculate uncertainty if requested
            uncertainty = None
            if return_uncertainty:
                uncertainty = self.calculate_uncertainty(X, prediction)
            
            # Calculate prediction latency
            latency = time.time() - start_time
            
            # Prepare response
            response = {
                'success': True,
                'prediction': float(prediction),
                'features_used': self.feature_names,
                'feature_count': len(self.feature_names),
                'model_name': self.metadata.get('model_name', 'unknown'),
                'model_version': self.metadata.get('creation_time', 'unknown'),
                'prediction_time': latency,
                'timestamp': datetime.now().isoformat()
            }
            
            if uncertainty is not None:
                response.update({
                    'uncertainty': float(uncertainty),
                    'prediction_interval': {
                        'lower': float(prediction - 1.96 * uncertainty),
                        'upper': float(prediction + 1.96 * uncertainty)
                    },
                    'confidence_level': 0.95
                })
            
            logger.info(f"Prediction made: {prediction:.4f} (latency: {latency:.3f}s)")
            
            return response
            
        except Exception as e:
            if PROMETHEUS_AVAILABLE:
                self.error_counter.labels(error_type='prediction_error').inc()
            else:
                self.error_counter['prediction_error'] += 1
            logger.error(f"Prediction error: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def calculate_uncertainty(self, X: np.ndarray, prediction: float) -> float:
        """Calculate prediction uncertainty."""
        try:
            # Method 1: Use model's internal uncertainty estimation
            if hasattr(self.model, 'estimators_'):
                # For ensemble models like RandomForest
                tree_predictions = []
                for tree in self.model.estimators_:
                    try:
                        tree_pred = tree.predict(X)[0]
                        tree_predictions.append(tree_pred)
                    except:
                        continue
                
                if tree_predictions:
                    uncertainty = np.std(tree_predictions)
                    return float(uncertainty)
            
            # Method 2: Use prediction residuals from training
            if 'training_metrics' in self.metadata:
                residuals_std = self.metadata['training_metrics'].get('residual_std', 0.1)
                return float(residuals_std)
            
            # Default uncertainty (10% of absolute prediction)
            return float(abs(prediction) * 0.1)
            
        except Exception as e:
            logger.warning(f"Uncertainty calculation failed: {e}")
            return 0.1
    
    def batch_predict(self, features_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        results = []
        
        for i, features in enumerate(features_list):
            try:
                result = self.predict(features, return_uncertainty=True)
                result['request_id'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e),
                    'request_id': i,
                    'features': list(features.keys())
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_name': self.metadata.get('model_name', 'unknown'),
            'model_type': type(self.model).__name__,
            'model_version': self.metadata.get('creation_time', 'unknown'),
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'training_samples': self.metadata.get('training_samples', 'unknown'),
            'performance': self.metadata.get('performance', {}),
            'created_at': self.metadata.get('creation_time', 'unknown'),
            'loaded_at': datetime.now().isoformat()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        if PROMETHEUS_AVAILABLE:
            return {
                'requests_total': self.request_counter._value.get(),
                'errors_total': {
                    'invalid_features': self.error_counter.labels(error_type='invalid_features')._value.get(),
                    'prediction_error': self.error_counter.labels(error_type='prediction_error')._value.get()
                }
            }
        else:
            avg_prediction_time = np.mean(self.prediction_times) if self.prediction_times else 0
            return {
                'requests_total': self.request_counter,
                'errors_total': self.error_counter,
                'average_prediction_time': avg_prediction_time,
                'prediction_count': len(self.prediction_times)
            }

# API Models (only if FastAPI is available)
if FASTAPI_AVAILABLE:
    class PredictionRequest(BaseModel):
        """Pydantic model for prediction request validation."""
        features: Dict[str, float] = Field(..., description="Feature values for prediction")
        return_uncertainty: bool = Field(False, description="Whether to return uncertainty estimate")
        
        class Config:
            schema_extra = {
                "example": {
                    "features": {
                        "CHL": 0.5,
                        "PP": 100.0,
                        "KD490": 0.1
                    },
                    "return_uncertainty": True
                }
            }
    
    class BatchPredictionRequest(BaseModel):
        """Pydantic model for batch prediction request."""
        predictions: List[PredictionRequest] = Field(..., description="List of prediction requests")
        
        class Config:
            schema_extra = {
                "example": {
                    "predictions": [
                        {
                            "features": {"CHL": 0.5, "PP": 100.0},
                            "return_uncertainty": True
                        },
                        {
                            "features": {"CHL": 0.3, "PP": 80.0},
                            "return_uncertainty": False
                        }
                    ]
                }
            }
    
    class ModelResponse(BaseModel):
        """Standardized model response."""
        success: bool
        prediction: Optional[float] = None
        uncertainty: Optional[float] = None
        prediction_interval: Optional[Dict[str, float]] = None
        model_info: Dict[str, Any]
        timestamp: str
        latency: float
        
        class Config:
            schema_extra = {
                "example": {
                    "success": True,
                    "prediction": 0.123,
                    "uncertainty": 0.012,
                    "prediction_interval": {"lower": 0.099, "upper": 0.147},
                    "model_info": {
                        "model_name": "RandomForest",
                        "feature_count": 25
                    },
                    "timestamp": "2024-01-01T12:00:00Z",
                    "latency": 0.123
                }
            }

# Create FastAPI application (if available)
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Marine Pollution Prediction API",
        description="API for predicting marine pollution levels using machine learning",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add Prometheus middleware if available
    if PROMETHEUS_AVAILABLE:
        try:
            from starlette_exporter import PrometheusMiddleware
            app.add_middleware(PrometheusMiddleware)
            app.add_route("/metrics", prometheus_client.make_asgi_app())
        except ImportError:
            logger.warning("starlette_exporter not available. Metrics endpoint disabled.")
    
    # Global model server instance
    model_server = None
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize model server on startup."""
        global model_server
        try:
            # Load model from environment variable or default
            model_path_str = os.getenv('MODEL_PATH', 'models/best_model.joblib')
            model_path = Path(model_path_str)
            model_server = ModelServer(model_path)
            logger.info("Model server initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model server: {e}")
            raise
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Marine Pollution Prediction API",
            "version": "1.0.0",
            "status": "operational" if model_server else "initializing",
            "endpoints": {
                "health": "/health",
                "model_info": "/model/info",
                "predict": "/predict",
                "batch_predict": "/predict/batch",
                "metrics": "/metrics" if PROMETHEUS_AVAILABLE else None,
                "docs": "/docs"
            }
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        if model_server is None:
            return {
                "status": "unhealthy",
                "message": "Model server not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Simple health check with test features
            test_features = {feature: 0.0 for feature in model_server.feature_names[:3]}
            model_server.predict(test_features)
            
            return {
                "status": "healthy",
                "model_loaded": True,
                "feature_count": len(model_server.feature_names),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    @app.get("/model/info")
    async def get_model_info():
        """Get model information endpoint."""
        if model_server is None:
            raise HTTPException(status_code=503, detail="Model server not initialized")
        
        return model_server.get_model_info()
    
    @app.post("/predict", response_model=ModelResponse)
    async def predict(request: PredictionRequest):
        """Single prediction endpoint."""
        if model_server is None:
            raise HTTPException(status_code=503, detail="Model server not initialized")
        
        start_time = time.time()
        
        try:
            result = model_server.predict(
                request.features, 
                return_uncertainty=request.return_uncertainty
            )
            
            response = ModelResponse(
                success=True,
                prediction=result.get('prediction'),
                uncertainty=result.get('uncertainty'),
                prediction_interval=result.get('prediction_interval'),
                model_info={
                    'model_name': result.get('model_name', 'unknown'),
                    'feature_count': result.get('feature_count', 0)
                },
                timestamp=result.get('timestamp'),
                latency=result.get('prediction_time', 0.0)
            )
            
            return response
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    @app.post("/predict/batch")
    async def batch_predict(request: BatchPredictionRequest):
        """Batch prediction endpoint."""
        if model_server is None:
            raise HTTPException(status_code=503, detail="Model server not initialized")
        
        start_time = time.time()
        
        try:
            # Extract features from requests
            features_list = []
            for pred_request in request.predictions:
                features_list.append({
                    'features': pred_request.features,
                    'return_uncertainty': pred_request.return_uncertainty
                })
            
            # Make batch predictions
            results = model_server.batch_predict(features_list)
            
            total_time = time.time() - start_time
            
            return {
                'success': True,
                'predictions': results,
                'count': len(results),
                'successful': sum(1 for r in results if r.get('success', False)),
                'failed': sum(1 for r in results if not r.get('success', True)),
                'total_time': total_time,
                'average_time': total_time / len(results) if len(results) > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    @app.get("/features")
    async def get_features():
        """Get list of required features."""
        if model_server is None:
            raise HTTPException(status_code=503, detail="Model server not initialized")
        
        return {
            'features': model_server.feature_names,
            'count': len(model_server.feature_names),
            'description': 'Features required for prediction',
            'example': {feature: 0.0 for feature in model_server.feature_names[:5] if model_server.feature_names}
        }
    
    @app.get("/metrics/simple")
    async def get_simple_metrics():
        """Get simple metrics (fallback when Prometheus not available)."""
        if model_server is None:
            raise HTTPException(status_code=503, detail="Model server not initialized")
        
        return model_server.get_metrics()

def run_server(host: str = "0.0.0.0", port: int = 8000, model_path: Path = None):
    """Run the FastAPI server."""
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI is not installed. Cannot start server.")
        print("Please install with: pip install fastapi uvicorn pydantic")
        return
    
    global model_server
    
    # Initialize model server
    try:
        if model_path:
            model_server = ModelServer(model_path)
        else:
            # Try default path
            default_path = Path("models/best_model.joblib")
            if default_path.exists():
                model_server = ModelServer(default_path)
            else:
                logger.warning("No model file found. Starting server without model.")
                model_server = None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_server = None
    
    # Start server
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"API Documentation: http://{host}:{port}/docs")
    if PROMETHEUS_AVAILABLE:
        logger.info(f"Metrics: http://{host}:{port}/metrics")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

def main():
    """Main entry point for deployment script."""
    parser = argparse.ArgumentParser(
        description='Marine Pollution Prediction - Model Deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run API server with default model
  python deploy_model.py serve
  
  # Run API server with specific model
  python deploy_model.py serve --model-path models/random_forest_20240101.joblib
  
  # Make single prediction
  python deploy_model.py predict --features '{"CHL": 0.5, "PP": 100.0}' --model-path models/model.joblib
  
  # Make batch prediction from CSV
  python deploy_model.py batch-predict --input data/predict.csv --model-path models/model.joblib
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute', required=True)
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    serve_parser.add_argument('--port', type=int, default=8000, help='Server port')
    serve_parser.add_argument('--model-path', type=Path, help='Path to model file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make single prediction')
    predict_parser.add_argument('--features', type=str, required=True, help='JSON string of features')
    predict_parser.add_argument('--model-path', type=Path, required=True, help='Path to model file')
    predict_parser.add_argument('--uncertainty', action='store_true', help='Return uncertainty estimate')
    
    # Batch predict command
    batch_parser = subparsers.add_parser('batch-predict', help='Make batch predictions')
    batch_parser.add_argument('--input', type=Path, required=True, help='Input CSV file')
    batch_parser.add_argument('--output', type=Path, help='Output JSON file')
    batch_parser.add_argument('--model-path', type=Path, required=True, help='Path to model file')
    
    args = parser.parse_args()
    
    if args.command == 'serve':
        if not FASTAPI_AVAILABLE:
            print("Error: FastAPI is required for serving. Install with:")
            print("pip install fastapi uvicorn pydantic")
            sys.exit(1)
        run_server(args.host, args.port, args.model_path)
    
    elif args.command == 'predict':
        try:
            features = json.loads(args.features)
            server = ModelServer(args.model_path)
            result = server.predict(features, args.uncertainty)
            
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
    
    elif args.command == 'batch-predict':
        try:
            # Load data
            if not args.input.exists():
                print(f"Error: Input file not found: {args.input}")
                sys.exit(1)
            
            df = pd.read_csv(args.input)
            
            # Initialize model server
            server = ModelServer(args.model_path)
            
            # Convert to features list
            features_list = df.to_dict('records')
            
            # Make predictions
            results = server.batch_predict(features_list)
            
            # Save results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to: {args.output}")
            else:
                print(json.dumps(results, indent=2))
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
