# api.py
# FastAPI application for Ocean Pollution Prediction
# Run with: uvicorn api:app --reload --host 0.0.0.0 --port 8000

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
import joblib
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from enum import Enum

# Create FastAPI app
app = FastAPI(
    title="🌊 Ocean Pollution Prediction API",
    description="REST API for predicting ocean pollution levels using machine learning",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enums
class PollutionLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

# Pydantic models
class FeatureInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "CHL": 2.5,
            "PP": 350.0,
            "KD490": 0.25,
            "DIATO": 0.03,
            "DINO": 0.02
        }
    })
    
    CHL: float = Field(..., description="Chlorophyll concentration in mg/m³", ge=0, le=50)
    PP: Optional[float] = Field(None, description="Primary Production", ge=0)
    KD490: Optional[float] = Field(None, description="Diffuse attenuation coefficient at 490nm", ge=0)
    DIATO: Optional[float] = Field(None, description="Diatom concentration", ge=0)
    DINO: Optional[float] = Field(None, description="Dinoflagellate concentration", ge=0)
    CDM: Optional[float] = Field(None, description="Colored dissolved organic matter", ge=0)
    BBP: Optional[float] = Field(None, description="Particulate backscattering coefficient", ge=0)

class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "samples": [
                {"CHL": 0.5},
                {"CHL": 2.5, "PP": 300},
                {"CHL": 8.0, "KD490": 0.4}
            ]
        }
    })
    
    samples: List[FeatureInput]

class PredictionResult(BaseModel):
    level: PollutionLevel
    confidence: float = Field(..., ge=0, le=1)
    probabilities: Dict[PollutionLevel, float]
    chl_value: float
    features_used: int
    timestamp: datetime

class PredictionResponse(BaseModel):
    success: bool
    prediction: PredictionResult
    model_info: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    service: str = "ocean-pollution-api"

# Model predictor class
class OceanPollutionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = []
        self.classes = [PollutionLevel.LOW, PollutionLevel.MEDIUM, PollutionLevel.HIGH]
        self.model_info = {}
        self.feature_stats = {}
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📁 Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        self._load_model()
    
    def _load_model(self):
        """Load model from various possible locations"""
        try:
            # Check multiple possible locations
            possible_paths = [
                ('../models/ocean_model.pkl', '../models/ocean_scaler.pkl'),
                ('models/ocean_model.pkl', 'models/ocean_scaler.pkl'),
                ('api/models/ocean_model.pkl', 'api/models/ocean_scaler.pkl'),
            ]
            
            model_loaded = False
            
            for model_path, scaler_path in possible_paths:
                print(f"🔍 Checking path: {model_path}")
                if os.path.exists(model_path):
                    print(f"📦 Found model at: {model_path}")
                    try:
                        self.model = joblib.load(model_path)
                        self.scaler = joblib.load(scaler_path)
                        print(f"✅ Model loaded from: {model_path}")
                        model_loaded = True
                        
                        # Load features list
                        features_path = model_path.replace('ocean_model.pkl', 'features.txt')
                        if os.path.exists(features_path):
                            with open(features_path, 'r') as f:
                                self.features = [line.strip() for line in f.readlines()]
                            print(f"📋 Loaded {len(self.features)} features")
                        
                        # Load model metadata
                        metadata_path = model_path.replace('ocean_model.pkl', 'metadata.json')
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                self.model_info = json.load(f)
                            print(f"📄 Loaded model metadata")
                        
                        # Load feature statistics
                        stats_path = model_path.replace('ocean_model.pkl', 'feature_statistics.json')
                        if os.path.exists(stats_path):
                            with open(stats_path, 'r') as f:
                                self.feature_stats = json.load(f)
                            print(f"📊 Loaded feature statistics")
                        
                        break
                    except Exception as e:
                        print(f"❌ Error loading from {model_path}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            if not model_loaded:
                print("⚠️ No ML model found. Running in simulation mode.")
            
            print(f"📊 Total features: {len(self.features)}")
            print(f"🎯 Classes: {[cls.value for cls in self.classes]}")
            
        except Exception as e:
            print(f"❌ Error in model initialization: {e}")
            import traceback
            traceback.print_exc()
    
    def _prepare_features(self, input_data: Dict[str, float]) -> Dict[str, float]:
        """Prepare features for prediction"""
        # Extract CHL value
        chl_value = None
        for key, value in input_data.items():
            if 'CHL' in key.upper():
                chl_value = float(value)
                break
        
        if chl_value is None:
            raise ValueError("CHL value is required")
        
        # Prepare features dictionary
        prepared = {}
        
        for feature in self.features:
            feature_found = False
            
            # Check if feature is in input data
            for input_key, input_val in input_data.items():
                if feature.upper() == input_key.upper():
                    try:
                        prepared[feature] = float(input_val)
                        feature_found = True
                        break
                    except:
                        continue
            
            # If feature not found, use default value
            if not feature_found:
                if feature in self.feature_stats:
                    # Use median from statistics
                    prepared[feature] = self.feature_stats[feature].get('median', 0.0)
                else:
                    # Smart defaults based on feature name
                    feature_upper = feature.upper()
                    if 'CHL' in feature_upper and 'UNCERTAINTY' not in feature_upper:
                        prepared[feature] = chl_value
                    elif 'PP' in feature_upper:
                        prepared[feature] = chl_value * 100
                    elif 'KD' in feature_upper:
                        prepared[feature] = 0.1 + (chl_value * 0.05)
                    elif 'DIATO' in feature_upper or 'DINO' in feature_upper:
                        prepared[feature] = chl_value * 0.01
                    elif 'CDM' in feature_upper:
                        prepared[feature] = 0.01 + (chl_value * 0.002)
                    elif 'BBP' in feature_upper:
                        prepared[feature] = 0.001 + (chl_value * 0.0003)
                    else:
                        prepared[feature] = 0.0
        
        return prepared, chl_value
    
    def predict(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction for a single sample"""
        try:
            # Extract CHL for rule-based fallback
            chl_value = None
            for key, value in input_data.items():
                if 'CHL' in key.upper():
                    chl_value = float(value)
                    break
            
            if chl_value is None:
                return {
                    "success": False,
                    "error": "CHL value is required"
                }
            
            # If ML model is available, use it
            if self.model is not None and self.scaler is not None:
                try:
                    print(f"🤖 Using ML model for prediction with CHL={chl_value}")
                    # Prepare features
                    prepared_features, chl_value = self._prepare_features(input_data)
                    
                    # Create DataFrame
                    df = pd.DataFrame([prepared_features])
                    
                    # Ensure all features are present
                    missing_features = set(self.features) - set(df.columns)
                    for feat in missing_features:
                        df[feat] = 0.0
                    
                    # Reorder columns
                    df = df[self.features]
                    
                    # Scale features
                    scaled_features = self.scaler.transform(df)
                    
                    # Make prediction
                    prediction = self.model.predict(scaled_features)[0]
                    probabilities = self.model.predict_proba(scaled_features)[0]
                    
                    # Get confidence and probabilities
                    confidence = float(np.max(probabilities))
                    prob_dict = {
                        PollutionLevel.LOW: float(probabilities[0]),
                        PollutionLevel.MEDIUM: float(probabilities[1]),
                        PollutionLevel.HIGH: float(probabilities[2])
                    }
                    
                    level = self.classes[int(prediction)]
                    
                    result = {
                        "success": True,
                        "prediction": {
                            "level": level,
                            "confidence": confidence,
                            "probabilities": prob_dict,
                            "chl_value": chl_value,
                            "features_used": len(self.features),
                            "timestamp": datetime.now()
                        },
                        "model_info": {
                            "model_type": "ML Model",
                            "accuracy": self.model_info.get('accuracy', 'N/A'),
                            "features_count": len(self.features)
                        }
                    }
                    
                    print(f"✅ ML Prediction: {level.value} (confidence: {confidence:.2f})")
                    return result
                    
                except Exception as e:
                    print(f"⚠️ ML prediction failed, falling back to rule-based: {e}")
                    # Fall through to rule-based
        
            # Rule-based prediction (fallback)
            print(f"📝 Using rule-based prediction for CHL={chl_value}")
            return self._predict_rule_based(chl_value)
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _predict_rule_based(self, chl_value: float) -> Dict[str, Any]:
        """Rule-based prediction when ML model is not available"""
        if chl_value < 1.0:
            level = PollutionLevel.LOW
            confidence = 0.85
            probs = {
                PollutionLevel.LOW: 0.7,
                PollutionLevel.MEDIUM: 0.2,
                PollutionLevel.HIGH: 0.1
            }
        elif chl_value < 3.0:
            level = PollutionLevel.MEDIUM
            confidence = 0.75
            probs = {
                PollutionLevel.LOW: 0.2,
                PollutionLevel.MEDIUM: 0.6,
                PollutionLevel.HIGH: 0.2
            }
        else:
            level = PollutionLevel.HIGH
            confidence = 0.80
            probs = {
                PollutionLevel.LOW: 0.1,
                PollutionLevel.MEDIUM: 0.3,
                PollutionLevel.HIGH: 0.6
            }
        
        return {
            "success": True,
            "prediction": {
                "level": level,
                "confidence": confidence,
                "probabilities": probs,
                "chl_value": chl_value,
                "features_used": 1,
                "timestamp": datetime.now()
            },
            "model_info": {
                "model_type": "Rule-based",
                "description": "Simple threshold-based prediction"
            }
        }

# Initialize predictor
predictor = OceanPollutionPredictor()

# API Endpoints
@app.get("/", response_model=Dict[str, Any])
def root():
    """Root endpoint - API information"""
    return {
        "message": "🌊 Ocean Pollution Prediction API",
        "version": "2.0.0",
        "description": "Machine Learning API for ocean water quality assessment",
        "endpoints": {
            "documentation": "/docs",
            "health": "/health",
            "model_info": "/model-info",
            "features": "/features",
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch",
            "raw_prediction": "/predict/raw"
        },
        "contact": {
            "email": "support@ocean-pollution-api.com",
            "documentation": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "model_loaded": predictor.model is not None,
        "service": "ocean-pollution-api"
    }

@app.get("/model-info", response_model=Dict[str, Any])
def model_info():
    """Get model information"""
    return {
        "success": True,
        "model": {
            "loaded": predictor.model is not None,
            "type": "ML Model" if predictor.model else "Rule-based",
            "features_count": len(predictor.features),
            "classes": [cls.value for cls in predictor.classes]
        },
        "metadata": predictor.model_info,
        "files": {
            "model_exists": predictor.model is not None,
            "scaler_exists": predictor.scaler is not None,
            "features_count": len(predictor.features),
            "has_statistics": len(predictor.feature_stats) > 0
        }
    }

@app.get("/features", response_model=Dict[str, Any])
def get_features():
    """Get list of model features"""
    return {
        "success": True,
        "features": predictor.features,
        "total": len(predictor.features),
        "required": ["CHL"],
        "optional": ["PP", "KD490", "DIATO", "DINO", "CDM", "BBP"],
        "has_statistics": len(predictor.feature_stats) > 0
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_single(request: FeatureInput):
    """Single prediction endpoint"""
    # Convert to dictionary
    input_data = request.dict(exclude_none=True)
    print(f"📥 Received prediction request: {input_data}")
    
    # Make prediction
    result = predictor.predict(input_data)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))
    
    return PredictionResponse(
        success=True,
        prediction=PredictionResult(**result["prediction"]),
        model_info=result.get("model_info")
    )

@app.post("/predict/batch", response_model=Dict[str, Any])
def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    try:
        results = []
        
        for sample in request.samples:
            input_data = sample.dict(exclude_none=True)
            result = predictor.predict(input_data)
            
            results.append({
                "input": input_data,
                "success": result["success"],
                "prediction": result.get("prediction"),
                "error": result.get("error")
            })
        
        # Calculate statistics
        successful = sum(1 for r in results if r["success"])
        total = len(results)
        
        # Count pollution levels
        level_counts = {level.value: 0 for level in PollutionLevel}
        for r in results:
            if r["success"] and r["prediction"]:
                level = r["prediction"]["level"]
                level_counts[level.value] = level_counts.get(level.value, 0) + 1
        
        return {
            "success": True,
            "results": results,
            "statistics": {
                "total_samples": total,
                "successful_predictions": successful,
                "success_rate": successful / total if total > 0 else 0,
                "pollution_distribution": level_counts
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict/raw", response_model=Dict[str, Any])
def predict_raw(data: Dict[str, Any]):
    """Raw prediction endpoint with flexible input format"""
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")
    
    print(f"📥 Received raw prediction request: {data}")
    result = predictor.predict(data)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))
    
    return result

@app.get("/predict/sample", response_model=Dict[str, Any])
def get_sample_prediction():
    """Get a sample prediction with example data"""
    sample_data = {
        "CHL": 2.5,
        "PP": 350.0,
        "KD490": 0.25
    }
    
    result = predictor.predict(sample_data)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail="Sample prediction failed")
    
    return {
        "success": True,
        "sample_data": sample_data,
        "result": result
    }

@app.get("/system/info", response_model=Dict[str, Any])
def system_info():
    """Get system information"""
    import platform
    import sys
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "processor": platform.processor(),
        "working_directory": os.getcwd(),
        "api_directory": os.path.dirname(os.path.abspath(__file__)),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/project/structure", response_model=Dict[str, Any])
def project_structure():
    """Get project structure information"""
    structure = {}
    
    for item in os.listdir('.'):
        if os.path.isdir(item):
            try:
                files = []
                for file in os.listdir(item):
                    if file.endswith(('.py', '.pkl', '.txt', '.json', '.yaml', '.yml', '.nc')):
                        files.append(file)
                structure[item] = files[:10]  # Limit to 10 files per directory
            except:
                structure[item] = ["Access error or empty"]
    
    return {
        "current_directory": os.getcwd(),
        "structure": structure,
        "main_files": [f for f in os.listdir('.') if f.endswith('.py')]
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "success": False,
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return {
        "success": False,
        "error": str(exc),
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
    }

# Run the application - FIXED LINE
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",  
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
