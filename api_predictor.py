from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional
import uvicorn

# Define request/response models
class WaterQualitySample(BaseModel):
    sea_surface_temp: float = Field(..., ge=-5, le=40)
    salinity: float = Field(..., ge=0, le=50)
    turbidity: float = Field(..., ge=0)
    ph: float = Field(..., ge=6, le=9)
    dissolved_oxygen: float = Field(..., ge=0)
    nitrate: float = Field(..., ge=0)
    phosphate: float = Field(..., ge=0)
    ammonia: float = Field(..., ge=0)
    chlorophyll_a: float = Field(..., ge=0)
    sechi_depth: float = Field(..., ge=0)
    lead: float = Field(..., ge=0)
    mercury: float = Field(..., ge=0)
    cadmium: float = Field(..., ge=0)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    month: int = Field(..., ge=1, le=12)

class PredictionResponse(BaseModel):
    pollution_level: int
    confidence: float
    probabilities: dict
    risk_level: str
    recommended_action: str

class BatchPredictionRequest(BaseModel):
    samples: List[WaterQualitySample]

class BatchPredictionResponse(BaseModel):
    predictions: List[dict]
    statistics: dict
    timestamp: str

# Initialize FastAPI app
app = FastAPI(
    title="Ocean Pollution Prediction API",
    description="AI-powered API for predicting ocean water pollution levels",
    version="1.0.0"
)

class APIPredictor:
    def __init__(self):
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            self.predictor = OceanPollutionPredictor()
            print("✅ API Predictor initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize predictor: {e}")
            self.predictor = None
    
    def is_ready(self):
        """Check if predictor is ready"""
        return self.predictor is not None

# Global predictor instance
predictor = APIPredictor()

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "Ocean Pollution Prediction API",
        "version": "1.0.0",
        "status": "operational" if predictor.is_ready() else "initializing",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if predictor.is_ready():
        return {"status": "healthy", "model_loaded": True}
    else:
        raise HTTPException(status_code=503, detail="Model not loaded")

@app.post("/predict", response_model=PredictionResponse)
async def predict_pollution(sample: WaterQualitySample):
    """
    Predict pollution level for a single water quality sample.
    
    Args:
        sample: Water quality parameters
        
    Returns:
        Prediction results with confidence scores
    """
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Predictor not ready")
    
    try:
        # Convert to dict
        sample_dict = sample.dict()
        
        # Make prediction
        result = predictor.predictor.predict(sample_dict)
        
        # Format response
        response = PredictionResponse(
            pollution_level=result['pollution_level'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            risk_level=result['risk_assessment']['level'],
            recommended_action=result['risk_assessment']['action']
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict pollution levels for multiple samples.
    
    Args:
        request: List of water quality samples
        
    Returns:
        Batch prediction results with statistics
    """
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Predictor not ready")
    
    try:
        # Convert to DataFrame
        samples = [s.dict() for s in request.samples]
        df = pd.DataFrame(samples)
        
        # Make predictions
        predictions = []
        for _, row in df.iterrows():
            result = predictor.predictor.predict(row.to_dict())
            predictions.append({
                "pollution_level": result['pollution_level'],
                "confidence": result['confidence'],
                "risk_level": result['risk_assessment']['level']
            })
        
        # Calculate statistics
        pollution_levels = [p['pollution_level'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        statistics = {
            "total_samples": len(predictions),
            "average_confidence": float(np.mean(confidences)),
            "level_distribution": {
                "low": pollution_levels.count(0),
                "medium": pollution_levels.count(1),
                "high": pollution_levels.count(2)
            }
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            statistics=statistics,
            timestamp=pd.Timestamp.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model_info")
async def get_model_info():
    """Get information about the trained model"""
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Predictor not ready")
    
    return {
        "model_type": type(predictor.predictor.model).__name__,
        "feature_count": len(predictor.predictor.feature_names),
        "features": predictor.predictor.feature_names,
        "supported_levels": ["low (0)", "medium (1)", "high (2)"]
    }

# Run the API server
if __name__ == "__main__":
    print("🚀 Starting Ocean Pollution Prediction API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
