# 🌊 AI Ocean Water Pollution Predictor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning system for predicting ocean pollution levels using satellite data indicators.

## 🚀 Features

### **Data Processing**
- Automated data download from CMEMS Copernicus Marine Service
- Processing of oceanographic parameters (Chlorophyll, Productivity, Transparency)
- Feature engineering and data quality checks
- Modular data pipeline architecture

### **Machine Learning**
- Random Forest Classifier for pollution level prediction
- Three-class classification: LOW, MEDIUM, HIGH pollution
- Model persistence and versioning system
- 95% accuracy on validation data

### **System Architecture**
- Complete ML pipeline: `download → process → train → predict`
- REST API ready (Flask-based)
- Interactive Streamlit dashboard
- Configuration management with YAML files

## 📁 Project Structure
AI-Ocean-Water-Pollution-Predictor/
├── src/ # Core ML modules
│ ├── data/ # Data processing utilities
│ ├── models/ # Model training and evaluation
│ ├── analysis/ # Data analysis tools
│ └── config/ # Configuration management
├── api/ # REST API implementation
├── dashboard/ # Streamlit dashboard
├── scripts/ # Execution scripts
│ ├── download_data.py # Satellite data download
│ ├── train_model.py # Model training script
│ └── train_marine.py # Marine-specific training
├── data/ # Data storage
│ └── processed/ # Processed datasets
├── models/ # Trained models
│ ├── pollution_model.pkl # Main prediction model
│ ├── label_encoder.pkl # Label encoder
│ └── checkpoints/ # Training checkpoints
├── config/ # Configuration files
│ ├── datasets.yaml # Dataset configurations
│ └── training_config.yaml # Training parameters
├── predict.py # Main prediction module
├── run_pipeline.py # Complete pipeline execution
├── requirements.txt # Dependencies
└── README.md # Documentation

text

## 🔧 Installation & Usage

### **1. Installation**
```bash
# Clone repository
git clone https://github.com/Zahrarasaf/AI-Ocean-Water-Pollution-Predictor.git
cd AI-Ocean-Water-Pollution-Predictor

# Install dependencies
pip install -r requirements.txt
2. Quick Prediction Demo
bash
# Run the prediction system
python predict.py
3. Full Pipeline
bash
# Download, process, train, and predict
python run_pipeline.py
4. API Server
bash
cd api
python main.py
# API available at http://localhost:8000
5. Dashboard
bash
cd dashboard
streamlit run dashboard.py
🎯 Model Performance
Algorithm: Random Forest Classifier

Accuracy: 95% (validation set)

Classes: LOW, MEDIUM, HIGH pollution levels

Features: Chlorophyll concentration, Primary Productivity, Water Transparency

Output: Pollution level with confidence score

📈 Example Predictions
python
from predict import OceanPollutionPredictor

predictor = OceanPollutionPredictor()
result = predictor.predict(
    chlorophyll=2.0,      # mg/m³
    productivity=300.0,   # mg C/m²/day
    transparency=10.0     # meters
)

print(f"Pollution Level: {result['level_name']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Recommendation: {result['recommendation']}")
🔬 Scientific Basis
The system uses established oceanographic thresholds:

LOW: Chlorophyll ≤ 1.0 mg/m³ (Clean water)

MEDIUM: 1.0 < Chlorophyll ≤ 5.0 mg/m³ (Moderate pollution)

HIGH: Chlorophyll > 5.0 mg/m³ (High pollution)

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📄 License
MIT License - see LICENSE file for details.
