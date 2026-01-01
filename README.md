# 🌊 AI Ocean Water Pollution Prediction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52+-red.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://docker.com)

A comprehensive machine learning system for real-time ocean pollution prediction using satellite data. Achieves **95.18% accuracy** in classifying pollution levels.

## 🎯 Features

- **Real-time Prediction**: Instant pollution level classification (LOW/MEDIUM/HIGH)
- **Interactive Dashboard**: Streamlit-based interface with visual analytics
- **Batch Processing**: Analyze multiple samples via CSV upload
- **Model Insights**: Feature importance, performance metrics, confusion matrix
- **Docker Support**: Containerized deployment ready for production
- **REST API**: Flask-based API for integration with other systems

## 🚀 Quick Start

### Local Installation
```bash
git clone https://github.com/Zahrarasaf/AI-Ocean-Water-Pollution-Predictor.git
cd AI-Ocean-Water-Pollution-Predictor
pip install -r requirements.txt
streamlit run dashboard/dashboard.py
Docker Deployment
bash
docker build -t ocean-pollution .
docker run -p 8501:8501 ocean-pollution
📊 Results
Accuracy: 95.18%

Data Processed: 2.3M+ satellite samples

Model: Random Forest Classifier

Data Source: CMEMS Copernicus Marine Service

🏗️ Architecture
text
Satellite Data → Data Processing → Feature Engineering → ML Model → Dashboard/API
📁 Project Structure
text
├── dashboard/          # Streamlit interactive dashboard
├── src/               # Core ML modules
├── api/               # Flask REST API
├── Dockerfile         # Container configuration
├── docker-compose.yml # Service orchestration
└── requirements.txt   # Dependencies
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📄 License
MIT License - see LICENSE file for details.

🙏 Acknowledgments
CMEMS Copernicus Marine Service for satellite data

NASA Ocean Biology Processing Group

Open-source communities for Python libraries
