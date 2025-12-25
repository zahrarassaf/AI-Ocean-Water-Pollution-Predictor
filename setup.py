# setup.py
from setuptools import setup, find_packages

setup(
    name="marine-pollution-predictor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "xarray>=2023.1.0",
        "netcdf4>=1.6.0",
        "dask>=2023.1.0",
        "joblib>=1.2.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "prometheus-client>=0.17.0",
        "python-multipart>=0.0.6",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "httpx>=0.24.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "pre-commit>=3.3.0"
        ],
        "full": [
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
            "streamlit>=1.25.0"
        ]
    },
    python_requires=">=3.9",
)
