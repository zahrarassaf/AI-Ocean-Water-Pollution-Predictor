"""
Environment-specific settings
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings"""
    
    # Environment
    ENV = os.getenv('ENVIRONMENT', 'development')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Paths
    DATA_PATH = os.getenv('DATA_PATH', 'data/')
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/')
    LOG_PATH = os.getenv('LOG_PATH', 'logs/')
    
    # Processing
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', os.cpu_count()))
    USE_GPU = os.getenv('USE_GPU', 'True').lower() == 'true'
    PRECISION = os.getenv('PRECISION', 'mixed')  # mixed, float32, float16
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # API
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 8000))
    
    @classmethod
    def validate(cls):
        """Validate all settings"""
        required = ['DATA_PATH', 'MODEL_PATH']
        for var in required:
            if not getattr(cls, var):
                raise ValueError(f"{var} must be set in environment")
    
    @classmethod
    def get_device(cls):
        """Get appropriate torch device"""
        import torch
        if cls.USE_GPU and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

settings = Settings()
settings.validate()
