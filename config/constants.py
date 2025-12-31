import numpy as np
from enum import Enum

class PollutionLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class Thresholds:
    CHLOROPHYLL_LOW = 1.0
    CHLOROPHYLL_MEDIUM = 5.0
    CHLOROPHYLL_HIGH = 20.0
    
    KD490_LOW = 0.1
    KD490_HIGH = 0.5
    
    SECCHI_LOW = 5.0
    SECCHI_HIGH = 20.0

class ModelConfig:
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    RANDOM_STATE = 42
    N_ESTIMATORS = 100
    MAX_DEPTH = 10
    LEARNING_RATE = 0.01
    EARLY_STOPPING_ROUNDS = 10

class Paths:
    RAW_DATA = "data/raw/"
    PROCESSED_DATA = "data/processed/"
    MODELS = "models/"
    FORECASTS = "data/forecasts/"
    REPORTS = "reports/"
