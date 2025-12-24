"""
Data preprocessing for marine data
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
import logging
from scipy import stats
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
# import bottleneck as bn  # Optional - remove or comment if not needed

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing for marine datasets"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.scalers = {}
    
    # بقیه کدها...
