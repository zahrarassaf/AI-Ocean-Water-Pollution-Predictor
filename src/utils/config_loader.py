"""
Configuration loader with singleton pattern for OceanPredict
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Singleton configuration loader with dot notation access"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.configs = {}
            self._load_all_configs()
            self._initialized = True
    
    def _load_all_configs(self):
        """Load all configuration files from config directory"""
        config_dir = Path(__file__).parent.parent.parent / "config"
        
        for config_file in config_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    config_name = config_file.stem
                    self.configs[config_name] = yaml.safe_load(f)
                logger.info(f"Loaded configuration: {config_name}")
            except Exception as e:
                logger.error(f"Error loading {config_file}: {e}")
    
    def get(self, key: str, default: Any = None, config: str = "settings") -> Any:
        """
        Get configuration value using dot notation
        
        Parameters
        ----------
        key : str
            Dot notation key (e.g., 'project.name')
        default : Any
            Default value if key not found
        config : str
            Configuration file name (without .yaml)
            
        Returns
        -------
        Any
            Configuration value
        """
        if config not in self.configs:
            return default
        
        keys = key.split('.')
        value = self.configs[config]
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any, config: str = "settings") -> None:
        """Update configuration value"""
        if config not in self.configs:
            self.configs[config] = {}
        
        keys = key.split('.')
        config_level = self.configs[config]
        
        for k in keys[:-1]:
            if k not in config_level:
                config_level[k] = {}
            config_level = config_level[k]
        
        config_level[keys[-1]] = value
        logger.info(f"Updated config {config}.{key} = {value}")
    
    def save(self, config: str = "settings") -> None:
        """Save configuration to file"""
        config_file = Path(__file__).parent.parent.parent / "config" / f"{config}.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(self.configs[config], f, default_flow_style=False)
        
        logger.info(f"Saved configuration to {config_file}")
