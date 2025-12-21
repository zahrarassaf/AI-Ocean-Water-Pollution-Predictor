"""
Configuration loader utility for the project
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Load and manage configuration settings"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        for key, value in updates.items():
            keys = key.split('.')
            config_level = self.config
            
            for k in keys[:-1]:
                if k not in config_level:
                    config_level[k] = {}
                config_level = config_level[k]
            
            config_level[keys[-1]] = value
        
        logger.info("Configuration updated")
    
    def save(self, output_path: Optional[str] = None) -> None:
        """Save configuration to file"""
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {output_path}")
