"""
Advanced logging configuration
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
from dataclasses import dataclass, asdict

@dataclass
class LogConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_file: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    json_format: bool = False

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""
    
    def format(self, record):
        log_record = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
        
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_record)

def setup_logger(
    name: str,
    config: Optional[LogConfig] = None,
    extra_handlers: Optional[list] = None
) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Parameters
    ----------
    name : str
        Logger name
    config : LogConfig, optional
        Logging configuration
    extra_handlers : list, optional
        Additional log handlers
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    config = config or LogConfig()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if config.json_format:
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            config.format,
            datefmt=config.date_format
        )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            filename=config.log_file,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        
        if config.json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                config.format,
                datefmt=config.date_format
            )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Add extra handlers
    if extra_handlers:
        for handler in extra_handlers:
            logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

# Optional: Create a default logger instance
# default_logger = setup_logger('marine_predictor')
