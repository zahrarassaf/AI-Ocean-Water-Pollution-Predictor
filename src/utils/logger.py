"""
Advanced logging configuration for OceanPredict
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

class OceanPredictLogger:
    """Custom logger with JSON formatting and structured logging"""
    
    def __init__(self, name: str = "OceanPredict"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handler(log_dir)
        self._setup_json_handler(log_dir)
        
        # Store execution context
        self.context = {
            "project": "OceanPredict",
            "start_time": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    
    def _setup_console_handler(self):
        """Setup console handler with colored output"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Custom formatter for console
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, log_dir: Path):
        """Setup file handler for detailed logs"""
        log_file = log_dir / f"oceanpredict_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(file_handler)
    
    def _setup_json_handler(self, log_dir: Path):
        """Setup JSON handler for structured logging"""
        json_file = log_dir / f"oceanpredict_structured_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        json_handler = logging.FileHandler(json_file)
        json_handler.setLevel(logging.INFO)
        
        json_handler.setFormatter(JsonFormatter())
        
        self.logger.addHandler(json_handler)
    
    def update_context(self, **kwargs):
        """Update logging context"""
        self.context.update(kwargs)
    
    def info(self, msg: str, **extra):
        """Log info message with extra context"""
        self.logger.info(msg, extra={**self.context, **extra})
    
    def warning(self, msg: str, **extra):
        """Log warning message with extra context"""
        self.logger.warning(msg, extra={**self.context, **extra})
    
    def error(self, msg: str, **extra):
        """Log error message with extra context"""
        self.logger.error(msg, extra={**self.context, **extra})
    
    def debug(self, msg: str, **extra):
        """Log debug message with extra context"""
        self.logger.debug(msg, extra={**self.context, **extra})
    
    def exception(self, msg: str, **extra):
        """Log exception with traceback"""
        self.logger.exception(msg, extra={**self.context, **extra})

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_record = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key.startswith('_') or key in log_record:
                continue
            log_record[key] = value
        
        return json.dumps(log_record)

# Global logger instance
logger = OceanPredictLogger()
