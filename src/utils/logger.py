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

class ModelLogger:
    """Specialized logger for model training"""
    
    def __init__(self, model_name: str, log_dir: str = "logs"):
        self.model_name = model_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training log file
        train_log_path = self.log_dir / f"{model_name}_training.log"
        self.train_logger = setup_logger(
            f"{model_name}.training",
            LogConfig(
                level="INFO",
                log_file=str(train_log_path),
                json_format=True
            )
        )
        
        # Metrics log file
        metrics_log_path = self.log_dir / f"{model_name}_metrics.log"
        self.metrics_logger = setup_logger(
            f"{model_name}.metrics",
            LogConfig(
                level="INFO",
                log_file=str(metrics_log_path),
                json_format=True
            )
        )
    
    def log_training_start(self, n_samples: int, n_features: int):
        """Log training start"""
        self.train_logger.info(
            "Training started",
            extra={
                'model': self.model_name,
                'n_samples': n_samples,
                'n_features': n_features,
                'event': 'training_start'
            }
        )
    
    def log_epoch(self, 
                  epoch: int, 
                  total_epochs: int,
                  train_loss: float,
                  val_loss: Optional[float] = None,
                  learning_rate: Optional[float] = None):
        """Log epoch information"""
        extra_data = {
            'model': self.model_name,
            'epoch': epoch,
            'total_epochs': total_epochs,
            'train_loss': train_loss,
            'event': 'epoch_complete'
        }
        
        if val_loss is not None:
            extra_data['val_loss'] = val_loss
        
        if learning_rate is not None:
            extra_data['learning_rate'] = learning_rate
        
        self.train_logger.info(f"Epoch {epoch}/{total_epochs}", extra=extra_data)
    
    def log_metrics(self, 
                    metrics: dict,
                    dataset: str = 'validation'):
        """Log model metrics"""
        self.metrics_logger.info(
            f"Metrics on {dataset} set",
            extra={
                'model': self.model_name,
                'dataset': dataset,
                'metrics': metrics,
                'event': 'metrics_evaluation'
            }
        )
    
    def log_training_complete(self, 
                             training_time: float,
                             best_metric: Optional[float] = None):
        """Log training completion"""
        self.train_logger.info(
            "Training completed",
            extra={
                'model': self.model_name,
                'training_time_seconds': training_time,
                'best_metric': best_metric,
                'event': 'training_complete'
            }
        )
    
    def log_prediction(self, 
                       n_predictions: int,
                       inference_time: float):
        """Log prediction information"""
        self.train_logger.info(
            "Prediction made",
            extra={
                'model': self.model_name,
                'n_predictions': n_predictions,
                'inference_time_seconds': inference_time,
                'event': 'prediction'
            }
        )
