"""
Professional logging system with file rotation, structured logging, and context management.
"""

import logging
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import colorlog


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_object = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.threadName,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_object['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_object.update(record.extra)
        
        return json.dumps(log_object, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """Add contextual information to log records."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add context to record
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    max_file_size: int = 100 * 1024 * 1024,  # 100 MB
    backup_count: int = 10,
    json_format: bool = False,
    console: bool = True,
    context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Set up a professional logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        json_format: Whether to use JSON formatting
        console: Whether to log to console
        context: Additional context for logs
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        # Colorful console formatter
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        # Simple file formatter
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(console_formatter if not json_format else formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(file_formatter if not json_format else formatter)
        logger.addHandler(file_handler)
    
    # Add context filter
    if context:
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)
    
    # Don't propagate to root logger
    logger.propagate = False
    
    return logger


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed {self.name} in {elapsed_time:.2f} seconds")
        else:
            self.logger.error(
                f"Failed {self.name} after {elapsed_time:.2f} seconds",
                exc_info=(exc_type, exc_val, exc_tb)
            )
    
    def get_elapsed(self) -> float:
        """Get elapsed time without exiting context."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        return time.time() - self.start_time


class ExperimentLogger:
    """Logger for machine learning experiments."""
    
    def __init__(
        self,
        experiment_id: str,
        log_dir: Path = Path("logs"),
        level: str = "INFO"
    ):
        self.experiment_id = experiment_id
        self.log_dir = log_dir / experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup different loggers for different purposes
        self.main_logger = setup_logger(
            name=f"{experiment_id}.main",
            level=level,
            log_file=self.log_dir / "main.log",
            json_format=True
        )
        
        self.metrics_logger = setup_logger(
            name=f"{experiment_id}.metrics",
            level="INFO",
            log_file=self.log_dir / "metrics.log",
            json_format=True
        )
        
        self.data_logger = setup_logger(
            name=f"{experiment_id}.data",
            level="INFO",
            log_file=self.log_dir / "data.log",
            json_format=True
        )
        
        # Store experiment start time
        self.start_time = datetime.now()
        self.main_logger.info(
            f"Experiment {experiment_id} started at {self.start_time.isoformat()}",
            extra={'experiment_id': experiment_id, 'event': 'experiment_start'}
        )
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.main_logger.info(
            "Experiment configuration",
            extra={'config': config, 'event': 'config_log'}
        )
        
        # Save config to file
        config_file = self.log_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def log_metrics(self, metrics: Dict[str, Any], stage: str = "train"):
        """Log training metrics."""
        self.metrics_logger.info(
            f"{stage.capitalize()} metrics",
            extra={'stage': stage, 'metrics': metrics, 'event': 'metrics_log'}
        )
    
    def log_data_info(self, info: Dict[str, Any], data_type: str = "raw"):
        """Log data information."""
        self.data_logger.info(
            f"{data_type.capitalize()} data info",
            extra={'data_type': data_type, 'info': info, 'event': 'data_info'}
        )
    
    def log_experiment_end(self, status: str = "completed"):
        """Log experiment completion."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.main_logger.info(
            f"Experiment {status}",
            extra={
                'experiment_id': self.experiment_id,
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'event': 'experiment_end',
                'status': status
            }
        )
    
    def get_log_paths(self) -> Dict[str, Path]:
        """Get paths to all log files."""
        return {
            'main': self.log_dir / "main.log",
            'metrics': self.log_dir / "metrics.log",
            'data': self.log_dir / "data.log",
            'config': self.log_dir / "config.json",
            'directory': self.log_dir
        }


# Global experiment logger instance
_experiment_logger: Optional[ExperimentLogger] = None


def init_experiment_logger(experiment_id: str, log_dir: Path = Path("logs")) -> ExperimentLogger:
    """Initialize global experiment logger."""
    global _experiment_logger
    _experiment_logger = ExperimentLogger(experiment_id, log_dir)
    return _experiment_logger


def get_experiment_logger() -> ExperimentLogger:
    """Get global experiment logger."""
    if _experiment_logger is None:
        raise RuntimeError("Experiment logger not initialized. Call init_experiment_logger first.")
    return _experiment_logger


# Convenience decorator for logging function execution
def logged_function(logger_name: str = __name__):
    """Decorator to log function execution."""
    logger = logging.getLogger(logger_name)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Starting {func.__name__}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.info(f"Completed {func.__name__} in {elapsed_time:.2f}s")
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"Failed {func.__name__} after {elapsed_time:.2f}s: {e}")
                raise
        
        return wrapper
    
    return decorator


if __name__ == "__main__":
    # Example usage
    logger = setup_logger("test_logger", level="DEBUG", log_file=Path("test.log"))
    
    with Timer("test_operation", logger):
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Example with extra context
        logger.info("Operation completed", extra={'operation': 'test', 'status': 'success'})
