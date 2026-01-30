"""
logging_config.py - Logging Configuration

Production-ready logging setup with structured logging,
file rotation, and configurable log levels.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default log directory (can be overridden via environment variable)
LOG_DIR = os.environ.get("SEISMIC_LOG_DIR", "logs")
LOG_LEVEL = os.environ.get("SEISMIC_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.environ.get(
    "SEISMIC_LOG_FORMAT",
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Max log file size (10 MB)
MAX_LOG_SIZE = 10 * 1024 * 1024
# Number of backup files to keep
BACKUP_COUNT = 5


# =============================================================================
# CUSTOM FORMATTER FOR JSON LOGGING
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# =============================================================================
# LOGGER SETUP
# =============================================================================

def setup_logger(
    name: str = "seismic_app",
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    json_format: bool = False
) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Parameters
    ----------
    name : str
        Logger name
    level : str, optional
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_to_file : bool
        Whether to write logs to file
    log_to_console : bool
        Whether to write logs to console
    json_format : bool
        Use JSON format for file logging
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if logger.handlers:
        return logger
    
    log_level = getattr(logging, level or LOG_LEVEL, logging.INFO)
    logger.setLevel(log_level)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Use colored formatter for console in development
        if os.environ.get("SEISMIC_ENV", "development") == "development":
            console_formatter = ColoredFormatter(LOG_FORMAT, LOG_DATE_FORMAT)
        else:
            console_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        # Ensure log directory exists
        log_path = Path(LOG_DIR)
        log_path.mkdir(parents=True, exist_ok=True)
        
        log_file = log_path / f"{name}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT
        )
        file_handler.setLevel(log_level)
        
        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "seismic_app") -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Parameters
    ----------
    name : str
        Logger name (will be prefixed with 'seismic_app.')
    
    Returns
    -------
    logging.Logger
        Logger instance
    """
    if not name.startswith("seismic_app"):
        name = f"seismic_app.{name}"
    return logging.getLogger(name)


# =============================================================================
# PERFORMANCE LOGGING
# =============================================================================

class PerformanceLogger:
    """Context manager for logging performance metrics."""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        self.operation = operation
        self.logger = logger or get_logger("performance")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type:
            self.logger.error(
                f"Failed: {self.operation} after {elapsed:.3f}s - {exc_type.__name__}: {exc_val}"
            )
        else:
            self.logger.info(f"Completed: {self.operation} in {elapsed:.3f}s")
        
        return False  # Don't suppress exceptions


def log_performance(operation: str):
    """Decorator for logging function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger("performance")
            start = datetime.now()
            try:
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start).total_seconds()
                logger.info(f"{func.__name__}: {operation} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = (datetime.now() - start).total_seconds()
                logger.error(f"{func.__name__}: {operation} failed after {elapsed:.3f}s - {e}")
                raise
        return wrapper
    return decorator


# =============================================================================
# ANALYSIS LOGGING
# =============================================================================

def log_analysis_start(logger: logging.Logger, params: dict) -> None:
    """Log the start of an analysis with key parameters."""
    logger.info(
        f"Analysis started: {params.get('analysis_type', 'Unknown')} | "
        f"Floors: {params.get('n_floors', '?')} | "
        f"Event: {params.get('event_type', '?')} | "
        f"PGA: {params.get('pga', '?')}g"
    )


def log_analysis_complete(logger: logging.Logger, results: dict, elapsed: float) -> None:
    """Log analysis completion with key results."""
    logger.info(
        f"Analysis complete in {elapsed:.2f}s | "
        f"Max Drift: {results.get('max_drift_ratio', 0):.2f}% | "
        f"Max Base Shear: {results.get('max_base_shear', 0):.0f} kN | "
        f"Performance: {results.get('perf_level', 'N/A')}"
    )


def log_validation_error(logger: logging.Logger, errors: list) -> None:
    """Log validation errors."""
    for error in errors:
        logger.warning(f"Validation: {error}")


# =============================================================================
# INITIALIZE DEFAULT LOGGER
# =============================================================================

# Create default application logger on module import
_default_logger = setup_logger(
    "seismic_app",
    log_to_file=os.environ.get("SEISMIC_LOG_TO_FILE", "true").lower() == "true",
    log_to_console=True
)
