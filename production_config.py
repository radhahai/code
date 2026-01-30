"""
production_config.py - Production Configuration

Environment-based configuration management for deployment.
Supports development, staging, and production environments.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AppConfig:
    """Application configuration settings."""
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"
    
    # Performance
    max_workers: int = 4
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Analysis limits
    max_floors: int = 100
    max_duration: float = 600.0
    max_time_steps: int = 100000
    
    # UI
    max_plot_points: int = 2000
    animation_max_frames: int = 100
    
    # Feature flags
    enable_nonlinear: bool = True
    enable_tmd: bool = True
    enable_base_isolation: bool = True
    enable_fragility: bool = True
    enable_is1893: bool = True
    
    @classmethod
    def from_environment(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        env = os.environ.get("SEISMIC_ENV", "development")
        
        if env == "production":
            return cls.production()
        elif env == "staging":
            return cls.staging()
        else:
            return cls.development()
    
    @classmethod
    def development(cls) -> "AppConfig":
        """Development configuration."""
        return cls(
            environment="development",
            debug=True,
            log_level="DEBUG",
            log_to_file=True,
            cache_enabled=True
        )
    
    @classmethod
    def staging(cls) -> "AppConfig":
        """Staging configuration."""
        return cls(
            environment="staging",
            debug=False,
            log_level="INFO",
            log_to_file=True,
            cache_enabled=True
        )
    
    @classmethod
    def production(cls) -> "AppConfig":
        """Production configuration."""
        return cls(
            environment="production",
            debug=False,
            log_level="WARNING",
            log_to_file=True,
            log_dir=os.environ.get("LOG_DIR", "/var/log/seismic_app"),
            max_workers=int(os.environ.get("MAX_WORKERS", "4")),
            cache_enabled=True,
            cache_ttl=7200
        )


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the current application configuration."""
    global _config
    if _config is None:
        _config = AppConfig.from_environment()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the application configuration."""
    global _config
    _config = config


# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

# Supported environment variables:
# 
# SEISMIC_ENV          - Environment name (development, staging, production)
# SEISMIC_LOG_LEVEL    - Log level (DEBUG, INFO, WARNING, ERROR)
# SEISMIC_LOG_DIR      - Log directory path
# SEISMIC_LOG_TO_FILE  - Enable file logging (true/false)
# LOG_DIR              - Alternative log directory (for production)
# MAX_WORKERS          - Maximum worker threads


# =============================================================================
# STREAMLIT SECRETS INTEGRATION
# =============================================================================

def load_streamlit_secrets() -> dict:
    """
    Load secrets from Streamlit secrets management.
    
    Returns empty dict if not running in Streamlit or no secrets configured.
    """
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            return dict(st.secrets)
    except Exception:
        pass
    return {}
