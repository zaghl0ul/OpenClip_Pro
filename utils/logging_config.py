"""Centralized logging configuration."""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Union

from utils.config_manager import get_config


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, fmt: Optional[str] = None, use_colors: bool = True):
        super().__init__(fmt)
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors if enabled."""
        if self.use_colors and record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        return super().format(record)


class LoggingConfig:
    """Centralized logging configuration manager."""
    
    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    
    def __init__(self):
        self.config = get_config()
        self._configured = False
        self._log_file: Optional[Path] = None
    
    def setup_logging(
        self,
        log_file: Optional[Union[str, Path]] = None,
        console: bool = True,
        use_colors: bool = True,
        detailed: bool = False
    ) -> None:
        """
        Configure logging for the application.
        
        Args:
            log_file: Optional log file path
            console: Whether to log to console
            use_colors: Whether to use colors in console output
            detailed: Whether to use detailed format
        """
        if self._configured:
            return
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Choose format
        log_format = self.DETAILED_FORMAT if detailed else self.DEFAULT_FORMAT
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.log_level))
            
            if use_colors:
                console_formatter = ColoredFormatter(log_format, use_colors=True)
            else:
                console_formatter = logging.Formatter(log_format)
            
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            self._setup_file_handler(root_logger, log_file, log_format)
        
        # Configure specific loggers
        self._configure_specific_loggers()
        
        self._configured = True
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured with level: {self.config.log_level}")
    
    def _setup_file_handler(
        self,
        logger: logging.Logger,
        log_file: Union[str, Path],
        log_format: str
    ) -> None:
        """Set up file handler with rotation."""
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        
        file_handler.setLevel(getattr(logging, self.config.log_level))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
        self._log_file = log_file
    
    def _configure_specific_loggers(self) -> None:
        """Configure specific third-party loggers."""
        # Reduce noise from third-party libraries
        noisy_loggers = [
            "urllib3",
            "requests",
            "httpx",
            "httpcore",
            "openai",
            "anthropic",
            "google",
            "PIL",
            "matplotlib",
        ]
        
        for logger_name in noisy_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
        
        # Set specific levels for our modules if in debug mode
        if self.config.debug_mode:
            our_modules = [
                "ai",
                "ui",
                "utils",
                "database",
                "media_utils",
            ]
            
            for module in our_modules:
                logger = logging.getLogger(module)
                logger.setLevel(logging.DEBUG)
    
    def get_log_file(self) -> Optional[Path]:
        """Get the current log file path."""
        return self._log_file
    
    def set_level(self, level: Union[str, int]) -> None:
        """
        Change the logging level.
        
        Args:
            level: Logging level (name or value)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Update all handlers
        for handler in root_logger.handlers:
            handler.setLevel(level)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)


# Singleton instance
_logging_config = None


def setup_logging(**kwargs) -> None:
    """
    Set up logging for the application.
    
    Args:
        **kwargs: Arguments for LoggingConfig.setup_logging()
    """
    global _logging_config
    
    if _logging_config is None:
        _logging_config = LoggingConfig()
    
    _logging_config.setup_logging(**kwargs)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Ensures logging is configured before returning logger.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    # Ensure logging is configured
    setup_logging()
    
    return logging.getLogger(name) 