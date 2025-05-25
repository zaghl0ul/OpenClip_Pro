"""Centralized configuration management."""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ai.error_handling import handle_errors, validate_input


@dataclass
class AppConfig:
    """Application configuration."""
    app_name: str = "OpenClip Pro"
    app_version: str = "1.1"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Paths
    data_dir: str = field(default_factory=lambda: str(Path.home() / ".openclip"))
    temp_dir: str = field(default_factory=lambda: str(Path.home() / ".openclip_temp"))
    db_file: str = field(default_factory=lambda: str(Path.home() / ".openclip" / "openclip.db"))
    api_key_file: str = "api_keys.json"
    
    # Defaults
    default_project_name: str = "Project %Y-%m-%d"
    default_clip_length: int = 60
    default_frame_sample_rate: float = 2.5
    default_score_threshold: int = 75
    default_ai_provider: str = "openai"
    default_ai_model: str = "gpt-4o"
    
    # Export settings
    export_format: str = "web_optimized"
    compression_quality: int = 85
    max_resolution: int = 720
    
    # Performance settings
    max_workers_extraction: int = 4
    max_workers_encoding: int = 4
    max_workers_api: int = 8
    max_workers_clip_gen: int = 4
    
    # AI settings
    default_temperature: float = 0.5
    default_max_tokens: int = 1000
    default_api_timeout: int = 120
    
    # UI settings
    default_theme: str = "dark"
    
    def validate(self) -> None:
        """Validate configuration values."""
        validate_input(self.app_name, str, "app_name", min_length=1)
        validate_input(self.app_version, str, "app_version", min_length=1)
        validate_input(self.log_level, str, "log_level", 
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        
        validate_input(self.default_clip_length, int, "default_clip_length", 
                      min_value=1, max_value=600)
        validate_input(self.default_frame_sample_rate, float, "default_frame_sample_rate",
                      min_value=0.1, max_value=30.0)
        validate_input(self.default_score_threshold, int, "default_score_threshold",
                      min_value=0, max_value=100)
        
        validate_input(self.compression_quality, int, "compression_quality",
                      min_value=1, max_value=100)
        validate_input(self.max_resolution, int, "max_resolution",
                      min_value=144, max_value=4320)
        
        validate_input(self.default_temperature, float, "default_temperature",
                      min_value=0.0, max_value=2.0)
        validate_input(self.default_max_tokens, int, "default_max_tokens",
                      min_value=1, max_value=100000)
        validate_input(self.default_api_timeout, int, "default_api_timeout",
                      min_value=1, max_value=600)
        
        # Validate worker counts
        for attr in ["max_workers_extraction", "max_workers_encoding", 
                    "max_workers_api", "max_workers_clip_gen"]:
            validate_input(getattr(self, attr), int, attr, min_value=1, max_value=32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Create from dictionary."""
        # Filter out unknown keys
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file) if config_file else Path.home() / ".openclip" / "config.json"
        self._config = AppConfig()
        self._load_config()
    
    @handle_errors(reraise=False, error_prefix="Config loading")
    def _load_config(self) -> None:
        """Load configuration from file and environment."""
        # Load from file if exists
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self._config = AppConfig.from_dict(data)
            except (json.JSONDecodeError, ValueError) as e:
                # Log error but continue with defaults
                pass
        
        # Override with environment variables
        self._load_env_overrides()
        
        # Validate configuration
        try:
            self._config.validate()
        except ValueError as e:
            # Reset to defaults if validation fails
            self._config = AppConfig()
    
    def _load_env_overrides(self) -> None:
        """Load configuration overrides from environment variables."""
        env_mapping = {
            "OPENCLIP_DEBUG": ("debug_mode", lambda x: x.lower() in ("true", "1", "yes")),
            "OPENCLIP_LOG_LEVEL": ("log_level", str.upper),
            "OPENCLIP_DATA_DIR": ("data_dir", str),
            "OPENCLIP_TEMP_DIR": ("temp_dir", str),
            "OPENCLIP_DB_FILE": ("db_file", str),
            "OPENCLIP_API_KEY_FILE": ("api_key_file", str),
            "OPENCLIP_DEFAULT_PROVIDER": ("default_ai_provider", str.lower),
            "OPENCLIP_DEFAULT_MODEL": ("default_ai_model", str),
            "OPENCLIP_MAX_WORKERS": ("max_workers_api", int),
            "OPENCLIP_API_TIMEOUT": ("default_api_timeout", int),
        }
        
        for env_var, (attr, converter) in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    setattr(self._config, attr, converter(value))
                except (ValueError, AttributeError):
                    pass
    
    @handle_errors(default_return=False, error_prefix="Config saving")
    def save_config(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            True if successful
        """
        # Ensure directory exists
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(self.config_file, 'w') as f:
            json.dump(self._config.to_dict(), f, indent=2)
        
        return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return getattr(self._config, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        if hasattr(self._config, key):
            setattr(self._config, key, value)
            # Validate after setting
            try:
                self._config.validate()
            except ValueError:
                # Revert if validation fails
                self._load_config()
                raise
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of updates
        """
        for key, value in updates.items():
            self.set(key, value)
    
    @property
    def config(self) -> AppConfig:
        """Get the configuration object."""
        return self._config
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self._config = AppConfig()
        self.save_config()


# Singleton instance
_config_manager = None


def get_config_manager(config_file: Optional[Union[str, Path]] = None) -> ConfigManager:
    """
    Get or create the singleton ConfigManager instance.
    
    Args:
        config_file: Configuration file path (only used on first call)
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    
    return _config_manager


def get_config() -> AppConfig:
    """
    Get the current configuration.
    
    Returns:
        AppConfig instance
    """
    return get_config_manager().config 