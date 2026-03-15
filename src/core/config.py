"""
Configuration management system using OmegaConf and Hydra.
Supports hierarchical configs, overrides, and validation.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_store
from hydra.core.config_store import ConfigStore


class Config:
    """Configuration manager with validation and type safety."""

    def __init__(self, config: Union[DictConfig, Dict[str, Any]]):
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self._config = config
        self._validate()

    def _validate(self):
        """Validate configuration for common issues."""
        # Check required top-level keys
        required_keys = ["experiment", "env", "model", "training"]
        for key in required_keys:
            if key not in self._config:
                raise ValueError(f"Required config key '{key}' missing")

        # Validate device setting
        if "device" in self._config:
            device = self._config.device
            if device not in ["auto", "cpu", "cuda"]:
                if not device.startswith("cuda:"):
                    raise ValueError(f"Invalid device: {device}")

        # Validate paths
        if "model_path" in self._config.get("env", {}).get("robot", {}):
            model_path = self._config.env.robot.model_path
            if not os.path.exists(model_path):
                print(f"Warning: Model path {model_path} does not exist")

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation."""
        try:
            return OmegaConf.select(self._config, key, default=default)
        except Exception:
            return default

    def set(self, key: str, value: Any) -> None:
        """Set config value using dot notation."""
        OmegaConf.update(self._config, key, value, merge=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return OmegaConf.to_container(self._config, resolve=True)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        OmegaConf.save(self._config, path)

    @property
    def raw(self) -> DictConfig:
        """Access raw OmegaConf object."""
        return self._config

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access."""
        if name.startswith("_"):
            return super().__getattribute__(name)
        return getattr(self._config, name)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._config[key]

    def __repr__(self) -> str:
        """String representation."""
        return f"Config({OmegaConf.to_yaml(self._config)})"


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[list] = None,
    validate: bool = True,
) -> Config:
    """
    Load configuration from file with optional overrides.

    Args:
        config_path: Path to config file
        overrides: List of override strings (e.g., ["training.batch_size=64"])
        validate: Whether to validate the config

    Returns:
        Config object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load base config
    base_config = OmegaConf.load(config_path)

    # Apply overrides
    if overrides:
        override_config = OmegaConf.from_cli(overrides)
        base_config = OmegaConf.merge(base_config, override_config)

    # Resolve interpolations
    OmegaConf.resolve(base_config)

    if validate:
        return Config(base_config)
    else:
        return base_config


def load_config_hydra(
    config_name: str,
    config_path: Optional[str] = None,
    overrides: Optional[list] = None,
) -> Config:
    """
    Load configuration using Hydra (for more complex scenarios).

    Args:
        config_name: Name of config (without .yaml)
        config_path: Directory containing configs
        overrides: List of override strings

    Returns:
        Config object
    """
    with initialize_config_store(config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides or [])
        return Config(cfg)


def merge_configs(*configs: Union[Config, DictConfig, dict]) -> Config:
    """
    Merge multiple configurations.

    Args:
        configs: Configuration objects to merge

    Returns:
        Merged configuration
    """
    merged = OmegaConf.create({})

    for config in configs:
        if isinstance(config, Config):
            config_dict = config.raw
        elif isinstance(config, dict):
            config_dict = OmegaConf.create(config)
        else:
            config_dict = config

        merged = OmegaConf.merge(merged, config_dict)

    return Config(merged)


def create_experiment_config(
    base_config_path: Union[str, Path],
    experiment_name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    Create experiment-specific configuration.

    Args:
        base_config_path: Path to base config
        experiment_name: Name for the experiment
        overrides: Dictionary of config overrides

    Returns:
        Experiment configuration
    """
    config = load_config(base_config_path, validate=False)

    # Set experiment name
    config.set("experiment.name", experiment_name)

    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            config.set(key, value)

    return Config(config.raw)


# Register common configurations
cs = ConfigStore.instance()

# TODO: Register structured configs for type safety
# cs.store(name="base_config", node=BaseConfig)