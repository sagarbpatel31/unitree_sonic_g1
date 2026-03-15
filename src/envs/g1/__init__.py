"""
Unitree G1 MuJoCo environment package.

This package provides a comprehensive MuJoCo-based environment for the
Unitree G1 humanoid robot with support for whole-body motion imitation,
domain randomization, and command-conditioned behaviors.

Main Components:
- G1Environment: Main environment class
- ObservationManager: Handles observation construction
- RewardManager: Implements reward functions
- ResetManager: Manages environment resets
- DomainRandomizer: Implements physics randomization
- CommandManager: Handles command-conditioned behaviors

Example usage:
    ```python
    from src.envs.g1 import G1Environment, create_g1_env

    # Create environment
    env = create_g1_env(
        model_path="path/to/unitree_g1.xml",
        config={
            "frame_skip": 10,
            "action_type": "position_delta",
            "observations": {"include_reference_motion": True},
            "rewards": {"joint_pos_weight": 1.0},
            "randomization": {"enabled": True},
            "commands": {"enabled": True}
        }
    )

    # Use environment
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    ```
"""

from .g1_env import G1Environment, create_g1_env
from .observations import ObservationManager, ObservationConfig
from .rewards import RewardManager, RewardConfig
from .resets import ResetManager, ResetConfig
from .randomization import DomainRandomizer, RandomizationConfig
from .commands import CommandManager, CommandConfig, CommandType

__all__ = [
    # Main environment
    "G1Environment",
    "create_g1_env",

    # Component managers
    "ObservationManager",
    "RewardManager",
    "ResetManager",
    "DomainRandomizer",
    "CommandManager",

    # Configuration classes
    "ObservationConfig",
    "RewardConfig",
    "ResetConfig",
    "RandomizationConfig",
    "CommandConfig",

    # Enums
    "CommandType",
]

# Version information
__version__ = "0.1.0"
__author__ = "Unitree G1 SONIC Team"
__email__ = "contact@example.com"