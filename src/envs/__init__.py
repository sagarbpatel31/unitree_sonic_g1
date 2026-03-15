"""
Environment module for MuJoCo-based Unitree G1 simulation.

This module contains:
- MuJoCo environment wrappers
- Motion imitation tasks
- Domain randomization
- Observation/action space definitions
"""

from .g1_env import G1Environment
from .motion_imitation import MotionImitationEnv
from .robust_training import RobustTrainingEnv

__all__ = [
    "G1Environment",
    "MotionImitationEnv",
    "RobustTrainingEnv",
]