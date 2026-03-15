"""
Data processing module for motion retargeting and preprocessing.

This module contains tools for converting human motion capture data
into robot-compatible trajectories for training.
"""

from .retarget_to_g1 import MotionRetargeter, retarget_motion_clip
from .skeleton_map import SkeletonMapper, G1_JOINT_MAP, HUMAN_SKELETON_MAP
from .contact_estimation import ContactEstimator, estimate_foot_contacts
from .normalization import MotionNormalizer, compute_motion_statistics
from .validate_retargeting import validate_retargeted_motion

__all__ = [
    "MotionRetargeter",
    "retarget_motion_clip",
    "SkeletonMapper",
    "G1_JOINT_MAP",
    "HUMAN_SKELETON_MAP",
    "ContactEstimator",
    "estimate_foot_contacts",
    "MotionNormalizer",
    "compute_motion_statistics",
    "validate_retargeted_motion",
]