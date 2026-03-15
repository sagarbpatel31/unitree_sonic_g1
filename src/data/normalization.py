"""
Motion data normalization module for training preparation.

This module provides utilities for normalizing and standardizing motion data
to improve training stability and performance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MotionStatistics:
    """Statistics for motion data normalization."""
    # Joint statistics
    joint_pos_mean: np.ndarray  # (n_joints,)
    joint_pos_std: np.ndarray   # (n_joints,)
    joint_vel_mean: np.ndarray  # (n_joints,)
    joint_vel_std: np.ndarray   # (n_joints,)
    joint_acc_mean: np.ndarray  # (n_joints,)
    joint_acc_std: np.ndarray   # (n_joints,)

    # Root motion statistics
    root_pos_mean: np.ndarray   # (3,)
    root_pos_std: np.ndarray    # (3,)
    root_orient_mean: np.ndarray  # (4,) - for quaternions
    root_orient_std: np.ndarray   # (4,)
    root_lin_vel_mean: np.ndarray  # (3,)
    root_lin_vel_std: np.ndarray   # (3,)
    root_ang_vel_mean: np.ndarray  # (3,)
    root_ang_vel_std: np.ndarray   # (3,)

    # Metadata
    n_sequences: int
    total_frames: int
    joint_names: List[str]
    data_type: str = "g1_trajectory"


@dataclass
class NormalizationConfig:
    """Configuration for motion normalization."""
    # What to normalize
    normalize_joint_pos: bool = True
    normalize_joint_vel: bool = True
    normalize_joint_acc: bool = False
    normalize_root_pos: bool = True
    normalize_root_orient: bool = False  # Quaternions are tricky
    normalize_root_lin_vel: bool = True
    normalize_root_ang_vel: bool = True

    # Normalization methods
    joint_pos_method: str = "standard"  # "standard", "minmax", "robust"
    joint_vel_method: str = "standard"
    joint_acc_method: str = "standard"
    root_pos_method: str = "standard"
    root_vel_method: str = "standard"

    # Outlier handling
    outlier_std_threshold: float = 3.0
    clip_outliers: bool = True

    # Minimum std for numerical stability
    min_std: float = 1e-6

    # Special handling
    preserve_root_height: bool = True  # Don't normalize Z-axis of root position
    preserve_root_yaw: bool = False    # Preserve yaw rotation


class MotionNormalizer:
    """Normalizer for motion trajectory data."""

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize motion normalizer.

        Args:
            config: Normalization configuration
        """
        self.config = config or NormalizationConfig()
        self.statistics: Optional[MotionStatistics] = None
        self.is_fitted = False

        logger.info("Initialized MotionNormalizer")

    def fit(self, trajectories: List[Dict[str, np.ndarray]]) -> None:
        """
        Compute normalization statistics from trajectory data.

        Args:
            trajectories: List of trajectory dicts with keys:
                - joint_positions: (T, n_joints)
                - joint_velocities: (T, n_joints)
                - joint_accelerations: (T, n_joints)
                - root_positions: (T, 3)
                - root_orientations: (T, 4)
                - root_linear_velocities: (T, 3)
                - root_angular_velocities: (T, 3)
        """
        if not trajectories:
            raise ValueError("No trajectories provided for fitting")

        logger.info(f"Computing normalization statistics from {len(trajectories)} trajectories")

        # Collect all data
        all_joint_pos = []
        all_joint_vel = []
        all_joint_acc = []
        all_root_pos = []
        all_root_orient = []
        all_root_lin_vel = []
        all_root_ang_vel = []

        total_frames = 0
        joint_names = None

        for i, traj in enumerate(trajectories):
            try:
                # Joint data
                if "joint_positions" in traj:
                    joint_pos = traj["joint_positions"]
                    all_joint_pos.append(joint_pos)
                    if joint_names is None and "joint_names" in traj:
                        joint_names = traj["joint_names"]

                if "joint_velocities" in traj:
                    all_joint_vel.append(traj["joint_velocities"])

                if "joint_accelerations" in traj:
                    all_joint_acc.append(traj["joint_accelerations"])

                # Root motion data
                if "root_positions" in traj:
                    all_root_pos.append(traj["root_positions"])

                if "root_orientations" in traj:
                    all_root_orient.append(traj["root_orientations"])

                if "root_linear_velocities" in traj:
                    all_root_lin_vel.append(traj["root_linear_velocities"])

                if "root_angular_velocities" in traj:
                    all_root_ang_vel.append(traj["root_angular_velocities"])

                total_frames += len(joint_pos) if len(all_joint_pos) > 0 else 0

            except Exception as e:
                logger.warning(f"Error processing trajectory {i}: {e}")
                continue

        if not all_joint_pos:
            raise ValueError("No valid joint position data found")

        # Concatenate all data
        all_joint_pos = np.concatenate(all_joint_pos, axis=0)
        n_joints = all_joint_pos.shape[1]

        if joint_names is None:
            joint_names = [f"joint_{i}" for i in range(n_joints)]

        logger.info(f"Collected data: {total_frames} frames, {n_joints} joints")

        # Remove outliers if requested
        if self.config.clip_outliers:
            all_joint_pos = self._remove_outliers(all_joint_pos)

        # Compute joint statistics
        joint_pos_mean, joint_pos_std = self._compute_statistics(
            all_joint_pos, self.config.joint_pos_method
        )

        joint_vel_mean = joint_vel_std = np.zeros(n_joints)
        if all_joint_vel:
            all_joint_vel = np.concatenate(all_joint_vel, axis=0)
            if self.config.clip_outliers:
                all_joint_vel = self._remove_outliers(all_joint_vel)
            joint_vel_mean, joint_vel_std = self._compute_statistics(
                all_joint_vel, self.config.joint_vel_method
            )

        joint_acc_mean = joint_acc_std = np.zeros(n_joints)
        if all_joint_acc:
            all_joint_acc = np.concatenate(all_joint_acc, axis=0)
            if self.config.clip_outliers:
                all_joint_acc = self._remove_outliers(all_joint_acc)
            joint_acc_mean, joint_acc_std = self._compute_statistics(
                all_joint_acc, self.config.joint_acc_method
            )

        # Compute root motion statistics
        root_pos_mean = root_pos_std = np.zeros(3)
        if all_root_pos:
            all_root_pos = np.concatenate(all_root_pos, axis=0)
            if self.config.clip_outliers:
                all_root_pos = self._remove_outliers(all_root_pos)
            root_pos_mean, root_pos_std = self._compute_statistics(
                all_root_pos, self.config.root_pos_method
            )

            # Special handling for root height
            if self.config.preserve_root_height:
                root_pos_mean[2] = 0.0
                root_pos_std[2] = 1.0

        root_orient_mean = root_orient_std = np.array([0, 0, 0, 1])
        if all_root_orient:
            all_root_orient = np.concatenate(all_root_orient, axis=0)
            # Quaternion statistics are complex - use identity for now
            root_orient_mean = np.array([0, 0, 0, 1])
            root_orient_std = np.array([1, 1, 1, 1])

        root_lin_vel_mean = root_lin_vel_std = np.zeros(3)
        if all_root_lin_vel:
            all_root_lin_vel = np.concatenate(all_root_lin_vel, axis=0)
            if self.config.clip_outliers:
                all_root_lin_vel = self._remove_outliers(all_root_lin_vel)
            root_lin_vel_mean, root_lin_vel_std = self._compute_statistics(
                all_root_lin_vel, self.config.root_vel_method
            )

        root_ang_vel_mean = root_ang_vel_std = np.zeros(3)
        if all_root_ang_vel:
            all_root_ang_vel = np.concatenate(all_root_ang_vel, axis=0)
            if self.config.clip_outliers:
                all_root_ang_vel = self._remove_outliers(all_root_ang_vel)
            root_ang_vel_mean, root_ang_vel_std = self._compute_statistics(
                all_root_ang_vel, self.config.root_vel_method
            )

        # Create statistics object
        self.statistics = MotionStatistics(
            joint_pos_mean=joint_pos_mean,
            joint_pos_std=joint_pos_std,
            joint_vel_mean=joint_vel_mean,
            joint_vel_std=joint_vel_std,
            joint_acc_mean=joint_acc_mean,
            joint_acc_std=joint_acc_std,
            root_pos_mean=root_pos_mean,
            root_pos_std=root_pos_std,
            root_orient_mean=root_orient_mean,
            root_orient_std=root_orient_std,
            root_lin_vel_mean=root_lin_vel_mean,
            root_lin_vel_std=root_lin_vel_std,
            root_ang_vel_mean=root_ang_vel_mean,
            root_ang_vel_std=root_ang_vel_std,
            n_sequences=len(trajectories),
            total_frames=total_frames,
            joint_names=joint_names
        )

        self.is_fitted = True
        logger.info("Normalization statistics computed successfully")

    def normalize(self, trajectory: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize a trajectory using computed statistics.

        Args:
            trajectory: Trajectory dict to normalize

        Returns:
            Normalized trajectory dict
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        normalized = trajectory.copy()

        # Normalize joint positions
        if self.config.normalize_joint_pos and "joint_positions" in trajectory:
            normalized["joint_positions"] = self._normalize_data(
                trajectory["joint_positions"],
                self.statistics.joint_pos_mean,
                self.statistics.joint_pos_std
            )

        # Normalize joint velocities
        if self.config.normalize_joint_vel and "joint_velocities" in trajectory:
            normalized["joint_velocities"] = self._normalize_data(
                trajectory["joint_velocities"],
                self.statistics.joint_vel_mean,
                self.statistics.joint_vel_std
            )

        # Normalize joint accelerations
        if self.config.normalize_joint_acc and "joint_accelerations" in trajectory:
            normalized["joint_accelerations"] = self._normalize_data(
                trajectory["joint_accelerations"],
                self.statistics.joint_acc_mean,
                self.statistics.joint_acc_std
            )

        # Normalize root positions
        if self.config.normalize_root_pos and "root_positions" in trajectory:
            normalized["root_positions"] = self._normalize_data(
                trajectory["root_positions"],
                self.statistics.root_pos_mean,
                self.statistics.root_pos_std
            )

        # Normalize root linear velocities
        if self.config.normalize_root_lin_vel and "root_linear_velocities" in trajectory:
            normalized["root_linear_velocities"] = self._normalize_data(
                trajectory["root_linear_velocities"],
                self.statistics.root_lin_vel_mean,
                self.statistics.root_lin_vel_std
            )

        # Normalize root angular velocities
        if self.config.normalize_root_ang_vel and "root_angular_velocities" in trajectory:
            normalized["root_angular_velocities"] = self._normalize_data(
                trajectory["root_angular_velocities"],
                self.statistics.root_ang_vel_mean,
                self.statistics.root_ang_vel_std
            )

        # Note: root_orientations are not normalized by default due to quaternion complexity

        return normalized

    def denormalize(self, trajectory: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Denormalize a trajectory back to original scale.

        Args:
            trajectory: Normalized trajectory dict

        Returns:
            Denormalized trajectory dict
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        denormalized = trajectory.copy()

        # Denormalize joint positions
        if self.config.normalize_joint_pos and "joint_positions" in trajectory:
            denormalized["joint_positions"] = self._denormalize_data(
                trajectory["joint_positions"],
                self.statistics.joint_pos_mean,
                self.statistics.joint_pos_std
            )

        # Denormalize joint velocities
        if self.config.normalize_joint_vel and "joint_velocities" in trajectory:
            denormalized["joint_velocities"] = self._denormalize_data(
                trajectory["joint_velocities"],
                self.statistics.joint_vel_mean,
                self.statistics.joint_vel_std
            )

        # Denormalize joint accelerations
        if self.config.normalize_joint_acc and "joint_accelerations" in trajectory:
            denormalized["joint_accelerations"] = self._denormalize_data(
                trajectory["joint_accelerations"],
                self.statistics.joint_acc_mean,
                self.statistics.joint_acc_std
            )

        # Denormalize root positions
        if self.config.normalize_root_pos and "root_positions" in trajectory:
            denormalized["root_positions"] = self._denormalize_data(
                trajectory["root_positions"],
                self.statistics.root_pos_mean,
                self.statistics.root_pos_std
            )

        # Denormalize root linear velocities
        if self.config.normalize_root_lin_vel and "root_linear_velocities" in trajectory:
            denormalized["root_linear_velocities"] = self._denormalize_data(
                trajectory["root_linear_velocities"],
                self.statistics.root_lin_vel_mean,
                self.statistics.root_lin_vel_std
            )

        # Denormalize root angular velocities
        if self.config.normalize_root_ang_vel and "root_angular_velocities" in trajectory:
            denormalized["root_angular_velocities"] = self._denormalize_data(
                trajectory["root_angular_velocities"],
                self.statistics.root_ang_vel_mean,
                self.statistics.root_ang_vel_std
            )

        return denormalized

    def _compute_statistics(self, data: np.ndarray, method: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std for data based on method."""
        if method == "standard":
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
        elif method == "robust":
            mean = np.median(data, axis=0)
            std = np.std(data, axis=0)  # Could use IQR instead
        elif method == "minmax":
            mean = np.min(data, axis=0)
            std = np.max(data, axis=0) - np.min(data, axis=0)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Ensure minimum std for numerical stability
        std = np.maximum(std, self.config.min_std)

        return mean, std

    def _normalize_data(self, data: np.ndarray,
                       mean: np.ndarray,
                       std: np.ndarray) -> np.ndarray:
        """Normalize data using mean and std."""
        return (data - mean) / std

    def _denormalize_data(self, data: np.ndarray,
                         mean: np.ndarray,
                         std: np.ndarray) -> np.ndarray:
        """Denormalize data using mean and std."""
        return data * std + mean

    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """Remove outliers beyond threshold standard deviations."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        # Identify outliers
        z_scores = np.abs((data - mean) / (std + 1e-8))
        outlier_mask = np.any(z_scores > self.config.outlier_std_threshold, axis=1)

        if np.sum(outlier_mask) > 0:
            logger.info(f"Removing {np.sum(outlier_mask)} outlier frames "
                       f"({100 * np.sum(outlier_mask) / len(data):.1f}%)")
            data = data[~outlier_mask]

        return data

    def save_statistics(self, filepath: Union[str, Path]) -> None:
        """Save normalization statistics to file."""
        if not self.is_fitted:
            raise ValueError("No statistics to save. Call fit() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'statistics': self.statistics,
                'config': self.config
            }, f)

        logger.info(f"Saved normalization statistics to {filepath}")

    def load_statistics(self, filepath: Union[str, Path]) -> None:
        """Load normalization statistics from file."""
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.statistics = data['statistics']
        self.config = data['config']
        self.is_fitted = True

        logger.info(f"Loaded normalization statistics from {filepath}")

    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get summary of normalization statistics."""
        if not self.is_fitted:
            return {}

        stats = self.statistics
        summary = {
            "n_sequences": stats.n_sequences,
            "total_frames": stats.total_frames,
            "n_joints": len(stats.joint_names),
            "joint_names": stats.joint_names,
            "joint_pos_range": {
                "mean_range": f"[{np.min(stats.joint_pos_mean):.3f}, {np.max(stats.joint_pos_mean):.3f}]",
                "std_range": f"[{np.min(stats.joint_pos_std):.3f}, {np.max(stats.joint_pos_std):.3f}]"
            },
            "joint_vel_range": {
                "mean_range": f"[{np.min(stats.joint_vel_mean):.3f}, {np.max(stats.joint_vel_mean):.3f}]",
                "std_range": f"[{np.min(stats.joint_vel_std):.3f}, {np.max(stats.joint_vel_std):.3f}]"
            },
            "root_pos_stats": {
                "mean": stats.root_pos_mean.tolist(),
                "std": stats.root_pos_std.tolist()
            },
            "root_vel_stats": {
                "lin_vel_mean": stats.root_lin_vel_mean.tolist(),
                "lin_vel_std": stats.root_lin_vel_std.tolist(),
                "ang_vel_mean": stats.root_ang_vel_mean.tolist(),
                "ang_vel_std": stats.root_ang_vel_std.tolist()
            }
        }

        return summary


def compute_motion_statistics(trajectories: List[Dict[str, np.ndarray]],
                            config: Optional[NormalizationConfig] = None) -> MotionStatistics:
    """
    Convenience function to compute motion statistics.

    Args:
        trajectories: List of trajectory dicts
        config: Optional normalization configuration

    Returns:
        MotionStatistics object
    """
    normalizer = MotionNormalizer(config)
    normalizer.fit(trajectories)
    return normalizer.statistics


def normalize_trajectory_batch(trajectories: List[Dict[str, np.ndarray]],
                              normalizer: Optional[MotionNormalizer] = None,
                              config: Optional[NormalizationConfig] = None) -> Tuple[List[Dict[str, np.ndarray]], MotionNormalizer]:
    """
    Normalize a batch of trajectories.

    Args:
        trajectories: List of trajectory dicts
        normalizer: Optional pre-fitted normalizer
        config: Optional normalization configuration

    Returns:
        Tuple of (normalized_trajectories, normalizer)
    """
    if normalizer is None:
        normalizer = MotionNormalizer(config)
        normalizer.fit(trajectories)

    normalized_trajectories = [
        normalizer.normalize(traj) for traj in trajectories
    ]

    return normalized_trajectories, normalizer