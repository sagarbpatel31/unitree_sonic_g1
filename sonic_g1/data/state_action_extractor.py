"""
State-action extraction utilities for behavior cloning.

This module provides configurable extraction of states and actions from
G1 motion trajectories for supervised learning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from omegaconf import DictConfig
import logging
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class StateActionExtractor:
    """
    Configurable extractor for state-action pairs from G1 trajectories.

    Supports various state representations and action types for
    different training objectives.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize state-action extractor.

        Args:
            config: Extraction configuration
        """
        self.config = config

        # State configuration
        self.include_joint_pos = config.state.get('include_joint_pos', True)
        self.include_joint_vel = config.state.get('include_joint_vel', True)
        self.include_root_pos = config.state.get('include_root_pos', True)
        self.include_root_orient = config.state.get('include_root_orient', True)
        self.include_root_lin_vel = config.state.get('include_root_lin_vel', True)
        self.include_root_ang_vel = config.state.get('include_root_ang_vel', True)
        self.include_previous_action = config.state.get('include_previous_action', False)
        self.include_reference_features = config.state.get('include_reference_features', True)

        # Reference motion features
        self.reference_horizon = config.state.get('reference_horizon', 10)
        self.reference_features = config.state.get('reference_features', ['joint_pos', 'root_pos'])

        # State processing
        self.root_orient_repr = config.state.get('root_orient_repr', 'quat')  # 'quat', 'euler', 'rotation_matrix'
        self.normalize_root_pos = config.state.get('normalize_root_pos', False)
        self.relative_root_pos = config.state.get('relative_root_pos', True)

        # Action configuration
        self.action_type = config.action.get('type', 'joint_positions')  # 'joint_positions', 'joint_deltas', 'joint_velocities'
        self.action_delta_scale = config.action.get('delta_scale', 1.0)
        self.clip_actions = config.action.get('clip_actions', False)
        self.action_clip_range = config.action.get('action_clip_range', [-1.0, 1.0])

        # Frame processing
        self.action_lookahead = config.get('action_lookahead', 1)  # Frames ahead for action
        self.state_history = config.get('state_history', 1)  # Frames of history to include

        logger.info(f"Initialized StateActionExtractor: action_type={self.action_type}, "
                   f"state_features={self._get_state_feature_summary()}")

    def _get_state_feature_summary(self) -> str:
        """Get summary of included state features."""
        features = []
        if self.include_joint_pos: features.append('joint_pos')
        if self.include_joint_vel: features.append('joint_vel')
        if self.include_root_pos: features.append('root_pos')
        if self.include_root_orient: features.append('root_orient')
        if self.include_root_lin_vel: features.append('root_lin_vel')
        if self.include_root_ang_vel: features.append('root_ang_vel')
        if self.include_previous_action: features.append('prev_action')
        if self.include_reference_features: features.append('reference')
        return ','.join(features)

    def extract_from_trajectory(self, trajectory) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Extract state-action pairs from a G1 trajectory.

        Args:
            trajectory: G1TrajectoryData object

        Returns:
            Tuple of (states, actions, metadata)
        """
        trajectory_length = len(trajectory.timestamps)

        # Determine valid indices (accounting for lookahead and history)
        start_idx = max(self.state_history - 1, 0)
        end_idx = trajectory_length - self.action_lookahead

        if end_idx <= start_idx:
            raise ValueError(f"Trajectory too short: {trajectory_length} frames, "
                           f"need at least {start_idx + self.action_lookahead + 1}")

        valid_indices = range(start_idx, end_idx)
        num_samples = len(valid_indices)

        # Extract states
        states = self._extract_states(trajectory, valid_indices)

        # Extract actions
        actions = self._extract_actions(trajectory, valid_indices)

        # Create metadata
        metadata = {
            'trajectory_length': trajectory_length,
            'num_samples': num_samples,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'state_dim': states.shape[1],
            'action_dim': actions.shape[1],
            'joint_names': getattr(trajectory, 'joint_names', None),
            'source_metadata': trajectory.metadata
        }

        logger.debug(f"Extracted {num_samples} state-action pairs, "
                    f"state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

        return states, actions, metadata

    def _extract_states(self, trajectory, indices: range) -> np.ndarray:
        """Extract state features for given indices."""
        num_samples = len(indices)
        state_components = []

        for i, idx in enumerate(indices):
            state_features = []

            # Joint positions
            if self.include_joint_pos:
                joint_pos = trajectory.joint_positions[idx]
                state_features.append(joint_pos)

            # Joint velocities
            if self.include_joint_vel:
                joint_vel = trajectory.joint_velocities[idx]
                state_features.append(joint_vel)

            # Root position
            if self.include_root_pos:
                root_pos = trajectory.root_positions[idx]

                if self.relative_root_pos and idx > 0:
                    # Relative to previous position
                    prev_root_pos = trajectory.root_positions[idx - 1]
                    root_pos = root_pos - prev_root_pos

                if self.normalize_root_pos:
                    # Simple normalization (could be improved with proper statistics)
                    root_pos = root_pos / 10.0  # Assume ~10m is reasonable scale

                state_features.append(root_pos)

            # Root orientation
            if self.include_root_orient:
                root_quat = trajectory.root_orientations[idx]
                root_orient = self._process_root_orientation(root_quat)
                state_features.append(root_orient)

            # Root linear velocity
            if self.include_root_lin_vel:
                root_lin_vel = trajectory.root_linear_velocities[idx]
                state_features.append(root_lin_vel)

            # Root angular velocity
            if self.include_root_ang_vel:
                root_ang_vel = trajectory.root_angular_velocities[idx]
                state_features.append(root_ang_vel)

            # Previous action
            if self.include_previous_action:
                if i > 0:
                    # Use previous action from trajectory
                    prev_action = self._compute_action_at_index(trajectory, indices[i-1])
                else:
                    # Use zero action for first frame
                    action_dim = self._get_action_dim(trajectory)
                    prev_action = np.zeros(action_dim)
                state_features.append(prev_action)

            # Reference motion features
            if self.include_reference_features:
                ref_features = self._extract_reference_features(trajectory, idx)
                state_features.append(ref_features)

            # Concatenate all features
            state_vector = np.concatenate(state_features)
            state_components.append(state_vector)

        states = np.array(state_components)
        return states

    def _extract_actions(self, trajectory, indices: range) -> np.ndarray:
        """Extract actions for given indices."""
        actions = []

        for idx in indices:
            action = self._compute_action_at_index(trajectory, idx)
            actions.append(action)

        actions = np.array(actions)

        # Apply action clipping if enabled
        if self.clip_actions:
            actions = np.clip(actions, self.action_clip_range[0], self.action_clip_range[1])

        return actions

    def _compute_action_at_index(self, trajectory, idx: int) -> np.ndarray:
        """Compute action for a specific index based on action type."""
        lookahead_idx = min(idx + self.action_lookahead, len(trajectory.timestamps) - 1)

        if self.action_type == 'joint_positions':
            # Direct joint positions as actions
            action = trajectory.joint_positions[lookahead_idx]

        elif self.action_type == 'joint_deltas':
            # Joint position deltas
            current_pos = trajectory.joint_positions[idx]
            target_pos = trajectory.joint_positions[lookahead_idx]
            action = (target_pos - current_pos) * self.action_delta_scale

        elif self.action_type == 'joint_velocities':
            # Joint velocities
            action = trajectory.joint_velocities[lookahead_idx]

        else:
            raise ValueError(f"Unknown action type: {self.action_type}")

        return action

    def _get_action_dim(self, trajectory) -> int:
        """Get action dimensionality."""
        return trajectory.joint_positions.shape[1]

    def _process_root_orientation(self, quaternion: np.ndarray) -> np.ndarray:
        """Process root orientation based on representation type."""
        if self.root_orient_repr == 'quat':
            return quaternion

        elif self.root_orient_repr == 'euler':
            # Convert to Euler angles (XYZ convention)
            rotation = R.from_quat(quaternion)
            euler_angles = rotation.as_euler('xyz')
            return euler_angles

        elif self.root_orient_repr == 'rotation_matrix':
            # Convert to rotation matrix (flattened)
            rotation = R.from_quat(quaternion)
            rot_matrix = rotation.as_matrix().flatten()
            return rot_matrix

        else:
            raise ValueError(f"Unknown root orientation representation: {self.root_orient_repr}")

    def _extract_reference_features(self, trajectory, idx: int) -> np.ndarray:
        """Extract reference motion features for future frames."""
        ref_features = []

        for lookahead in range(1, self.reference_horizon + 1):
            ref_idx = min(idx + lookahead, len(trajectory.timestamps) - 1)

            # Extract specified reference features
            if 'joint_pos' in self.reference_features:
                ref_features.append(trajectory.joint_positions[ref_idx])

            if 'joint_vel' in self.reference_features:
                ref_features.append(trajectory.joint_velocities[ref_idx])

            if 'root_pos' in self.reference_features:
                ref_pos = trajectory.root_positions[ref_idx]
                if self.relative_root_pos:
                    current_pos = trajectory.root_positions[idx]
                    ref_pos = ref_pos - current_pos
                ref_features.append(ref_pos)

            if 'root_orient' in self.reference_features:
                ref_quat = trajectory.root_orientations[ref_idx]
                ref_orient = self._process_root_orientation(ref_quat)
                ref_features.append(ref_orient)

        # Flatten reference features
        if ref_features:
            return np.concatenate(ref_features)
        else:
            return np.array([])

    def get_state_dim(self, trajectory) -> int:
        """Get the dimensionality of extracted states."""
        # Extract a single state to determine dimensionality
        dummy_states, _, _ = self.extract_from_trajectory(trajectory)
        return dummy_states.shape[1]

    def get_action_dim(self, trajectory) -> int:
        """Get the dimensionality of extracted actions."""
        return self._get_action_dim(trajectory)

    def get_state_feature_info(self) -> Dict[str, Any]:
        """Get information about state features."""
        return {
            'include_joint_pos': self.include_joint_pos,
            'include_joint_vel': self.include_joint_vel,
            'include_root_pos': self.include_root_pos,
            'include_root_orient': self.include_root_orient,
            'include_root_lin_vel': self.include_root_lin_vel,
            'include_root_ang_vel': self.include_root_ang_vel,
            'include_previous_action': self.include_previous_action,
            'include_reference_features': self.include_reference_features,
            'reference_horizon': self.reference_horizon,
            'reference_features': self.reference_features,
            'root_orient_repr': self.root_orient_repr,
            'state_history': self.state_history
        }

    def get_action_feature_info(self) -> Dict[str, Any]:
        """Get information about action features."""
        return {
            'action_type': self.action_type,
            'action_lookahead': self.action_lookahead,
            'action_delta_scale': self.action_delta_scale,
            'clip_actions': self.clip_actions,
            'action_clip_range': self.action_clip_range
        }


class MinimalStateActionExtractor(StateActionExtractor):
    """
    Minimal extractor with only essential features for fast training.
    """

    def __init__(self, config: DictConfig):
        # Override config for minimal setup
        minimal_config = OmegaConf.create({
            'state': {
                'include_joint_pos': True,
                'include_joint_vel': True,
                'include_root_pos': False,
                'include_root_orient': False,
                'include_root_lin_vel': False,
                'include_root_ang_vel': False,
                'include_previous_action': False,
                'include_reference_features': False
            },
            'action': {
                'type': 'joint_positions'
            },
            'action_lookahead': 1,
            'state_history': 1
        })

        # Merge with user config
        config = OmegaConf.merge(minimal_config, config)
        super().__init__(config)


class FullStateActionExtractor(StateActionExtractor):
    """
    Full-featured extractor with all available state information.
    """

    def __init__(self, config: DictConfig):
        # Override config for full setup
        full_config = OmegaConf.create({
            'state': {
                'include_joint_pos': True,
                'include_joint_vel': True,
                'include_root_pos': True,
                'include_root_orient': True,
                'include_root_lin_vel': True,
                'include_root_ang_vel': True,
                'include_previous_action': True,
                'include_reference_features': True,
                'reference_horizon': 10,
                'reference_features': ['joint_pos', 'root_pos'],
                'root_orient_repr': 'quat',
                'relative_root_pos': True
            },
            'action': {
                'type': 'joint_positions'
            },
            'action_lookahead': 1,
            'state_history': 1
        })

        # Merge with user config
        config = OmegaConf.merge(full_config, config)
        super().__init__(config)