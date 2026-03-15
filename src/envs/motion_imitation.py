"""
Motion imitation environment for Unitree G1.
Implements reference motion tracking tasks for behavior cloning.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .g1_env import G1Environment
from ..core.config import Config
from ..core.utils import normalize_angle, angle_difference, quaternion_to_euler


class MotionImitationEnv(G1Environment):
    """
    Environment for motion imitation learning.

    This environment provides:
    - Reference motion loading and playback
    - Motion tracking rewards
    - Phase-based motion progression
    - Early termination for failed tracking
    """

    def __init__(self, config: Config):
        super().__init__(config)

        self.task_config = config.env.task
        self.reward_config = config.env.rewards

        # Motion data configuration
        self.reference_data_path = Path(self.task_config.reference_data_path)
        self.motion_length = self.task_config.get("motion_length", 10.0)
        self.early_termination = self.task_config.get("early_termination", True)
        self.termination_height = self.task_config.get("termination_height", 0.3)

        # Load motion library
        self._load_motion_library()

        # Current motion tracking
        self.current_motion = None
        self.motion_time = 0.0
        self.motion_phase = 0.0
        self.reference_traj = None

        # Tracking error history
        self.tracking_errors = []
        self.max_tracking_error = 0.2  # radians

        # Update observation space to include reference motion
        self._update_observation_space()

    def _load_motion_library(self):
        """Load reference motion data."""
        # TODO: Implement motion data loading
        # This would load preprocessed AMASS data or custom motion clips

        # For now, create a simple walking motion as placeholder
        self.motion_library = {
            "walk_forward": self._create_placeholder_walk(),
            "walk_backward": self._create_placeholder_walk(backward=True),
            "idle": self._create_placeholder_idle(),
        }

        self.motion_names = list(self.motion_library.keys())
        print(f"Loaded {len(self.motion_names)} reference motions")

    def _create_placeholder_walk(self, backward: bool = False) -> Dict[str, np.ndarray]:
        """Create a simple placeholder walking motion."""
        duration = 2.0  # seconds
        freq = 50  # Hz
        num_frames = int(duration * freq)

        time_steps = np.linspace(0, duration, num_frames)

        # Create simple sinusoidal walking pattern
        motion_data = {
            "time": time_steps,
            "joint_positions": np.zeros((num_frames, self.num_actuated_joints)),
            "joint_velocities": np.zeros((num_frames, self.num_actuated_joints)),
            "base_position": np.zeros((num_frames, 3)),
            "base_orientation": np.tile([1, 0, 0, 0], (num_frames, 1)),  # quaternion
            "base_linear_velocity": np.zeros((num_frames, 3)),
            "base_angular_velocity": np.zeros((num_frames, 3)),
            "foot_contacts": np.zeros((num_frames, 2)),  # left, right foot
        }

        # Simple walking parameters
        step_freq = 1.0  # Hz
        step_length = 0.3 if not backward else -0.3

        for i, t in enumerate(time_steps):
            # Hip oscillation for walking
            hip_angle = 0.3 * np.sin(2 * np.pi * step_freq * t)
            knee_angle = max(0, 0.6 * np.sin(2 * np.pi * step_freq * t))

            # Set joint positions (simplified pattern)
            # Assuming specific joint ordering - this would need to match actual G1
            motion_data["joint_positions"][i, 0] = hip_angle    # left hip pitch
            motion_data["joint_positions"][i, 3] = knee_angle   # left knee
            motion_data["joint_positions"][i, 6] = -hip_angle   # right hip pitch
            motion_data["joint_positions"][i, 9] = knee_angle   # right knee

            # Base motion
            motion_data["base_position"][i, 0] = step_length * t  # forward motion
            motion_data["base_linear_velocity"][i, 0] = step_length

            # Foot contacts (alternating)
            phase = (t * step_freq) % 1.0
            if phase < 0.5:
                motion_data["foot_contacts"][i] = [1, 0]  # left foot down
            else:
                motion_data["foot_contacts"][i] = [0, 1]  # right foot down

        return motion_data

    def _create_placeholder_idle(self) -> Dict[str, np.ndarray]:
        """Create placeholder idle/standing motion."""
        duration = 5.0
        freq = 50
        num_frames = int(duration * freq)

        time_steps = np.linspace(0, duration, num_frames)

        motion_data = {
            "time": time_steps,
            "joint_positions": np.zeros((num_frames, self.num_actuated_joints)),
            "joint_velocities": np.zeros((num_frames, self.num_actuated_joints)),
            "base_position": np.zeros((num_frames, 3)),
            "base_orientation": np.tile([1, 0, 0, 0], (num_frames, 1)),
            "base_linear_velocity": np.zeros((num_frames, 3)),
            "base_angular_velocity": np.zeros((num_frames, 3)),
            "foot_contacts": np.ones((num_frames, 2)),  # both feet down
        }

        # Slight swaying motion
        for i, t in enumerate(time_steps):
            sway = 0.05 * np.sin(0.5 * t)
            motion_data["joint_positions"][i, 12] = sway  # torso roll

        return motion_data

    def _update_observation_space(self):
        """Update observation space to include reference motion observations."""
        # Base observation dimension
        base_obs_dim = self.observation_space.shape[0]

        # Reference motion observations
        ref_obs_dim = (
            self.num_actuated_joints * 2  # target joint pos + vel
            + 7  # target base pose (pos + quat)
            + 6  # target base velocity (lin + ang)
            + 2  # foot contact targets
            + 1  # motion phase
            + 1  # motion ID (one-hot would be better but simplified)
        )

        total_obs_dim = base_obs_dim + ref_obs_dim

        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and select new reference motion."""
        obs, info = super().reset(seed=seed, options=options)

        # Select random motion
        motion_name = self.np_random.choice(self.motion_names)
        self.current_motion = motion_name
        self.reference_traj = self.motion_library[motion_name]

        # Reset motion tracking
        self.motion_time = 0.0
        self.motion_phase = 0.0
        self.tracking_errors.clear()

        # Get updated observation with reference motion
        observation = self._get_observation()
        info.update({
            "motion_name": motion_name,
            "motion_length": len(self.reference_traj["time"]),
        })

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step with motion tracking."""
        # Update motion time
        self.motion_time += self.dt * self.frame_skip

        # Get current reference state
        ref_state = self._get_reference_state()

        # Execute base step
        observation, reward, terminated, truncated, info = super().step(action)

        # Compute motion tracking reward
        tracking_reward = self._compute_tracking_reward(ref_state)
        reward = tracking_reward

        # Check tracking-based termination
        if self.early_termination:
            tracking_terminated = self._check_tracking_termination(ref_state)
            terminated = terminated or tracking_terminated

        # Update motion phase
        motion_duration = self.reference_traj["time"][-1]
        self.motion_phase = min(1.0, self.motion_time / motion_duration)

        # Check if motion completed
        if self.motion_phase >= 1.0:
            truncated = True

        info.update({
            "motion_phase": self.motion_phase,
            "motion_time": self.motion_time,
            "tracking_error": self.tracking_errors[-1] if self.tracking_errors else 0.0,
        })

        return observation, reward, terminated, truncated, info

    def _get_reference_state(self) -> Dict[str, np.ndarray]:
        """Get reference state at current time."""
        if self.reference_traj is None:
            raise ValueError("No reference trajectory loaded")

        time_steps = self.reference_traj["time"]

        # Find closest time index
        time_idx = np.searchsorted(time_steps, self.motion_time)
        time_idx = min(time_idx, len(time_steps) - 1)

        # Get reference state
        ref_state = {
            "joint_positions": self.reference_traj["joint_positions"][time_idx],
            "joint_velocities": self.reference_traj["joint_velocities"][time_idx],
            "base_position": self.reference_traj["base_position"][time_idx],
            "base_orientation": self.reference_traj["base_orientation"][time_idx],
            "base_linear_velocity": self.reference_traj["base_linear_velocity"][time_idx],
            "base_angular_velocity": self.reference_traj["base_angular_velocity"][time_idx],
            "foot_contacts": self.reference_traj["foot_contacts"][time_idx],
        }

        return ref_state

    def _get_observation(self) -> np.ndarray:
        """Get observation including reference motion."""
        # Get base observation
        base_obs = super()._get_observation()

        # Get reference state
        if self.reference_traj is not None:
            ref_state = self._get_reference_state()

            # Reference observations
            ref_obs_parts = [
                ref_state["joint_positions"],
                ref_state["joint_velocities"],
                ref_state["base_position"],
                ref_state["base_orientation"],
                ref_state["base_linear_velocity"],
                ref_state["base_angular_velocity"],
                ref_state["foot_contacts"],
                [self.motion_phase],
                [float(self.motion_names.index(self.current_motion))],  # Simple motion ID
            ]

            ref_obs = np.concatenate([np.atleast_1d(part).flatten() for part in ref_obs_parts])
        else:
            # Zero reference if no motion loaded
            ref_obs_dim = self.observation_space.shape[0] - base_obs.shape[0]
            ref_obs = np.zeros(ref_obs_dim)

        # Combine observations
        observation = np.concatenate([base_obs, ref_obs])

        return observation.astype(np.float32)

    def _compute_tracking_reward(self, ref_state: Dict[str, np.ndarray]) -> float:
        """Compute reward based on motion tracking performance."""
        reward = 0.0

        # Get current state
        current_joint_pos = self.get_joint_positions()
        current_joint_vel = self.get_joint_velocities()
        current_base_pose = self.get_base_pose()
        current_base_vel = self.get_base_velocity()

        # Joint position tracking
        joint_pos_error = np.mean(np.abs(current_joint_pos - ref_state["joint_positions"]))
        joint_pos_reward = np.exp(-10 * joint_pos_error)  # Exponential decay
        reward += self.reward_config.tracking_pos * joint_pos_reward

        # Joint velocity tracking
        joint_vel_error = np.mean(np.abs(current_joint_vel - ref_state["joint_velocities"]))
        joint_vel_reward = np.exp(-5 * joint_vel_error)
        reward += self.reward_config.tracking_vel * joint_vel_reward

        # Root position tracking
        root_pos_error = np.linalg.norm(current_base_pose[:3] - ref_state["base_position"])
        root_pos_reward = np.exp(-20 * root_pos_error)
        reward += self.reward_config.tracking_root * root_pos_reward

        # Overall tracking error for termination
        total_error = joint_pos_error + joint_vel_error + root_pos_error
        self.tracking_errors.append(total_error)

        # Energy penalty (from base class)
        # This is already included in base reward computation

        # Alive bonus
        reward += self.reward_config.get("alive", 0.1)

        return reward

    def _check_tracking_termination(self, ref_state: Dict[str, np.ndarray]) -> bool:
        """Check if episode should terminate due to poor tracking."""
        if len(self.tracking_errors) < 5:  # Need some history
            return False

        # Check recent tracking performance
        recent_errors = self.tracking_errors[-5:]
        avg_error = np.mean(recent_errors)

        if avg_error > self.max_tracking_error:
            return True

        return False

    def get_motion_library(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get loaded motion library."""
        return self.motion_library

    def set_reference_motion(self, motion_name: str):
        """Set specific reference motion."""
        if motion_name not in self.motion_library:
            raise ValueError(f"Motion '{motion_name}' not in library")

        self.current_motion = motion_name
        self.reference_traj = self.motion_library[motion_name]
        self.motion_time = 0.0
        self.motion_phase = 0.0